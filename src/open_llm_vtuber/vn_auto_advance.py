"""
Auto-advance for visual novel: send a click or key after the VTuber finishes
reading the current dialogue so the game advances to the next line.
"""
import sys
from typing import Tuple

from loguru import logger

from .vn_reader import VNReaderConfig


def get_advance_click_position(config: VNReaderConfig) -> Tuple[int, int] | None:
    """
    Return (x, y) screen coordinates for the center of the VN dialogue ROI
    on the configured monitor, for use as auto-advance click target.
    """
    try:
        import mss
    except ImportError:
        return None

    with mss.mss() as sct:
        monitors = sct.monitors
        if not monitors or config.monitor_index >= len(monitors):
            return None
        mon = monitors[config.monitor_index]
        left, top, width, height = mon["left"], mon["top"], mon["width"], mon["height"]
        roi = config.roi_rel
        if hasattr(roi, "__iter__") and not isinstance(roi, tuple):
            roi = tuple(roi)
        x1_rel, y1_rel, x2_rel, y2_rel = roi
        cx_rel = (x1_rel + x2_rel) / 2.0
        cy_rel = (y1_rel + y2_rel) / 2.0
        x = left + int(cx_rel * width)
        y = top + int(cy_rel * height)
        return (x, y)


def send_click(x: int, y: int) -> bool:
    """
    Send a left mouse click at screen coordinates (x, y).
    Uses ctypes on Windows; no extra dependencies.
    Returns True if the click was sent, False otherwise.
    """
    if sys.platform != "win32":
        logger.warning("VN auto-advance (click) is only implemented on Windows. Install pyautogui for other OS.")
        return False

    try:
        import ctypes
        from ctypes import wintypes

        # Windows API constants
        INPUT_MOUSE = 0
        MOUSEEVENTF_MOVE = 0x0001
        MOUSEEVENTF_LEFTDOWN = 0x0002
        MOUSEEVENTF_LEFTUP = 0x0004
        MOUSEEVENTF_ABSOLUTE = 0x8000
        MOUSEEVENTF_VIRTUALDESK = 0x4000  # multi-monitor: use virtual screen

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        # Use virtual screen for multi-monitor (SM_* = 76,77,78,79)
        SM_XVIRTUALSCREEN = 76
        SM_YVIRTUALSCREEN = 77
        SM_CXVIRTUALSCREEN = 78
        SM_CYVIRTUALSCREEN = 79
        vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        if vw < 1 or vh < 1:
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)
            vx, vy, vw, vh = 0, 0, screen_w, screen_h

        # INPUT structure: type, union(mi, ki, hi)
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD),
            ]

        class INPUT_UNION(ctypes.Union):
            _fields_ = [
                ("mi", MOUSEINPUT),
                ("ki", KEYBDINPUT),
                ("hi", HARDWAREINPUT),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", wintypes.DWORD),
                ("union", INPUT_UNION),
            ]

        def _send_input(*inputs: INPUT) -> int:
            n = len(inputs)
            arr = (INPUT * n)(*inputs)
            return user32.SendInput(n, ctypes.byref(arr), ctypes.sizeof(INPUT))

        # Normalize to 0-65535 relative to virtual screen (correct for multi-monitor)
        x_rel = x - vx
        y_rel = y - vy
        nx = int(65535 * x_rel / max(1, vw - 1))
        ny = int(65535 * y_rel / max(1, vh - 1))
        nx = max(0, min(65535, nx))
        ny = max(0, min(65535, ny))

        # 1) Move to position (VIRTUALDESK = use virtual screen for multi-monitor)
        move = INPUT(
            type=INPUT_MOUSE,
            union=INPUT_UNION(
                mi=MOUSEINPUT(
                    dx=nx,
                    dy=ny,
                    mouseData=0,
                    dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                    time=0,
                    dwExtraInfo=None,
                )
            ),
        )
        # 2) Left down
        down = INPUT(
            type=INPUT_MOUSE,
            union=INPUT_UNION(
                mi=MOUSEINPUT(
                    dx=0, dy=0, mouseData=0,
                    dwFlags=MOUSEEVENTF_LEFTDOWN,
                    time=0, dwExtraInfo=None,
                )
            ),
        )
        # 3) Left up
        up = INPUT(
            type=INPUT_MOUSE,
            union=INPUT_UNION(
                mi=MOUSEINPUT(
                    dx=0, dy=0, mouseData=0,
                    dwFlags=MOUSEEVENTF_LEFTUP,
                    time=0, dwExtraInfo=None,
                )
            ),
        )

        _send_input(move)
        _send_input(down)
        _send_input(up)
        return True

    except Exception as e:
        logger.warning(f"VN auto-advance click failed: {e}")
        return False


def trigger_vn_advance(config: VNReaderConfig) -> None:
    """
    Trigger one advance action for the visual novel (click at dialogue ROI center).
    Uses the same monitor_index and roi_rel as the VN reader OCR. Safe to call from a thread.
    """
    roi = getattr(config, "roi_rel", (0.30, 0.45, 0.80, 0.70))
    if hasattr(roi, "__iter__") and not isinstance(roi, tuple):
        roi = tuple(roi)
    logger.info(
        "VN auto-advance: triggering click (monitor_index=%s roi_rel=%s)",
        getattr(config, "monitor_index", "?"),
        roi,
    )
    pos = get_advance_click_position(config)
    if pos is None:
        logger.warning(
            "VN auto-advance: could not get click position (check monitor_index and roi_rel)"
        )
        return
    x, y = pos
    if send_click(x, y):
        logger.info("VN auto-advance: sent click at (%s, %s)", x, y)
    else:
        logger.warning("VN auto-advance: send_click returned False")
