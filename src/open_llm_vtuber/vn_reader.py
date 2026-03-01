import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from loguru import logger
import mss
import numpy as np
import cv2
import pytesseract


# Don't trigger LLM more than once per this many seconds (avoids 429s, matches human speaking pace)
MIN_SECONDS_BETWEEN_VN_CALLS = 2.0


@dataclass
class VNReaderConfig:
    """Configuration for the visual novel reader."""

    monitor_index: int = 2
    capture_interval_s: float = 0.5  # Check twice per second
    roi_rel: Tuple[float, float, float, float] = (0.30, 0.45, 0.80, 0.70)
    min_change_chars: int = 8
    min_seconds_between_emits: float = MIN_SECONDS_BETWEEN_VN_CALLS


@dataclass
class DialogueEvent:
    text: str
    timestamp: float


class VNReader:
    """Simple screen-capture + OCR loop that emits dialogue text events."""

    def __init__(self, config: Optional[VNReaderConfig] = None) -> None:
        self.config = config or VNReaderConfig()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_text: str = ""
        self._last_emit_time: float = 0.0
        self._callback: Optional[Callable[[DialogueEvent], None]] = None

    def start(self, callback: Callable[[DialogueEvent], None]) -> None:
        """Start the capture loop in a background thread."""
        if self._running:
            return

        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the capture loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        with mss.mss() as sct:
            while self._running:
                start_time = time.time()
                img = self._capture_monitor(sct)
                if img is not None:
                    roi = self._crop_roi(img)
                    text = self._run_ocr(roi)
                    self._maybe_emit(text)

                elapsed = time.time() - start_time
                delay = max(0.0, self.config.capture_interval_s - elapsed)
                if delay > 0:
                    time.sleep(delay)

    def _capture_monitor(self, sct: mss.mss) -> Optional[np.ndarray]:
        monitors = sct.monitors
        if not monitors or self.config.monitor_index >= len(monitors):
            return None

        monitor = monitors[self.config.monitor_index]
        raw = sct.grab(monitor)
        img = np.array(raw)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _crop_roi(self, img: np.ndarray) -> np.ndarray:
        h, w, _ = img.shape
        x1_rel, y1_rel, x2_rel, y2_rel = self.config.roi_rel
        x1 = int(x1_rel * w)
        y1 = int(y1_rel * h)
        x2 = int(x2_rel * w)
        y2 = int(y2_rel * h)
        x1 = max(0, min(x1, w - 1))
        x2 = max(1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(1, min(y2, h))
        return img[y1:y2, x1:x2]

    def _run_ocr(self, roi: np.ndarray) -> str:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = "--psm 6"
        text = pytesseract.image_to_string(thresh, config=config)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return " ".join(lines)

    def _maybe_emit(self, text: str) -> None:
        if self._callback is None:
            return

        cleaned = text.strip()
        if not cleaned or len(cleaned) < self.config.min_change_chars:
            return

        if cleaned == self._last_text:
            return

        # Debounce: don't fire more than once per min_seconds_between_emits (avoids 429s)
        now = time.time()
        if now - self._last_emit_time < self.config.min_seconds_between_emits:
            logger.debug(
                "VN reader debounce: skipping emit (%.1fs since last)",
                now - self._last_emit_time,
            )
            return

        self._last_emit_time = now
        self._last_text = cleaned
        logger.info(
            "VN reader detected new dialogue: %s",
            cleaned[:60] + ("..." if len(cleaned) > 60 else ""),
        )
        event = DialogueEvent(text=cleaned, timestamp=now)
        self._callback(event)

