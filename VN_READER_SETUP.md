## Visual Novel VN Reader Integration

This project has been extended so the VTuber can **read and react** to dialogue
from a visual novel shown on your screen.

At a high level:

- A lightweight screen‑capture + OCR loop reads text from the bottom of your game.
- Each new line is injected into the normal conversation pipeline as a `text-input`.
- The message is prefixed with `"[GAME DIALOGUE]"`, and the updated persona prompt
  tells the VTuber to read the line out loud and then add a short reaction.

> This feature is **opt‑in** and controlled via an environment variable.

### 1. Install extra dependencies (inside the Open‑LLM‑VTuber env)

You need a few additional Python packages and the Tesseract OCR binary.

1. From the `Open-LLM-VTuber` folder, install Python deps (using `uv` or `pip`):

   ```bash
   # Using uv (recommended)
   uv add mss opencv-python pytesseract numpy Pillow

   # Or with pip, if you're not using uv
   pip install mss opencv-python pytesseract numpy Pillow
   ```

2. Install **Tesseract OCR** on Windows (for example, from the UB Mannheim build)
   and make sure `tesseract.exe` is on your `PATH`.

### 2. Enable the VN reader

Set an environment variable before starting the server:

```bash
set ENABLE_VN_READER=true          # Windows (cmd)
$env:ENABLE_VN_READER = "true"     # Windows (PowerShell)
```

When `ENABLE_VN_READER` is true:

- The backend starts a background `VNReader` (`src/open_llm_vtuber/vn_reader.py`).
- The newest connected client is treated as the **primary VN listener**.
- Each new OCR’d line triggers a synthetic `text-input` message for that client.

### 3. Default region and monitor

The VN reader uses the following defaults (see `VNReaderConfig`):

- `monitor_index = 1` – primary monitor (change if your game is on another display).
- `roi_rel = (0.05, 0.7, 0.95, 0.95)` – a band along the **bottom** of the screen,
  assuming the pink dialogue box lives there.
- `capture_interval_s = 0.4` – capture every ~400 ms.

You can tweak these in `src/open_llm_vtuber/vn_reader.py` if needed:

- Move `roi_rel` up/down or narrow/widen it until it wraps the dialogue box.
- Increase `min_change_chars` if noise causes false positives.

### 4. How it flows through the backend

- `src/open_llm_vtuber/vn_reader.py`:
  - Captures the configured monitor with `mss`.
  - Crops to `roi_rel`.
  - Runs Tesseract via `pytesseract` to get text.
  - Emits a `DialogueEvent` when the text changes.

- `src/open_llm_vtuber/websocket_handler.py`:
  - On first client connection, marks that client as `_vn_primary_client_uid`.
  - Lazily starts a single global `VNReader` (if `ENABLE_VN_READER` is true).
  - For each `DialogueEvent`, calls `_dispatch_vn_event(...)`, which:
    - Builds a `{"type": "text-input", "text": "[GAME DIALOGUE] ..."}` payload.
    - Routes it through `_handle_conversation_trigger(...)` so it behaves like normal text input.

- `conf.yaml`:
  - Persona prompt updated to explain how to react to `[GAME DIALOGUE]` messages.

### 5. Running end‑to‑end

1. Start your visual novel in a consistent resolution on the target monitor.
2. In another terminal, from `Open-LLM-VTuber`:

   ```bash
   uv run run_server.py --verbose
   ```

3. Open the VTuber UI in your browser, connect, and ensure:
   - Normal chatting works using Mistral + Edge TTS.
   - When the VN shows a new line in the bottom text box, after a short delay:
     - The VTuber reads the line.
     - Then adds a short, in‑character comment.

If OCR is off (missing words, wrong region), adjust `roi_rel` and `capture_interval_s`
in `vn_reader.py` and restart the server.

