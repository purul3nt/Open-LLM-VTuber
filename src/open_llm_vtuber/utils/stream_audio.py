"""
Audio payload preparation for TTS output.
Supports PCM (no ffmpeg) and MP3 (requires ffmpeg) formats.
"""
import base64
import os
import io
import struct

from loguru import logger

# Log TTS audio backend at import (PCM = no ffmpeg; MP3 = needs ffmpeg)
if os.environ.get("OPEN_LLM_VTUBER_TTS_DEBUG"):
    try:
        import imageio_ffmpeg
        logger.info(f"TTS: ffmpeg available at {imageio_ffmpeg.get_ffmpeg_exe()}")
    except Exception:
        logger.warning("TTS: ffmpeg not available (imageio-ffmpeg). Use output_format: pcm_22050 to avoid ffmpeg.")

from ..agent.output_types import Actions
from ..agent.output_types import DisplayText


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """Convert raw PCM to WAV bytes (pure Python, no ffmpeg)."""
    num_samples = len(pcm_bytes) // (channels * (bits // 8))
    byte_rate = sample_rate * channels * (bits // 8)
    block_align = channels * (bits // 8)
    data_size = num_samples * block_align

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,   # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size,
    )
    return header + pcm_bytes


def _volumes_from_pcm(pcm_bytes: bytes, sample_rate: int, chunk_length_ms: int = 20) -> list:
    """Compute normalized RMS volumes from PCM (16-bit mono) using numpy."""
    import numpy as np
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    samples_float = samples.astype(np.float32) / 32768.0
    samples_per_chunk = int(sample_rate * chunk_length_ms / 1000)
    n_chunks = max(1, len(samples_float) // samples_per_chunk)
    volumes = []
    for i in range(n_chunks):
        start = i * samples_per_chunk
        end = min(start + samples_per_chunk, len(samples_float))
        chunk = samples_float[start:end]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        volumes.append(rms)
    max_vol = max(volumes) if volumes else 1.0
    if max_vol == 0:
        return [0.0] * len(volumes)
    return [v / max_vol for v in volumes]


def _prepare_from_pcm(audio_bytes: bytes, audio_format: str, chunk_length_ms: int) -> tuple[bytes, list]:
    """Handle PCM format (pcm_16000, pcm_22050, etc.) - no ffmpeg needed."""
    if not audio_format.startswith("pcm_"):
        raise ValueError(f"Expected pcm_* format, got {audio_format}")
    try:
        sample_rate = int(audio_format.split("_")[1])
    except (IndexError, ValueError):
        sample_rate = 22050
    wav_bytes = _pcm_to_wav(audio_bytes, sample_rate)
    volumes = _volumes_from_pcm(audio_bytes, sample_rate, chunk_length_ms)
    return wav_bytes, volumes


def _prepare_from_mp3_or_file(audio_bytes: bytes | None, audio_path: str | None, audio_format: str, chunk_length_ms: int) -> tuple[bytes, list]:
    """Handle MP3 or file path using pydub (requires ffmpeg)."""
    try:
        import imageio_ffmpeg
        from pydub import AudioSegment
        from pydub.utils import make_chunks
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise ValueError(
            f"TTS needs ffmpeg for MP3. Install: pip install imageio-ffmpeg. Or use pcm_22050 in conf.yaml. ({e})"
        ) from e
    except Exception as e:
        logger.warning(f"Could not set ffmpeg from imageio-ffmpeg: {e}. Using system ffmpeg.")

    from pydub import AudioSegment
    from pydub.utils import make_chunks

    try:
        if audio_bytes is not None:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
        else:
            audio = AudioSegment.from_file(audio_path)
        wav_bytes = audio.export(format="wav").read()
        chunks = make_chunks(audio, chunk_length_ms)
        volumes = [c.rms for c in chunks]
        max_vol = max(volumes) if volumes else 1.0
        volumes = [v / max_vol if max_vol else 0 for v in volumes]
        return wav_bytes, volumes
    except FileNotFoundError as e:
        raise ValueError(
            f"ffmpeg not found. Install: pip install imageio-ffmpeg. Or set ElevenLabs output_format to pcm_22050 in conf.yaml. ({e})"
        ) from e
    except Exception as e:
        raise ValueError(
            f"Audio conversion failed. Try setting output_format: pcm_22050 in conf.yaml (elevenlabs_tts) to avoid ffmpeg. ({e})"
        ) from e


def prepare_audio_payload(
    audio_path: str | None,
    chunk_length_ms: int = 20,
    display_text: DisplayText = None,
    actions: Actions = None,
    forwarded: bool = False,
    audio_bytes: bytes | None = None,
    audio_format: str = "mp3",
) -> dict[str, any]:
    """
    Prepares the audio payload for sending to a broadcast endpoint.
    Supports pcm_* (no ffmpeg) and mp3 (requires ffmpeg) formats.
    """
    if isinstance(display_text, DisplayText):
        display_text = display_text.to_dict()

    if not audio_path and not audio_bytes:
        return {
            "type": "audio",
            "audio": None,
            "volumes": [],
            "slice_length": chunk_length_ms,
            "display_text": display_text,
            "actions": actions.to_dict() if actions else None,
            "forwarded": forwarded,
        }

    try:
        if audio_format.startswith("pcm_"):
            wav_bytes, volumes = _prepare_from_pcm(audio_bytes, audio_format, chunk_length_ms)
        else:
            wav_bytes, volumes = _prepare_from_mp3_or_file(
                audio_bytes, audio_path, audio_format, chunk_length_ms
            )
    except ValueError as e:
        logger.error(f"TTS audio conversion failed: {e}")
        raise

    audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")
    return {
        "type": "audio",
        "audio": audio_base64,
        "volumes": volumes,
        "slice_length": chunk_length_ms,
        "display_text": display_text,
        "actions": actions.to_dict() if actions else None,
        "forwarded": forwarded,
    }
