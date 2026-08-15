"""
modules/transcriber.py
Speech-to-text using Faster-Whisper. Multilingual by default —
auto-detects Arabic vs English (or anything else Whisper supports),
no separate config needed per language.
"""

from faster_whisper import WhisperModel

# "small" balances speed vs accuracy well for CPU; use "base" if this
# feels slow on your machine, or "medium"/"large-v3" if you have a GPU.
MODEL_SIZE = "medium"  # was "small" — meaningfully better Arabic accuracy

_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe_audio(file_path: str) -> dict:
    segments, info = _model.transcribe(
        file_path,
        beam_size=5,
        vad_filter=True,  # trims silence/noise at start/end, helps accuracy
    )
    text = " ".join(segment.text.strip() for segment in segments)
    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
    }