"""Listening: speech to text on this computer (Whisper base, Apache-2.0, 290 MB).

Studio records the microphone only while the button is held and sends a
16 kHz mono WAV; nothing is uploaded anywhere.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any

import numpy as np

from reverse3d.config import Settings

logger = logging.getLogger(__name__)

SPEECH_MODEL = "openai/whisper-base"
#: model id -> (kind, licence); every entry must be commercial-friendly.
ALLOWED_LISTENERS: dict[str, tuple[str, str]] = {SPEECH_MODEL: ("speech", "Apache-2.0")}
RATE = 16_000
MAX_SECONDS = 30  # one Whisper window: a spoken request, not a dictation


def read_wav(path: Path) -> np.ndarray:
    """Mono float32 samples at 16 kHz from a PCM WAV."""
    with wave.open(str(path), "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(min(w.getnframes(), rate * MAX_SECONDS))
    if width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported.")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != RATE and len(audio):
        positions = np.linspace(0, len(audio) - 1, int(len(audio) * RATE / rate))
        audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
    return audio


class Ears:
    """Loads Whisper once; transcribe() turns a spoken request into text."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from reverse3d.assistant.brain import LOAD_LOCK

        with LOAD_LOCK:  # never import transformers while the language model loads
            self._load_locked()

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        from reverse3d.offline import resolve_model

        path = resolve_model(SPEECH_MODEL, self.settings)
        # Local files of the pinned revision only (resolve_model checked them).
        self._processor = WhisperProcessor.from_pretrained(str(path), local_files_only=True)  # nosec B615
        model: Any = WhisperForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True
        )  # nosec B615
        self._model = model.eval()

    def transcribe(self, wav: Path, lang: str = "es") -> str:
        import torch

        audio = read_wav(wav)
        if len(audio) < RATE // 4 or float(np.abs(audio).max(initial=0.0)) < 0.01:
            return ""  # silence or a click
        self._load()
        features = self._processor(audio, sampling_rate=RATE, return_tensors="pt").input_features
        with torch.no_grad():
            ids = self._model.generate(
                features,
                language="spanish" if lang == "es" else "english",
                task="transcribe",
                max_new_tokens=120,
            )
        return str(self._processor.batch_decode(ids, skip_special_tokens=True)[0]).strip()
