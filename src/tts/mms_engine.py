"""
ARIA V3 - MMS Urdu/Minglish TTS Engine
Language: URDU, MINGLISH (Roman Urdu / Latin script)
Model: facebook/mms-tts-urd-script_latin (VITS, PyTorch CPU)
Output: float32 PCM, 16000 Hz mono
VRAM: 0 MB (CPU only)
License: CC BY-NC 4.0 (Non-Commercial use only)

CRITICAL: This engine must NEVER receive Arabic/Perso-Urdu Unicode (U+0600-U+06FF).
All input must be pre-sanitized by OutputGuard before reaching this engine.
"""

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Model path - pre-downloaded to local filesystem
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MMS_MODEL_DIR = _PROJECT_ROOT / "models" / "mms"

# Sample rate from config.json
MMS_SAMPLE_RATE: int = 16000


class MMSEngine:
    """
    Thread-safe, lazy-loading MMS-TTS engine for Roman Urdu / Minglish.

    Strictly CPU-only. Uses facebook/mms-tts-urd-script_latin loaded from
    local model directory (offline, zero cloud calls).

    Usage:
        engine = MMSEngine()
        pcm, sr = engine.synthesize("Aaj ka mausam bohot acha hai.")
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self._model_dir = model_dir or _MMS_MODEL_DIR
        self._model = None
        self._tokenizer = None
        self._sample_rate: int = MMS_SAMPLE_RATE
        self._lock = threading.Lock()
        self._load_time_s: float = 0.0

    def ensure_loaded(self) -> None:
        """Load the MMS model and tokenizer if not already loaded (idempotent)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._load()

    def synthesize(self, text: str) -> tuple:
        """
        Synthesize Roman Urdu / Minglish text to float32 PCM.

        Args:
            text: Plain Latin-script text. Must NOT contain Arabic Unicode
                  (U+0600-U+06FF). OutputGuard guarantees this upstream.

        Returns:
            (pcm_float32_ndarray, sample_rate_hz)
        """
        self.ensure_loaded()

        if not text or not text.strip():
            return np.zeros(1, dtype=np.float32), self._sample_rate

        # Guard: VITS attention mechanism crashes on punctuation-only inputs
        # (e.g. '?', '!', '.') because they tokenize to sequence length=1,
        # causing `length - 1 = 0` in relative position padding.
        # Any text with no alphabetic content is not speakable; return silence.
        if not any(ch.isalpha() for ch in text):
            return np.zeros(1, dtype=np.float32), self._sample_rate

        # Validate: no Arabic/Perso-Urdu script reaches MMS
        if any(0x0600 <= ord(ch) <= 0x06FF for ch in text):
            raise ValueError(
                "MMSEngine received Arabic/Perso-Urdu Unicode. "
                "OutputGuard must sanitize before TTS."
            )

        inputs = self._tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self._model(**inputs).waveform

        # output shape: (1, samples) or (samples,)
        waveform = output.squeeze().cpu().numpy().astype(np.float32)

        # Normalize to [-1.0, 1.0] if not already in range
        peak = np.abs(waveform).max()
        if peak > 1.0:
            waveform = waveform / peak

        return waveform, self._sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def load_time_s(self) -> float:
        return self._load_time_s

    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        from transformers import VitsModel, AutoTokenizer  # type: ignore

        if not self._model_dir.exists():
            raise FileNotFoundError(
                f"MMS model directory not found: {self._model_dir}\n"
                "Ensure models/mms/ contains model.safetensors and config.json."
            )

        t0 = time.perf_counter()

        # Load tokenizer and model from local directory (offline)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self._model_dir),
            local_files_only=True,
        )
        self._model = VitsModel.from_pretrained(
            str(self._model_dir),
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        self._model.eval()

        # Confirm CPU execution
        self._model = self._model.to("cpu")

        self._load_time_s = time.perf_counter() - t0
        self._sample_rate = self._model.config.sampling_rate
