"""
ARIA V3 - Piper English TTS Engine
Language: ENGLISH
Model: en_US-lessac-medium (ONNX Runtime, CPU)
Output: float32 PCM, 22050 Hz mono
VRAM: 0 MB (CPU only)
"""

import threading
import time
from pathlib import Path

import numpy as np

# Model paths
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PIPER_DIR    = _PROJECT_ROOT / "models" / "piper"
_ONNX_FILE    = _PIPER_DIR / "en_US-lessac-medium.onnx"
_JSON_FILE    = _PIPER_DIR / "en_US-lessac-medium.onnx.json"


class PiperEngine:
    """
    Thread-safe, lazy-loading Piper TTS engine for English.
    Usage:
        engine = PiperEngine()
        pcm, sr = engine.synthesize("Hello, how can I help?")
    """

    def __init__(self):
        self._voice = None
        self._sample_rate: int = 22050
        self._lock = threading.Lock()
        self._load_time_s: float = 0.0

    def ensure_loaded(self) -> None:
        """Load the Piper voice model if not already loaded (idempotent)."""
        if self._voice is not None:
            return
        with self._lock:
            if self._voice is not None:
                return
            self._load()

    def synthesize(self, text: str) -> tuple:
        """
        Synthesize English text to float32 PCM.

        Args:
            text: Plain text sentence (no markdown, no Arabic script).

        Returns:
            (pcm_float32_ndarray, sample_rate_hz)
        """
        self.ensure_loaded()
        if not text or not text.strip():
            return np.zeros(1, dtype=np.float32), self._sample_rate

        chunks = []
        for chunk in self._voice.synthesize(text):
            arr = np.asarray(chunk.audio_int16_array, dtype=np.int16)
            chunks.append(arr)

        if not chunks:
            return np.zeros(1, dtype=np.float32), self._sample_rate

        pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
        return pcm, self._sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def load_time_s(self) -> float:
        return self._load_time_s

    def is_loaded(self) -> bool:
        return self._voice is not None

    def _load(self) -> None:
        from piper import PiperVoice  # type: ignore

        if not _ONNX_FILE.exists():
            raise FileNotFoundError(
                f"Piper ONNX model not found: {_ONNX_FILE}\n"
                "Run: python scripts/download_piper.py"
            )
        if not _JSON_FILE.exists():
            raise FileNotFoundError(
                f"Piper config not found: {_JSON_FILE}\n"
                "Run: python scripts/download_piper.py"
            )

        t0 = time.perf_counter()
        self._voice = PiperVoice.load(
            str(_ONNX_FILE),
            config_path=str(_JSON_FILE),
            use_cuda=False,
        )
        self._load_time_s = time.perf_counter() - t0
        self._sample_rate = self._voice.config.sample_rate
