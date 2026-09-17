"""
ARIA V3 - Multilingual TTS Dispatcher
Routes synthesize requests to the correct engine based on lang_mode.

Language modes:
  "english"  -> PiperEngine  (en_US-lessac-medium, 22050 Hz)
    "urdu"     -> MMSEngine    (mms-tts-urd-script_latin, 16000 Hz)

Architecture constraints:
  - Both engines run CPU-only (0 VRAM)
  - request_id checked before synthesis to support barge-in cancellation
  - Raises ValueError on unknown lang_mode (no silent fallback)
  - Engines loaded lazily on first use
"""

import time
import logging
from typing import Optional

import numpy as np

from src.tts.piper_engine import PiperEngine
from src.tts.mms_engine import MMSEngine

# Canonical language mode strings (matches src/router output)
LANG_ENGLISH  = "english"
LANG_URDU     = "urdu"

_VALID_MODES = {LANG_ENGLISH, LANG_URDU}
logger = logging.getLogger("aria.tts.dispatcher")
MMS_TARGET_RMS = 0.14
PCM_PEAK_LIMIT = 0.95


def _normalize_mms_pcm(pcm: np.ndarray) -> np.ndarray:
    """Raise quiet MMS speech toward Piper loudness without exceeding safe headroom."""
    waveform = np.asarray(pcm, dtype=np.float32)
    if waveform.size == 0:
        return waveform
    before_peak = float(np.max(np.abs(waveform)))
    before_rms = float(np.sqrt(np.mean(waveform * waveform)))
    if before_rms <= 1e-6:
        return waveform
    gain = MMS_TARGET_RMS / before_rms
    if before_peak * gain > PCM_PEAK_LIMIT:
        gain = PCM_PEAK_LIMIT / before_peak
    if gain <= 1.0:
        return waveform
    normalized = np.clip(waveform * gain, -PCM_PEAK_LIMIT, PCM_PEAK_LIMIT)
    logger.info(
        "MMS loudness normalization rms_before=%.6f peak_before=%.6f gain=%.3f "
        "rms_after=%.6f peak_after=%.6f",
        before_rms,
        before_peak,
        gain,
        float(np.sqrt(np.mean(normalized * normalized))),
        float(np.max(np.abs(normalized))),
    )
    return normalized


class TTSDispatcher:
    """
    Routes TTS requests to the appropriate engine for the given language mode.

    Engines are loaded lazily on first synthesize() call for that language
    to avoid loading unused models on startup.

    Usage (TTS worker thread):
        dispatcher = TTSDispatcher()
        # In worker loop:
        item = tts_input_queue.get()
        if item.request_id != state_manager.current_request_id:
            continue  # stale request, discard
        pcm, sr = dispatcher.synthesize(item.text, item.lang_mode)
    """

    def __init__(self):
        self._piper = PiperEngine()
        self._mms   = MMSEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        lang_mode: str,
        request_id: Optional[int] = None,
        current_request_id: Optional[int] = None,
    ) -> tuple:
        """
        Synthesize text to float32 PCM using the engine for lang_mode.

        Args:
            text: Pre-sanitized plain text sentence (from OutputGuard).
            lang_mode: One of 'english', 'urdu', 'minglish'.
            request_id: The request_id this sentence belongs to.
            current_request_id: Active request_id from AppStateManager.
                If provided and request_id != current_request_id, synthesis
                is skipped and (empty_array, 0) is returned.

        Returns:
            (pcm_float32_ndarray, sample_rate_hz)
            Returns (np.zeros(1), 0) if request cancelled or text is empty.

        Raises:
            ValueError: If lang_mode is not a known language mode.
        """
        # Stale request guard
        if (request_id is not None and
                current_request_id is not None and
                request_id != current_request_id):
            return np.zeros(1, dtype=np.float32), 0

        if not text or not text.strip():
            return np.zeros(1, dtype=np.float32), 0

        lang_mode = lang_mode.lower().strip()
        if lang_mode not in _VALID_MODES:
            raise ValueError(
                f"TTSDispatcher: Unknown lang_mode '{lang_mode}'. "
                f"Must be one of: {_VALID_MODES}"
            )

        if lang_mode == LANG_ENGLISH:
            pcm, sample_rate = self._piper.synthesize(text)
            engine_name = "piper"
        else:
            # Urdu and English/Urdu mixed speech use MMS Latin-script engine
            pcm, sample_rate = self._mms.synthesize(text)
            pcm = _normalize_mms_pcm(pcm)
            engine_name = "mms"
        logger.info(
            "TTS synthesized mode=%s engine=%s text=%r samples=%s sample_rate=%s "
            "dtype=%s rms=%.6f peak=%.6f",
            lang_mode,
            engine_name,
            text,
            len(pcm),
            sample_rate,
            pcm.dtype,
            float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0,
            float(np.max(np.abs(pcm))) if len(pcm) else 0.0,
        )
        return pcm, sample_rate

    def preload_english(self) -> float:
        """
        Preload the English Piper engine. Call on startup to reduce
        first-sentence latency. Returns load time in seconds.
        """
        t0 = time.perf_counter()
        self._piper.ensure_loaded()
        return time.perf_counter() - t0

    def preload_urdu(self) -> float:
        """
        Preload the MMS Urdu engine. Call on startup if warm load
        is needed. MMS load is slow (~2-4s), so consider lazy loading.
        Returns load time in seconds.
        """
        t0 = time.perf_counter()
        self._mms.ensure_loaded()
        return time.perf_counter() - t0

    def piper_sample_rate(self) -> int:
        return self._piper.sample_rate

    def mms_sample_rate(self) -> int:
        return self._mms.sample_rate

    def is_english_loaded(self) -> bool:
        return self._piper.is_loaded()

    def is_urdu_loaded(self) -> bool:
        return self._mms.is_loaded()
