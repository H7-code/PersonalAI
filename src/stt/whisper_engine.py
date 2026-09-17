"""
ARIA Speech-to-Text Engine (Faster-Whisper CTranslate2, CPU-only INT8)
Implements runtime-derived thread allocation, initial prompt biasing, and Roman Urdu script normalization.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
from faster_whisper import WhisperModel

from src.stt.transliteration import transliterate_to_roman_urdu, is_perso_arabic

logger = logging.getLogger("aria.stt.whisper")

@dataclass
class STTResult:
    text: str
    raw_text: str
    language: str
    language_probability: float
    duration_ms: float
    latency_ms: float
    rtf: float
    transliterated: bool
    accepted: bool = True
    selected_language: str = ""
    retry_used: bool = False

class WhisperEngine:
    """
    Faster-Whisper CTranslate2 STT Worker.
    Strictly CPU-only execution with INT8 quantization.
    """

    DEFAULT_INITIAL_PROMPT = None
    ALLOWED_LANGUAGES = {"en", "ur"}
    UNSUPPORTED_SCRIPT_RE = re.compile(r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0600-\u06FF]")

    def __init__(
        self,
        model_size: str = "tiny",
        download_root: str = "models/whisper",
        cpu_threads: int = 4,
        compute_type: str = "int8",
        beam_size: int = 1,
        fallback_model_size: Optional[str] = None,
    ):
        self.model_size = model_size
        self.download_root = str(Path(download_root))
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self.compute_type = compute_type
        self.fallback_model_size = fallback_model_size
        self._fallback_engine: Optional["WhisperEngine"] = None

        logger.info(f"Loading Production STT: Faster-Whisper '{self.model_size}' (INT8, CPU, threads={self.cpu_threads})...")
        t0 = time.perf_counter()
        self.model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type=self.compute_type,
            download_root=self.download_root,
            cpu_threads=self.cpu_threads,
            num_workers=1,
        )
        self.load_duration_s = time.perf_counter() - t0
        logger.info(f"Faster-Whisper '{self.model_size}' loaded in {self.load_duration_s:.2f}s.")

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = DEFAULT_INITIAL_PROMPT,
        temperature: float = 0.0,
    ) -> STTResult:
        """
        Transcribes 16 kHz float32 audio ndarray or path to WAV.
        Production settings: beam_size=1, best_of=1, condition_on_previous_text=False, word_timestamps=False, language=None.
        Standardizes output to Latin script (Roman Urdu for Urdu speech).
        """
        t0 = time.perf_counter()

        segments, info = self.model.transcribe(
            audio,
            beam_size=self.beam_size,
            best_of=1,
            language=language,
            initial_prompt=initial_prompt,
            temperature=temperature,
            condition_on_previous_text=False,
            word_timestamps=False,
            no_speech_threshold=0.6,
            vad_filter=False,  # VAD already handled upstream by Silero VAD
        )

        raw_text = " ".join(s.text.strip() for s in segments).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        audio_array = np.asarray(audio, dtype=np.float32) if isinstance(audio, np.ndarray) else None
        rms = float(np.sqrt(np.mean(audio_array ** 2))) if audio_array is not None and audio_array.size else 0.0
        peak = float(np.max(np.abs(audio_array))) if audio_array is not None and audio_array.size else 0.0
        logger.info(
            "STT raw boundary model=%s audio_duration_ms=%.1f sample_rate=16000 "
            "channel_count=1 rms=%.6f peak=%.6f language=%s "
            "language_probability=%.4f raw_text=%r",
            self.model_size,
            info.duration * 1000.0 if info.duration else 0.0,
            rms,
            peak,
            info.language,
            info.language_probability,
            raw_text,
        )

        if self._should_use_fallback(info.language, info.language_probability, raw_text):
            fallback = self._get_fallback_engine()
            logger.info(
                "STT fallback start primary_model=%s fallback_model=%s "
                "language=%s language_probability=%.4f raw_text=%r",
                self.model_size,
                fallback.model_size,
                info.language,
                info.language_probability,
                raw_text,
            )
            fallback_result = fallback.transcribe(
                audio,
                language=language,
                initial_prompt=initial_prompt,
                temperature=temperature,
            )
            logger.info(
                "STT fallback result primary_model=%s fallback_model=%s "
                "raw_text=%r language=%s language_probability=%.4f latency_ms=%.1f",
                self.model_size,
                fallback.model_size,
                fallback_result.raw_text,
                fallback_result.language,
                fallback_result.language_probability,
                fallback_result.latency_ms,
            )
            return fallback_result

        if not self._is_accepted(info.language, raw_text):
            logger.warning(
                "STT rejected model=%s detected_language=%s language_probability=%.4f "
                "raw_text=%r; retrying forced en/ur",
                self.model_size,
                info.language,
                info.language_probability,
                raw_text,
            )
            candidates = []
            for forced_language in ("en", "ur"):
                retry = self._transcribe_forced(audio, forced_language, initial_prompt, temperature)
                if retry is not None:
                    candidates.append(retry)
            accepted = [candidate for candidate in candidates if candidate.accepted]
            if accepted:
                best = max(accepted, key=lambda candidate: candidate.language_probability)
                best.retry_used = True
                return best
            return self._rejected_result(audio, info, raw_text, latency_ms)

        # Determine audio duration
        duration_ms = info.duration * 1000.0 if info.duration else 0.0
        if duration_ms == 0.0 and isinstance(audio, np.ndarray):
            duration_ms = len(audio) / 16.0  # 16 samples per ms at 16 kHz

        rtf = (latency_ms / duration_ms) if duration_ms > 0 else 0.0

        # Script normalization (Roman Urdu conversion if output is Perso-Arabic)
        transliterated = False
        if is_perso_arabic(raw_text):
            final_text = transliterate_to_roman_urdu(raw_text)
            transliterated = True
        else:
            final_text = raw_text

        return STTResult(
            text=final_text,
            raw_text=raw_text,
            language=info.language,
            language_probability=info.language_probability,
            duration_ms=duration_ms,
            latency_ms=latency_ms,
            rtf=rtf,
            transliterated=transliterated,
            accepted=True,
            selected_language=info.language,
            retry_used=False,
        )

    def _is_accepted(self, language: str, raw_text: str) -> bool:
        return language in self.ALLOWED_LANGUAGES and not self.UNSUPPORTED_SCRIPT_RE.search(raw_text)

    def _transcribe_forced(
        self,
        audio: Union[str, np.ndarray],
        language: str,
        initial_prompt: Optional[str],
        temperature: float,
    ) -> Optional[STTResult]:
        if language not in self.ALLOWED_LANGUAGES:
            return None
        segments, info = self.model.transcribe(
            audio,
            beam_size=self.beam_size,
            best_of=1,
            language=language,
            initial_prompt=initial_prompt,
            temperature=temperature,
            condition_on_previous_text=False,
            word_timestamps=False,
            no_speech_threshold=0.6,
            vad_filter=False,
        )
        raw_text = " ".join(segment.text.strip() for segment in segments).strip()
        accepted = self._is_accepted(language, raw_text)
        logger.info(
            "STT constrained retry model=%s selected_language=%s detected_language=%s "
            "language_probability=%.4f raw_text=%r accepted=%s",
            self.model_size,
            language,
            info.language,
            info.language_probability,
            raw_text,
            accepted,
        )
        if not accepted:
            return None
        return self._build_result(audio, raw_text, language, float(info.language_probability), True)

    def _build_result(
        self,
        audio: Union[str, np.ndarray],
        raw_text: str,
        language: str,
        probability: float,
        retry_used: bool,
    ) -> STTResult:
        duration_ms = len(audio) / 16.0 if isinstance(audio, np.ndarray) else 0.0
        final_text = raw_text
        transliterated = False
        if is_perso_arabic(raw_text):
            final_text = transliterate_to_roman_urdu(raw_text)
            transliterated = True
        return STTResult(
            text=final_text,
            raw_text=raw_text,
            language=language,
            language_probability=probability,
            duration_ms=duration_ms,
            latency_ms=0.0,
            rtf=0.0,
            transliterated=transliterated,
            accepted=True,
            selected_language=language,
            retry_used=retry_used,
        )

    def _rejected_result(self, audio, info, raw_text: str, latency_ms: float) -> STTResult:
        duration_ms = info.duration * 1000.0 if info.duration else (len(audio) / 16.0 if isinstance(audio, np.ndarray) else 0.0)
        logger.warning(
            "STT rejected model=%s detected_language=%s language_probability=%.4f "
            "raw_text=%r accepted=false retry_used=true",
            self.model_size,
            info.language,
            info.language_probability,
            raw_text,
        )
        return STTResult(
            text="",
            raw_text=raw_text,
            language=info.language,
            language_probability=float(info.language_probability),
            duration_ms=duration_ms,
            latency_ms=latency_ms,
            rtf=(latency_ms / duration_ms) if duration_ms else 0.0,
            transliterated=False,
            accepted=False,
            selected_language="",
            retry_used=True,
        )

    def _should_use_fallback(self, language: str, probability: float, raw_text: str) -> bool:
        """Retry non-English tiny decodes that commonly mislabel Roman Urdu; keep English fast."""
        if not self.fallback_model_size or self.model_size == self.fallback_model_size:
            return False
        if not raw_text:
            return False
        return language not in {"en", "ur"}

    def _get_fallback_engine(self) -> "WhisperEngine":
        if self._fallback_engine is None:
            self._fallback_engine = WhisperEngine(
                self.fallback_model_size,
                download_root=self.download_root,
                cpu_threads=self.cpu_threads,
                compute_type=self.compute_type,
                beam_size=self.beam_size,
            )
        return self._fallback_engine
