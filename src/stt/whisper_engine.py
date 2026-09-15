"""
ARIA Speech-to-Text Engine (Faster-Whisper CTranslate2, CPU-only INT8)
Implements runtime-derived thread allocation, initial prompt biasing, and Roman Urdu script normalization.
"""

import logging
import os
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

class WhisperEngine:
    """
    Faster-Whisper CTranslate2 STT Worker.
    Strictly CPU-only execution with INT8 quantization.
    """

    DEFAULT_INITIAL_PROMPT = None

    def __init__(
        self,
        model_size: str = "tiny",
        download_root: str = "models/whisper",
        cpu_threads: int = 4,
        compute_type: str = "int8",
        beam_size: int = 1,
    ):
        self.model_size = model_size
        self.download_root = str(Path(download_root))
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self.compute_type = compute_type

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
        )
