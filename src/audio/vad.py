"""
ARIA Silero VAD v5 ONNX Engine & Speech Segment Assembler
Two-stage pipeline: RMS Energy Pre-Filter + Silero VAD v5 ONNX State Machine
"""

import enum
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("aria.audio.vad")

class VADState(enum.Enum):
    SILENCE = "SILENCE"
    SPEECH_ONSET_CANDIDATE = "SPEECH_ONSET_CANDIDATE"
    SPEAKING = "SPEAKING"
    SPEECH_OFFSET_HOLD = "SPEECH_OFFSET_HOLD"

@dataclass
class SpeechSegment:
    audio: np.ndarray
    duration_ms: float
    start_frame_idx: int
    end_frame_idx: int
    max_speech_prob: float
    mean_rms_dbfs: float
    frames_count: int

@dataclass
class VADMetrics:
    total_frames_processed: int = 0
    energy_bypassed_frames: int = 0
    onnx_inferences: int = 0
    total_inference_time_ms: float = 0.0
    max_inference_time_ms: float = 0.0
    speech_segments_emitted: int = 0
    rejected_transients: int = 0

    @property
    def mean_inference_time_ms(self) -> float:
        if self.onnx_inferences == 0:
            return 0.0
        return self.total_inference_time_ms / self.onnx_inferences

    @property
    def bypass_ratio(self) -> float:
        if self.total_frames_processed == 0:
            return 0.0
        return self.energy_bypassed_frames / self.total_frames_processed


class SileroVAD:
    """
    Silero VAD v5 ONNX Real-Time Voice Activity Detector.
    Strictly CPU execution via ONNX Runtime.
    """

    FRAME_SIZE: int = 512          # 32 ms at 16,000 Hz
    SAMPLE_RATE: int = 16000

    def __init__(
        self,
        model_path: str = "models/vad/silero_vad.onnx",
        energy_threshold_dbfs: float = -45.0,  # ~0.0056 linear amplitude
        speech_prob_threshold: float = 0.50,
        onset_consecutive_frames: int = 3,    # 3 * 32 ms = 96 ms (< 100 ms target)
        silence_padding_ms: float = 600.0,    # 600 ms silence padding before offset
        min_speech_duration_ms: float = 250.0,# reject speech < 250 ms
        max_speech_duration_s: float = 15.0,  # hard cap at 15 seconds
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # Fallback to models/silero_vad.onnx
            alt = Path("models/silero_vad.onnx")
            if alt.exists():
                self.model_path = alt
            else:
                raise FileNotFoundError(f"Silero VAD model not found at {model_path} or {alt}")

        # Enforce CPU execution
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.energy_threshold_dbfs = energy_threshold_dbfs
        self.speech_prob_threshold = speech_prob_threshold
        self.onset_consecutive_frames = onset_consecutive_frames
        self.silence_padding_frames = int(np.ceil(silence_padding_ms / 32.0))  # ~19 frames
        self.min_speech_frames = int(np.ceil(min_speech_duration_ms / 32.0))   # ~8 frames
        self.max_speech_frames = int(np.ceil(max_speech_duration_s * 1000.0 / 32.0)) # ~469 frames

        self.metrics = VADMetrics()
        self.reset_state()

    def reset_state(self):
        """Resets the internal ONNX recurrent state and speech assembler state."""
        self.onnx_state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.sr_tensor = np.array(self.SAMPLE_RATE, dtype=np.int64)
        self.state = VADState.SILENCE
        self.active_frames: List[np.ndarray] = []
        self.active_probs: List[float] = []
        self.active_rms: List[float] = []
        self.speech_onset_counter: int = 0
        self.silence_hold_counter: int = 0
        self.current_frame_idx: int = 0
        self.speech_start_frame_idx: int = 0

    @staticmethod
    def calculate_rms(frame: np.ndarray) -> Tuple[float, float]:
        """Returns (rms_linear, rms_dbfs)."""
        rms_linear = float(np.sqrt(np.mean(frame ** 2)))
        dbfs = float(20.0 * np.log10(rms_linear + 1e-9))
        return rms_linear, dbfs

    def step(self, frame: np.ndarray) -> Tuple[float, bool, Optional[SpeechSegment]]:
        """
        Processes a single 512-sample float32 frame (32 ms at 16 kHz).
        Returns:
            prob: speech probability [0.0 - 1.0]
            is_speech: boolean threshold decision
            segment: completed SpeechSegment if speech offset / max duration occurred, else None
        """
        self.metrics.total_frames_processed += 1
        self.current_frame_idx += 1

        if len(frame) != self.FRAME_SIZE:
            raise ValueError(f"Frame length {len(frame)} != required {self.FRAME_SIZE}")

        rms_linear, rms_dbfs = self.calculate_rms(frame)

        # Stage 1: Energy Pre-Filter (RMS floor)
        # If quiet ambient room noise < energy threshold, bypass ONNX inference!
        if rms_dbfs < self.energy_threshold_dbfs and self.state == VADState.SILENCE:
            self.metrics.energy_bypassed_frames += 1
            prob = 0.0
            is_speech = False
            self.context = frame[-64:].reshape(1, 64).astype(np.float32)
        else:
            # Stage 2: Silero VAD v5 ONNX Inference with 64-sample context
            t0 = time.perf_counter()
            frame_tensor = frame.reshape(1, 512).astype(np.float32)
            inp = np.concatenate([self.context, frame_tensor], axis=1)  # 576 samples
            out, new_state = self.session.run(
                None,
                {
                    "input": inp,
                    "state": self.onnx_state,
                    "sr": self.sr_tensor,
                },
            )
            inf_dur = (time.perf_counter() - t0) * 1000.0
            self.metrics.onnx_inferences += 1
            self.metrics.total_inference_time_ms += inf_dur
            if inf_dur > self.metrics.max_inference_time_ms:
                self.metrics.max_inference_time_ms = inf_dur

            prob = float(out[0, 0])
            self.onnx_state = new_state
            self.context = inp[:, -64:]
            is_speech = prob >= self.speech_prob_threshold

        # Update Speech Segment Assembler State Machine
        completed_segment: Optional[SpeechSegment] = None

        if self.state == VADState.SILENCE:
            if is_speech:
                self.speech_onset_counter = 1
                self.state = VADState.SPEECH_ONSET_CANDIDATE
                self.active_frames = [frame.copy()]
                self.active_probs = [prob]
                self.active_rms = [rms_dbfs]
                self.speech_start_frame_idx = self.current_frame_idx
            else:
                self.speech_onset_counter = 0

        elif self.state == VADState.SPEECH_ONSET_CANDIDATE:
            self.active_frames.append(frame.copy())
            self.active_probs.append(prob)
            self.active_rms.append(rms_dbfs)

            if is_speech:
                self.speech_onset_counter += 1
                if self.speech_onset_counter >= self.onset_consecutive_frames:
                    # Confirmed onset! Transition to SPEAKING
                    self.state = VADState.SPEAKING
                    self.silence_hold_counter = 0
            else:
                # Glitch / transient non-speech burst during onset candidate
                self.speech_onset_counter = 0
                self.state = VADState.SILENCE
                self.active_frames = []
                self.active_probs = []
                self.active_rms = []

        elif self.state == VADState.SPEAKING:
            self.active_frames.append(frame.copy())
            self.active_probs.append(prob)
            self.active_rms.append(rms_dbfs)

            # Check maximum duration cap (15.0 s)
            if len(self.active_frames) >= self.max_speech_frames:
                logger.info("Maximum speech segment cap reached (15.0 s). Emitting segment.")
                completed_segment = self._emit_segment()
                self.reset_state()
                return prob, is_speech, completed_segment

            if not is_speech:
                self.state = VADState.SPEECH_OFFSET_HOLD
                self.silence_hold_counter = 1
            else:
                self.silence_hold_counter = 0

        elif self.state == VADState.SPEECH_OFFSET_HOLD:
            self.active_frames.append(frame.copy())
            self.active_probs.append(prob)
            self.active_rms.append(rms_dbfs)

            # Check max cap
            if len(self.active_frames) >= self.max_speech_frames:
                completed_segment = self._emit_segment()
                self.reset_state()
                return prob, is_speech, completed_segment

            if is_speech:
                # Interrupted silence: user resumed speaking within 600 ms window
                self.state = VADState.SPEAKING
                self.silence_hold_counter = 0
            else:
                self.silence_hold_counter += 1
                if self.silence_hold_counter >= self.silence_padding_frames:
                    # Confirmed speech offset! (600 ms of silence)
                    if len(self.active_frames) >= self.min_speech_frames:
                        completed_segment = self._emit_segment()
                    else:
                        # Rejected as transient below 250 ms
                        self.metrics.rejected_transients += 1
                    self.reset_state()

        return prob, is_speech, completed_segment

    def _emit_segment(self) -> SpeechSegment:
        """Assembles active frames into SpeechSegment."""
        audio_concat = np.concatenate(self.active_frames).astype(np.float32)
        dur_ms = len(audio_concat) / self.SAMPLE_RATE * 1000.0
        max_prob = max(self.active_probs) if self.active_probs else 0.0
        mean_rms = float(np.mean(self.active_rms)) if self.active_rms else -100.0
        self.metrics.speech_segments_emitted += 1
        return SpeechSegment(
            audio=audio_concat,
            duration_ms=dur_ms,
            start_frame_idx=self.speech_start_frame_idx,
            end_frame_idx=self.current_frame_idx,
            max_speech_prob=max_prob,
            mean_rms_dbfs=mean_rms,
            frames_count=len(self.active_frames),
        )
