"""
ARIA Real-Time Audio Capture Manager
PortAudio WASAPI backend with non-blocking callback and bounded queue.
"""

import logging
import queue
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Callable
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

logger = logging.getLogger("aria.audio.capture")

@dataclass
class CaptureMetrics:
    total_frames_captured: int = 0
    echo_frames_discarded: int = 0
    audio_capture_overflow_count: int = 0
    max_callback_latency_ms: float = 0.0
    total_callback_latency_ms: float = 0.0
    callback_invocations: int = 0
    status_warnings: int = 0

    @property
    def mean_callback_latency_ms(self) -> float:
        if self.callback_invocations == 0:
            return 0.0
        return self.total_callback_latency_ms / self.callback_invocations


class AudioCaptureManager:
    """
    Manages real-time audio input capture via sounddevice.InputStream.
    Standardized for 16,000 Hz mono float32 PCM output in 512-sample frames (32 ms).
    """

    TARGET_SAMPLE_RATE: int = 16000
    TARGET_BLOCK_SIZE: int = 512  # 32 ms at 16 kHz
    MAX_QUEUE_SIZE: int = 50       # ~1.6 seconds of audio backpressure buffer

    def __init__(
        self,
        device: Optional[int] = None,
        latency: Optional[str] = "low",
        max_queue_size: int = MAX_QUEUE_SIZE,
    ):
        self.device = device
        self.latency = latency
        self.max_queue_size = max_queue_size
        self.queue: queue.Queue = queue.Queue(maxsize=self.max_queue_size)
        self.stream: Optional[sd.InputStream] = None
        self.is_listening: bool = True
        self.echo_gate_checker: Optional[Callable[[], bool]] = None
        self.metrics = CaptureMetrics()
        
        # Determine device capabilities & resampling necessity
        self.native_rate, self.hardware_blocksize, self.needs_resample = self._detect_device_config(self.device)

    def set_echo_gate_checker(self, checker: Optional[Callable[[], bool]]):
        """Register dynamic echo gate callback (e.g. state_manager.should_discard_audio_frame)."""
        self.echo_gate_checker = checker

    @classmethod
    def list_input_devices(cls) -> List[Dict[str, Any]]:
        """Enumerates all input-capable devices across all host APIs."""
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
        input_devs = []
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                api_name = hostapis[d["hostapi"]]["name"]
                input_devs.append({
                    "index": i,
                    "name": d["name"],
                    "hostapi": api_name,
                    "hostapi_index": d["hostapi"],
                    "max_input_channels": d["max_input_channels"],
                    "default_samplerate": d["default_samplerate"],
                    "low_latency_ms": d.get("default_low_input_latency", 0) * 1000.0,
                    "high_latency_ms": d.get("default_high_input_latency", 0) * 1000.0,
                })
        return input_devs

    @classmethod
    def find_wasapi_device(cls) -> Optional[int]:
        """Finds the default WASAPI input device if available."""
        hostapis = sd.query_hostapis()
        wasapi_api_idx = None
        for i, api in enumerate(hostapis):
            if "WASAPI" in api["name"].upper():
                wasapi_api_idx = i
                break
        if wasapi_api_idx is None:
            return None
        default_input = hostapis[wasapi_api_idx].get("default_input_device")
        if isinstance(default_input, int) and default_input >= 0:
            return default_input

        for index, device in enumerate(sd.query_devices()):
            if device.get("hostapi") == wasapi_api_idx and device.get("max_input_channels", 0) > 0:
                return index
        return None

    def _detect_device_config(self, device: Optional[int]) -> Tuple[int, int, bool]:
        """
        Determines whether the device accepts direct 16 kHz or requires native 48 kHz
        with polyphase decimation (resample_poly 1:3).
        """
        # Test direct 16 kHz capture
        try:
            with sd.InputStream(
                device=device,
                channels=1,
                samplerate=self.TARGET_SAMPLE_RATE,
                blocksize=self.TARGET_BLOCK_SIZE,
                dtype="float32",
                latency=self.latency,
            ):
                pass
            logger.info(f"Device {device}: Direct 16,000 Hz capture supported.")
            return self.TARGET_SAMPLE_RATE, self.TARGET_BLOCK_SIZE, False
        except Exception:
            # Direct 16 kHz rejected (typical of WASAPI hardware clocks)
            logger.info(f"Device {device}: Direct 16 kHz rejected. Testing 48,000 Hz with 1:3 decimation.")
            # 48000 Hz / 3 = 16000 Hz. 1536 samples / 3 = 512 samples.
            return 48000, 1536, True

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags):
        """
        Real-time PortAudio OS callback context.
        MUST NEVER: block, perform heavy allocations, or raise unhandled exceptions.
        """
        t0 = time.perf_counter()

        if status:
            self.metrics.status_warnings += 1

        # Application-level half-duplex turn gate & Echo Gate (Phase 10/11)
        if not self.is_listening or (self.echo_gate_checker and self.echo_gate_checker()):
            self.metrics.echo_frames_discarded += 1
            return

        try:
            # Flatten to 1D float32
            raw_frame = indata[:, 0] if indata.ndim > 1 else indata.flatten()

            if self.needs_resample:
                # Decimate 48 kHz (1536 samples) to 16 kHz (512 samples)
                pcm16k = resample_poly(raw_frame, 1, 3).astype(np.float32)
            else:
                pcm16k = raw_frame.copy().astype(np.float32)

            # Ensure exactly TARGET_BLOCK_SIZE samples
            if len(pcm16k) > self.TARGET_BLOCK_SIZE:
                pcm16k = pcm16k[:self.TARGET_BLOCK_SIZE]

            # Bounded non-blocking enqueue
            self.queue.put_nowait(pcm16k)
            self.metrics.total_frames_captured += 1

        except queue.Full:
            # Bounded overflow policy: drop newest frame and increment metric
            self.metrics.audio_capture_overflow_count += 1
        except Exception as e:
            # Never let an exception escape and terminate PortAudio stream
            logger.error(f"Callback exception: {e}")
        finally:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics.callback_invocations += 1
            self.metrics.total_callback_latency_ms += dur_ms
            if dur_ms > self.metrics.max_callback_latency_ms:
                self.metrics.max_callback_latency_ms = dur_ms

    def start(self):
        """Starts the audio input stream."""
        if self.stream is not None and self.stream.active:
            return

        devices_to_try = [self.device]
        default_input = sd.default.device[0]
        if isinstance(default_input, int) and default_input >= 0 and default_input not in devices_to_try:
            devices_to_try.append(default_input)
        try:
            for info in self.list_input_devices():
                device_index = info["index"]
                if device_index not in devices_to_try:
                    devices_to_try.append(device_index)
        except Exception as exc:
            logger.warning(f"Could not enumerate fallback input devices: {exc}")

        last_error = None
        for device in devices_to_try:
            stream = None
            try:
                stream = sd.InputStream(
                    device=device,
                    channels=1,
                    samplerate=self.native_rate,
                    blocksize=self.hardware_blocksize,
                    dtype="float32",
                    latency=self.latency,
                    callback=self._audio_callback,
                )
                stream.start()
                self.stream = stream
                self.device = device
                break
            except Exception as exc:
                last_error = exc
                try:
                    stream.close()
                except Exception:
                    pass
        else:
            raise last_error

        logger.info(
            f"Audio stream started on device {self.device} at {self.native_rate} Hz, "
            f"blocksize={self.hardware_blocksize}, latency={self.latency}, needs_resample={self.needs_resample}"
        )

    def stop(self):
        """Stops and closes the audio input stream."""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Error closing stream: {e}")
            finally:
                self.stream = None
        logger.info("Audio stream stopped.")

    def pause_listening(self):
        """Echo gate closed: discard incoming audio."""
        self.is_listening = False

    def resume_listening(self):
        """Echo gate open: accept incoming audio."""
        self.is_listening = True

    def get_frame(self, block: bool = True, timeout: Optional[float] = 0.5) -> Optional[np.ndarray]:
        """Retrieves a single 512-sample float32 frame (32 ms) from the capture queue."""
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def drain(self):
        """Empties the capture queue."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
