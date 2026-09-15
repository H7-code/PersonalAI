"""
ARIA V3 — Real-Time Resource Telemetry & Hardware Monitor
Implements Section 30 & Phase 13 Specifications:
- Samples Host CPU %, Process RAM, NVIDIA GPU VRAM (used/free), GPU Temp every 1.0s
- Thread-safe queue depth sampling with zero lock contention
- Ultra-low CPU overhead (< 0.5%)
- Direct WebSocket broadcast integration for the UI HUD
"""

import os
import threading
import time
from typing import Any, Callable, Dict, Optional

import psutil

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


class TelemetryMonitor:
    """
    Lightweight background monitor thread sampling host resources,
    GPU telemetry via NVML, and application queue depths every interval seconds.
    """

    def __init__(
        self,
        interval_seconds: float = 1.0,
        broadcast_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.interval = interval_seconds
        self.broadcast_callback = broadcast_callback
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid())
        self._gpu_handle = None

        if _NVML_AVAILABLE:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._gpu_handle = None

        self._latest_sample: Dict[str, Any] = {}
        self._sample_lock = threading.Lock()
        self._total_sampling_duration = 0.0
        self._sample_count = 0

    def sample_now(self) -> Dict[str, Any]:
        """Collect a single telemetry snapshot."""
        t0 = time.perf_counter()

        # 1. CPU & System Memory
        cpu_percent = psutil.cpu_percent(interval=None)
        proc_memory = self._process.memory_info()
        proc_ram_mb = proc_memory.rss / (1024 * 1024)

        sys_mem = psutil.virtual_memory()
        sys_ram_used_mb = (sys_mem.total - sys_mem.available) / (1024 * 1024)
        sys_ram_total_mb = sys_mem.total / (1024 * 1024)

        # 2. NVIDIA GPU VRAM & Temperature via NVML
        vram_used_mb = 0.0
        vram_free_mb = 0.0
        vram_total_mb = 0.0
        gpu_temp_c = 0
        gpu_util_pct = 0

        if self._gpu_handle:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                vram_used_mb = round(mem_info.used / (1024 * 1024), 1)
                vram_free_mb = round(mem_info.free / (1024 * 1024), 1)
                vram_total_mb = round(mem_info.total / (1024 * 1024), 1)
                gpu_temp_c = pynvml.nvmlDeviceGetTemperature(self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util_pct = util_info.gpu
            except Exception:
                pass

        # 3. Queue Depths & State Machine Snapshot (Zero Lock Contention)
        from src.state_manager import get_state_manager
        sm = get_state_manager()
        sm_snap = sm.get_snapshot()

        sample = {
            "timestamp": time.time(),
            "cpu_percent": round(cpu_percent, 1),
            "proc_ram_mb": round(proc_ram_mb, 1),
            "sys_ram_used_mb": round(sys_ram_used_mb, 1),
            "sys_ram_total_mb": round(sys_ram_total_mb, 1),
            "vram_used_mb": vram_used_mb,
            "vram_free_mb": vram_free_mb,
            "vram_total_mb": vram_total_mb,
            "vram_headroom_ok": (vram_free_mb >= 500),
            "gpu_temp_c": gpu_temp_c,
            "gpu_util_pct": gpu_util_pct,
            "pipeline_state": sm_snap["state"],
            "request_id": sm_snap["request_id"],
            "echo_gate_active": sm_snap["echo_gate_active"],
            "audio_queue_depth": sm_snap["audio_queue_depth"],
            "stt_queue_depth": sm_snap["stt_queue_depth"],
            "tts_queue_depth": sm_snap["tts_queue_depth"],
            "playback_queue_depth": sm_snap["playback_queue_depth"],
        }

        dur = time.perf_counter() - t0
        with self._sample_lock:
            self._latest_sample = sample
            self._total_sampling_duration += dur
            self._sample_count += 1

        return sample

    def _run_loop(self):
        while not self._stop_event.is_set():
            sample = self.sample_now()
            if self.broadcast_callback:
                try:
                    self.broadcast_callback(sample)
                except Exception:
                    pass
            self._stop_event.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="ARIA-TelemetryMonitor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_latest_sample(self) -> Dict[str, Any]:
        with self._sample_lock:
            return dict(self._latest_sample)

    def get_metrics(self, target_interval_seconds: float = 1.0) -> Dict[str, Any]:
        with self._sample_lock:
            mean_dur_ms = (
                (self._total_sampling_duration / self._sample_count) * 1000.0
                if self._sample_count > 0
                else 0.0
            )
            overhead_pct = (mean_dur_ms / (target_interval_seconds * 1000.0)) * 100.0
            return {
                "samples_collected": self._sample_count,
                "mean_sample_duration_ms": round(mean_dur_ms, 3),
                "estimated_cpu_overhead_pct": round(overhead_pct, 4),
            }


_monitor_instance: Optional[TelemetryMonitor] = None

def get_telemetry_monitor(interval_seconds: float = 1.0) -> TelemetryMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = TelemetryMonitor(interval_seconds=interval_seconds)
    return _monitor_instance
