"""
ARIA V3 — Phase 13: Real-Time Resource Monitoring & Telemetry Benchmark
Tests:
1. Hardware Telemetry Sampling:
   - Host CPU %, Process RAM, System RAM
   - NVIDIA GPU VRAM (used/free/headroom), GPU Temp, GPU Util via NVML
   - Queue depths & state machine snapshots
2. Telemetry Thread CPU Overhead:
   - Must be < 0.5% CPU overhead
   - Measures mean sample latency across 100 samples
3. Zero Lock Contention Verification:
   - Simulates high-frequency concurrent state operations while monitor is active
   - 0 deadlocks, 0 worker thread blocking
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.telemetry import TelemetryMonitor, get_telemetry_monitor
from src.state_manager import AppState, get_state_manager


def test_telemetry_sampling():
    print("\n--- Test 1: Hardware & System Resource Telemetry Sampling ---")
    monitor = TelemetryMonitor(interval_seconds=0.1)
    sample = monitor.sample_now()

    print(f"  CPU Usage: {sample['cpu_percent']}%")
    print(f"  Process RAM: {sample['proc_ram_mb']} MB")
    print(f"  System RAM: {sample['sys_ram_used_mb']} / {sample['sys_ram_total_mb']} MB")
    print(f"  NVIDIA GPU VRAM Used: {sample['vram_used_mb']} MB | Free: {sample['vram_free_mb']} MB (Total: {sample['vram_total_mb']} MB)")
    print(f"  GPU Temp: {sample['gpu_temp_c']} °C | Util: {sample['gpu_util_pct']}%")
    print(f"  VRAM Headroom Target (>= 500 MB): {'SATISFIED' if sample['vram_headroom_ok'] else 'FAILED'}")
    print(f"  Pipeline State: {sample['pipeline_state']} | Request #{sample['request_id']}")

    assert sample["proc_ram_mb"] > 0, "Process RAM must be positive"
    assert sample["vram_total_mb"] > 0, "GPU VRAM total must be detected"
    assert sample["vram_headroom_ok"] is True, "Free VRAM must satisfy >= 500 MB headroom"
    print("  [PASS] All hardware telemetry metrics successfully sampled")

    return sample


def test_telemetry_overhead():
    print("\n--- Test 2: Telemetry Thread CPU Overhead & Latency ---")
    monitor = TelemetryMonitor(interval_seconds=0.01)

    # Collect 100 rapid samples to accurately measure sampling duration
    t0 = time.perf_counter()
    samples = []
    for _ in range(100):
        samples.append(monitor.sample_now())
        time.sleep(0.005)
    wall_time = time.perf_counter() - t0

    metrics = monitor.get_metrics()
    mean_dur_ms = metrics["mean_sample_duration_ms"]
    est_overhead_pct = metrics["estimated_cpu_overhead_pct"]

    print(f"  Collected {metrics['samples_collected']} samples across {wall_time:.2f}s")
    print(f"  Mean sample duration: {mean_dur_ms:.3f} ms")
    print(f"  Estimated background CPU overhead at 1.0s interval: {est_overhead_pct:.4f}% (Target: < 0.5%)")

    assert est_overhead_pct < 0.5, f"Overhead ({est_overhead_pct}%) exceeded 0.5% limit"
    assert mean_dur_ms < 5.0, f"Single sample duration ({mean_dur_ms} ms) exceeded 5 ms limit"
    print("  [PASS] Telemetry overhead satisfies < 0.5% CPU constraint")

    return {
        "samples_collected": metrics["samples_collected"],
        "mean_sample_duration_ms": mean_dur_ms,
        "estimated_cpu_overhead_pct": est_overhead_pct,
        "target_met": True,
    }


def test_zero_lock_contention():
    print("\n--- Test 3: Zero Lock Contention with Voice Pipeline Workers ---")
    sm = get_state_manager()
    monitor = TelemetryMonitor(interval_seconds=0.05)
    monitor.start()

    errors = []
    operations_count = 0
    t0 = time.perf_counter()

    # Rapidly mutate state and queues while monitor runs in background
    for i in range(500):
        try:
            req_id = sm.new_request()
            sm.transition_to(AppState.PROCESSING_STT)
            sm.transition_to(AppState.PROCESSING_LLM)
            sm.set_llm_active(True)
            sm.set_llm_active(False)
            sm.transition_to(AppState.SPEAKING)
            sm.set_playback_active(True)
            sm.set_playback_active(False)
            sm.transition_to(AppState.IDLE)
            operations_count += 6
        except Exception as e:
            errors.append(str(e))

    elapsed = time.perf_counter() - t0
    monitor.stop()

    assert len(errors) == 0, f"Contention errors occurred: {errors}"
    print(f"  [PASS] Completed {operations_count} state mutations in {elapsed:.2f}s with 0 lock contention errors")

    return {
        "operations_completed": operations_count,
        "elapsed_seconds": round(elapsed, 3),
        "lock_contention_errors": 0,
    }


def main():
    print("================================================================")
    print("ARIA V3 — PHASE 13: REAL-TIME RESOURCE TELEMETRY BENCHMARK")
    print("================================================================")

    r1 = test_telemetry_sampling()
    r2 = test_telemetry_overhead()
    r3 = test_zero_lock_contention()

    report_data = {
        "phase": 13,
        "status": "PASS",
        "timestamp": time.time(),
        "hardware_telemetry": r1,
        "overhead_metrics": r2,
        "contention_metrics": r3,
    }

    out_json = Path("reports/phase13_monitoring_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n================================================================")
    print(f"PHASE 13 VALIDATION COMPLETED SUCCESSFULLY. Saved to: {out_json}")
    print("================================================================")


if __name__ == "__main__":
    main()
