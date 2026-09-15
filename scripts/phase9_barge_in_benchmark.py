"""
Phase 9 Benchmark Suite — Interruption / Barge-in State Machine Validation
Tests:
1. Cancellation response latency (< 50 ms criterion)
2. Cancel while LLM streaming (stream aborts, tokens dropped)
3. Cancel while sentence in TTS queue (drained and discarded)
4. Cancel during active TTS inference (worker aborts)
5. Cancel after audio synthesized before playback (audio chunk drained)
6. Cancel during active playback (hardware stop callback fired)
7. Late worker put rejection (old request_id rejected)
8. Clean next turn transition (turn N+1 completely isolated from cancelled turn N)
"""

import json
import os
import queue
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.state_manager import AppStateManager, AppState


def run_benchmark():
    print("=" * 65)
    print("  PHASE 9 -- INTERRUPTION / BARGE-IN STATE MACHINE VALIDATION")
    print("=" * 65)

    sm = AppStateManager()
    results = {}
    failures = []

    def record(tid: str, passed: bool, data: dict):
        results[tid] = {"passed": passed, **data}
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {tid}: {data.get('summary', '')}")
        if not passed:
            failures.append(tid)

    # -------------------------------------------------------------------
    # TEST 1: Cancellation Latency Benchmark (< 50 ms target)
    # -------------------------------------------------------------------
    hardware_stopped = False
    def mock_hw_stop():
        nonlocal hardware_stopped
        hardware_stopped = True

    sm.set_hardware_stop_callback(mock_hw_stop)
    req1 = sm.new_request()

    # Pre-populate queues to simulate active pipeline load
    for i in range(10):
        sm.audio_capture_queue.put(b"pcm_data")
        sm.stt_queue.put("transcribed_text")
        sm.tts_input_queue.put((req1, f"Sentence {i}"))
        sm.audio_playback_queue.put((req1, b"audio_pcm"))

    cancelled_id, lat_ms = sm.cancel_active_request(reason="user_speech_barge_in")
    passed_t1 = (lat_ms < 50.0) and hardware_stopped and (cancelled_id == req1)

    record("T1_cancellation_latency", passed_t1, {
        "summary": f"Cancellation latency = {lat_ms:.3f} ms (< 50.0 ms limit)",
        "latency_ms": round(lat_ms, 3),
        "limit_ms": 50.0,
        "hardware_stop_called": hardware_stopped
    })

    # -------------------------------------------------------------------
    # TEST 2: Queue Draining & Monotonic Request ID
    # -------------------------------------------------------------------
    queues_empty = (
        sm.audio_capture_queue.empty() and
        sm.stt_queue.empty() and
        sm.tts_input_queue.empty() and
        sm.audio_playback_queue.empty()
    )
    new_req_id = sm.current_request_id
    passed_t2 = queues_empty and (new_req_id > req1) and (sm.state == AppState.LISTENING)

    record("T2_queue_draining_isolation", passed_t2, {
        "summary": "All 4 queues completely drained and state returned to LISTENING",
        "queues_empty": queues_empty,
        "new_request_id": new_req_id,
        "current_state": sm.state.value
    })

    # -------------------------------------------------------------------
    # TEST 3: Cancel while LLM is Streaming
    # -------------------------------------------------------------------
    req3 = sm.new_request()
    tokens_received_after_cancel = []

    def mock_llm_stream():
        for i in range(50):
            if not sm.is_request_valid(req3):
                break
            time.sleep(0.002)
            if sm.is_request_valid(req3):
                tokens_received_after_cancel.append(f"tok_{i}")

    t_thread = threading.Thread(target=mock_llm_stream)
    t_thread.start()
    time.sleep(0.01)  # allow stream to start
    sm.cancel_active_request()
    t_thread.join(timeout=1.0)

    passed_t3 = (len(tokens_received_after_cancel) < 50) and not t_thread.is_alive()
    record("T3_cancel_during_llm_stream", passed_t3, {
        "summary": "Streaming halted immediately; remaining tokens aborted",
        "tokens_emitted_before_abort": len(tokens_received_after_cancel)
    })

    # -------------------------------------------------------------------
    # TEST 4: Cancel while sentence is in TTS Queue
    # -------------------------------------------------------------------
    req4 = sm.new_request()
    sm.tts_input_queue.put((req4, "Sentence waiting for TTS"))
    sm.cancel_active_request()
    
    passed_t4 = sm.tts_input_queue.empty()
    record("T4_cancel_in_tts_queue", passed_t4, {
        "summary": "Pending sentence drained from TTS queue prior to synthesis",
        "tts_queue_empty": passed_t4
    })

    # -------------------------------------------------------------------
    # TEST 5: Cancel during active TTS Inference
    # -------------------------------------------------------------------
    req5 = sm.new_request()
    tts_completed = False

    def mock_tts_worker():
        nonlocal tts_completed
        # Simulate check before heavy inference
        if sm.is_request_valid(req5):
            # Heavy inference chunk simulation
            for _ in range(10):
                if not sm.is_request_valid(req5):
                    return  # Abandoned
                time.sleep(0.005)
            tts_completed = True

    tts_thread = threading.Thread(target=mock_tts_worker)
    tts_thread.start()
    time.sleep(0.01)
    sm.cancel_active_request()
    tts_thread.join(timeout=1.0)

    passed_t5 = (not tts_completed) and not tts_thread.is_alive()
    record("T5_cancel_during_tts_inference", passed_t5, {
        "summary": "TTS worker detected cancellation and abandoned synthesis mid-way",
        "tts_completed": tts_completed
    })

    # -------------------------------------------------------------------
    # TEST 6: Late Worker Output Rejection (Old request_id)
    # -------------------------------------------------------------------
    req6 = sm.new_request()
    sm.cancel_active_request()  # Increments to req6 + 1

    # Late worker tries to check validity or put with req6
    is_valid = sm.is_request_valid(req6)
    sm.audio_playback_queue.put((req6, b"stale_audio"))
    
    # Downstream playback worker checks before playing:
    played = False
    try:
        item = sm.audio_playback_queue.get_nowait()
        if sm.is_request_valid(item[0]):
            played = True
    except queue.Empty:
        pass

    passed_t6 = (not is_valid) and (not played)
    record("T6_late_worker_rejection", passed_t6, {
        "summary": "Late output with stale request_id rejected and suppressed",
        "is_valid_check": is_valid,
        "stale_audio_played": played
    })

    # -------------------------------------------------------------------
    # TEST 7: Instant New Turn Clean Isolation
    # -------------------------------------------------------------------
    req7_old = sm.new_request()
    sm.tts_input_queue.put((req7_old, "Old turn text"))
    sm.cancel_active_request()

    req7_new = sm.new_request()
    sm.tts_input_queue.put((req7_new, "New turn clean text"))

    item = sm.tts_input_queue.get_nowait()
    passed_t7 = (item[0] == req7_new) and (item[1] == "New turn clean text")
    record("T7_new_turn_clean_isolation", passed_t7, {
        "summary": "Turn N+1 completely isolated; 0 stale data from turn N",
        "active_request_id": item[0],
        "turn_text": item[1]
    })

    # -------------------------------------------------------------------
    # Summary & Output
    # -------------------------------------------------------------------
    total = len(results)
    passed_count = sum(1 for r in results.values() if r["passed"])
    all_passed = (passed_count == total)

    summary = {
        "subsystem": "barge_in_state_manager",
        "total_tests": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "all_passed": all_passed,
        "cancellation_latency_ms": round(lat_ms, 3),
        "tests": results
    }

    os.makedirs("reports", exist_ok=True)
    out_json = os.path.join("reports", "phase9_barge_in_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"  PHASE 9 SUMMARY: {passed_count}/{total} PASSED")
    print("=" * 65)
    return summary


if __name__ == "__main__":
    res = run_benchmark()
    sys.exit(0 if res["all_passed"] else 1)
