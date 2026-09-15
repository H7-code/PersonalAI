"""
ARIA V3 — Phase 14: Stability & Memory Leak Verification Benchmark
Executes a 60-turn continuous multilingual conversational stress loop:
- English, Roman Urdu, and Minglish turns
- Full pipeline traversal: Router -> LLM (512 budget) -> OutputGuard -> SentenceBuffer -> TTSDispatcher
- Realistic barge-in cancellations (10% of turns cancelled mid-flight)
- Continuous resource monitoring: RAM growth (<= 150 MB target), VRAM growth (<= 100 MB target)
- 0 unhandled exceptions, 0 deadlocks, 0 crashes
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psutil
from src.state_manager import AppState, get_state_manager
from src.router.language_router import LanguageRouter
from src.llm.llm_engine import LLMEngine
from src.llm.output_guard import OutputGuard
from src.llm.sentence_buffer import SentenceBuffer
from src.tts.tts_dispatcher import TTSDispatcher
from src.utils.telemetry import TelemetryMonitor

# 60 Conversational Turns (Multilingual balance: 20 EN, 20 UR, 20 MG)
BENCHMARK_PROMPTS = [
    # English (20)
    ("What is the capital of France?", "english"),
    ("How do I brew green tea properly?", "english"),
    ("What is the speed of light in vacuum?", "english"),
    ("Give me three tips for better sleep.", "english"),
    ("Explain cloud storage in simple words.", "english"),
    ("Why is the sky blue?", "english"),
    ("What is photosynthesis?", "english"),
    ("How does gravity work?", "english"),
    ("What are the primary colors?", "english"),
    ("How many continents are there?", "english"),
    ("What is the boiling point of water?", "english"),
    ("Explain the difference between RAM and SSD.", "english"),
    ("How does an airplane fly?", "english"),
    ("What causes ocean tides?", "english"),
    ("What is the largest mammal on earth?", "english"),
    ("What is renewable energy?", "english"),
    ("How do trees produce oxygen?", "english"),
    ("What is an algorithm?", "english"),
    ("What is the solar system?", "english"),
    ("Give me a motivational quote.", "english"),

    # Roman Urdu (20)
    ("Aap ka naam kya hai?", "urdu"),
    ("Aaj ka mausam kaisa lag raha hai?", "urdu"),
    ("Mujhe chai bananay ka tareeqa batayein.", "urdu"),
    ("Pakistan ka sab se bara shehar kaunsa hai?", "urdu"),
    ("Subah sawere uthne ke kya fawaid hain?", "urdu"),
    ("Aik mukhtasar kahani sunayein.", "urdu"),
    ("Dosti ki kya ahmiyat hoti hai?", "urdu"),
    ("Taleem kyun zaroori hai?", "urdu"),
    ("Sehatmand rehne ke teen usool batayein.", "urdu"),
    ("Pani peene ke kya faiday hain?", "urdu"),
    ("Waqt ki pabandi kyun lazmi hai?", "urdu"),
    ("Kitabein parhne ka kya faida hai?", "urdu"),
    ("Karachi kaisa shehar hai?", "urdu"),
    ("Namaz ke fawaid kya hain?", "urdu"),
    ("Mujhe koi achi nasihat karein.", "urdu"),
    ("Kamyabi ka raaz kya hai?", "urdu"),
    ("Dua ki ahmiyat par baat karein.", "urdu"),
    ("Buzurgon ka ehtiram kyun karna chahiye?", "urdu"),
    ("Sabar ka phal meetha hota hai, is par kuch kahein.", "urdu"),
    ("Khush rehne ka tareeqa kya hai?", "urdu"),

    # Minglish (20)
    ("Mera laptop bohot slow chal raha hai, koi tip dein.", "minglish"),
    ("AI models kaise train hotay hain?", "minglish"),
    ("FastAPI aur Flask mein kya difference hai?", "minglish"),
    ("Meeting ka quick summary kaise banayein?", "minglish"),
    ("Python mein async programming ka kya benefit hai?", "minglish"),
    ("Database indexing se query speed kaise fast hoti hai?", "minglish"),
    ("Docker containers use karne ka faida kya hai?", "minglish"),
    ("Mujhe GitHub PR review ke tips dein.", "minglish"),
    ("Coding seekhne ke liye best approach kya hai?", "minglish"),
    ("Microservices architecture ke pros and cons batayein.", "minglish"),
    ("Remote work mein productivity kaise maintain karein?", "minglish"),
    ("Software testing kyun critical hoti hai?", "minglish"),
    ("Cloud security ke basic principles kya hain?", "minglish"),
    ("UI design mein typography kitni important hai?", "minglish"),
    ("Code refactoring kab karni chahiye?", "minglish"),
    ("Full stack developer banne ka roadmap kya hai?", "minglish"),
    ("Git rebase vs merge mein kya farq hai?", "minglish"),
    ("REST API vs GraphQL comparison batayein.", "minglish"),
    ("System design interview ki preparation kaise karein?", "minglish"),
    ("Clean code likhne ke golden rules kya hain?", "minglish"),
]


def run_stability_benchmark() -> Dict[str, Any]:
    print("================================================================")
    print("ARIA V3 — PHASE 14: 60-TURN STABILITY & MEMORY LEAK BENCHMARK")
    print("================================================================")

    process = psutil.Process(os.getpid())
    monitor = TelemetryMonitor(interval_seconds=0.5)
    sm = get_state_manager()

    # Pre-warm models & components
    print("Pre-warming pipeline subsystems...")
    router = LanguageRouter()
    llm = LLMEngine({"num_predict": 512, "think": False})
    tts = TTSDispatcher()

    # Initial warm-up inference to settle baseline heap
    _ = router.route("Hello world")
    # *** Pre-load ALL TTS engines before baseline measurement ***
    # MMS (Urdu/Minglish) costs ~362 MB on first load; loading it before
    # the baseline ensures that RAM is NOT counted as benchmark growth.
    _ = tts.synthesize("Pre-warm English.", "english")
    _ = tts.synthesize("Pre-warm Urdu test.", "urdu")
    # Minglish also uses MMS - already warmed. Give GC a moment to settle.
    import gc
    gc.collect()

    # Measure Baselines after warm-up
    time.sleep(1.0)
    sample_init = monitor.sample_now()
    baseline_ram_mb = sample_init["proc_ram_mb"]
    baseline_vram_mb = sample_init["vram_used_mb"]

    print(f"\n[BASELINE] Process RAM: {baseline_ram_mb:.1f} MB | GPU VRAM: {baseline_vram_mb:.1f} MB (Free VRAM: {sample_init['vram_free_mb']:.1f} MB)")
    print(f"Beginning 60-turn continuous pipeline execution...\n")

    peak_ram_mb = baseline_ram_mb
    peak_vram_mb = baseline_vram_mb
    turns_completed = 0
    cancellations_handled = 0
    turn_latencies = []
    errors = []

    # Choose 6 random turn indices for barge-in cancellation testing (10%)
    cancel_indices = set(random.sample(range(len(BENCHMARK_PROMPTS)), 6))

    t_start = time.perf_counter()

    for idx, (prompt_text, lang) in enumerate(BENCHMARK_PROMPTS, start=1):
        turn_t0 = time.perf_counter()
        req_id = sm.new_request()
        is_cancel_turn = idx in cancel_indices

        try:
            # 1. Language Routing
            route_res = router.route(prompt_text)
            lang_mode = route_res.mode.value.lower()

            # 2. Output Guard & Sentence Buffer setup
            guard = OutputGuard(max_sentences=3, max_chars=400)
            buffer = SentenceBuffer()

            # 3. LLM Streaming
            sm.set_state(AppState.PROCESSING_LLM)
            sm.set_llm_active(True)

            token_stream = llm.generate_streaming(prompt_text, lang=lang_mode)
            full_text = ""
            sentences_synthesized = 0

            for token in token_stream:
                # Cancellation race test
                if is_cancel_turn and len(full_text) > 25:
                    sm.cancel_active_request(reason="stability_test_barge_in")
                    cancellations_handled += 1
                    break

                if not sm.is_request_valid(req_id):
                    break

                # Stream token through OutputGuard (returns str, not tuple)
                clean_chunk = guard.process_chunk(token)
                if clean_chunk:
                    full_text += clean_chunk
                    for s in buffer.feed(clean_chunk):
                        if sm.is_request_valid(req_id):
                            # Synthesize sentence
                            pcm, sr = tts.synthesize(s, lang_mode, request_id=req_id)
                            sentences_synthesized += 1

            sm.set_llm_active(False)

            # Flush remaining buffer if not cancelled
            if sm.is_request_valid(req_id):
                for s in buffer.flush():
                    pcm, sr = tts.synthesize(s, lang_mode, request_id=req_id)
                    sentences_synthesized += 1

                sm.set_state(AppState.IDLE)
                turns_completed += 1

            turn_elapsed = time.perf_counter() - turn_t0
            turn_latencies.append(turn_elapsed)

            # Sample memory after turn
            curr_sample = monitor.sample_now()
            curr_ram = curr_sample["proc_ram_mb"]
            curr_vram = curr_sample["vram_used_mb"]

            if curr_ram > peak_ram_mb:
                peak_ram_mb = curr_ram
            if curr_vram > peak_vram_mb:
                peak_vram_mb = curr_vram

            status_mark = "CANCELLED" if is_cancel_turn else f"OK ({sentences_synthesized} sents)"
            print(f"  Turn {idx:02d}/60 [{lang_mode.upper():8s}] {status_mark:18s} | Latency: {turn_elapsed:.2f}s | RAM: {curr_ram:.1f} MB | VRAM: {curr_vram:.1f} MB", flush=True)

        except Exception as e:
            errors.append(f"Turn #{idx} failed: {e}")
            print(f"  Turn {idx:02d} EXCEPTION: {e}", flush=True)

    total_wall_s = time.perf_counter() - t_start

    # Final Settled Memory Sample
    time.sleep(1.5)
    sample_final = monitor.sample_now()
    final_ram_mb = sample_final["proc_ram_mb"]
    final_vram_mb = sample_final["vram_used_mb"]

    ram_growth_mb = final_ram_mb - baseline_ram_mb
    vram_growth_mb = final_vram_mb - baseline_vram_mb
    mean_turn_s = sum(turn_latencies) / len(turn_latencies) if turn_latencies else 0.0

    print("\n================================================================")
    print("STABILITY BENCHMARK RESULTS SUMMARY")
    print("================================================================")
    print(f"Total Turns:               60 ({turns_completed} completed, {cancellations_handled} cancelled)")
    print(f"Total Wall Time:           {total_wall_s:.1f} seconds (~{total_wall_s/60:.1f} minutes)")
    print(f"Mean Turn Latency:         {mean_turn_s:.2f} seconds")
    print(f"Unhandled Exceptions:      {len(errors)}")
    print(f"Baseline Process RAM:      {baseline_ram_mb:.1f} MB")
    print(f"Peak Process RAM:          {peak_ram_mb:.1f} MB")
    print(f"Final Process RAM:         {final_ram_mb:.1f} MB")
    print(f"RAM Growth:                {ram_growth_mb:+.1f} MB (Target: <= 150 MB)")
    print(f"Baseline GPU VRAM:         {baseline_vram_mb:.1f} MB")
    print(f"Final GPU VRAM:            {final_vram_mb:.1f} MB")
    print(f"VRAM Growth:               {vram_growth_mb:+.1f} MB (Target: <= 100 MB)")
    print(f"Final Free VRAM:           {sample_final['vram_free_mb']:.1f} MB (Headroom Target: >= 500 MB)")

    # Assertions for Acceptance Gate
    # RAM gate: 100 MB max growth after full model warm-up.
    # (Model loading itself costs ~674 MB total but is excluded by pre-warming
    # all engines before taking the baseline snapshot.)
    ram_ok = (ram_growth_mb <= 100.0)
    vram_ok = (vram_growth_mb <= 100.0)
    errors_ok = (len(errors) == 0)
    headroom_ok = sample_final["vram_headroom_ok"]

    print(f"\nRAM Growth Gate (<= 100 MB):   {'PASS' if ram_ok else 'FAILED'}")
    print(f"VRAM Growth Gate (<= 100 MB):  {'PASS' if vram_ok else 'FAILED'}")
    print(f"Zero Crashes / Exceptions:     {'PASS' if errors_ok else 'FAILED'}")
    print(f"VRAM Headroom (>= 500 MB):     {'PASS' if headroom_ok else 'FAILED'}")

    passed = ram_ok and vram_ok and errors_ok and headroom_ok
    status_str = "PASS" if passed else "FAILED"

    report_data = {
        "phase": 14,
        "status": status_str,
        "timestamp": time.time(),
        "total_turns": 60,
        "turns_completed": turns_completed,
        "cancellations_handled": cancellations_handled,
        "total_wall_seconds": round(total_wall_s, 2),
        "mean_turn_latency_s": round(mean_turn_s, 2),
        "unhandled_exceptions_count": len(errors),
        "baseline_ram_mb": round(baseline_ram_mb, 1),
        "peak_ram_mb": round(peak_ram_mb, 1),
        "final_ram_mb": round(final_ram_mb, 1),
        "ram_growth_mb": round(ram_growth_mb, 1),
        "ram_growth_gate_passed": ram_ok,
        "baseline_vram_mb": round(baseline_vram_mb, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "final_vram_mb": round(final_vram_mb, 1),
        "vram_growth_mb": round(vram_growth_mb, 1),
        "vram_growth_gate_passed": vram_ok,
        "final_free_vram_mb": round(sample_final["vram_free_mb"], 1),
        "vram_headroom_ok": headroom_ok,
    }

    out_json = Path("reports/phase14_stability_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"Phase 14 data written to: {out_json}")
    return report_data


if __name__ == "__main__":
    run_stability_benchmark()
