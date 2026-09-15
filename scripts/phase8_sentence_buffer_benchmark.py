"""
Phase 8 Benchmark Suite — Lookahead Sentence Boundary Buffer Validation
Tests:
1. Immediate first-sentence emission on boundary detection
2. Decimal protection (e.g. 3.14, 1.5)
3. Abbreviation protection (e.g. Dr., e.g., i.e.)
4. Streaming incremental feed latency (< 1 ms per chunk)
5. Multilingual and exclamation/question marks splitting (. ! ? \n)
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.llm.sentence_buffer import SentenceBuffer


def run_benchmark():
    print("=" * 65)
    print("  PHASE 8 -- LOOKAHEAD SENTENCE BOUNDARY BUFFER VALIDATION")
    print("=" * 65)

    buffer = SentenceBuffer()
    results = {}
    failures = []

    def record(tid: str, passed: bool, data: dict):
        results[tid] = {"passed": passed, **data}
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {tid}: {data.get('summary', '')}")
        if not passed:
            failures.append(tid)

    # -------------------------------------------------------------------
    # TEST 1: Immediate First-Sentence Emission
    # -------------------------------------------------------------------
    buffer.reset()
    stream1 = ["Hello ", "world! ", "This ", "is ", "sentence ", "two. "]
    emitted_at_step = []
    for step, chunk in enumerate(stream1):
        s = buffer.feed(chunk)
        if s:
            emitted_at_step.append((step, s))
    
    # Sentence 1 should emit at step 1 ("world! ") immediately, not waiting for step 5
    passed_t1 = (len(emitted_at_step) >= 1) and (emitted_at_step[0][0] == 1) and ("Hello world!" in emitted_at_step[0][1])
    record("T1_immediate_first_sentence", passed_t1, {
        "summary": "First sentence emitted immediately upon delimiter chunk",
        "emitted_at_step": str(emitted_at_step),
        "step_emitted": emitted_at_step[0][0] if emitted_at_step else None
    })

    # -------------------------------------------------------------------
    # TEST 2: Decimal & Number Protection
    # -------------------------------------------------------------------
    buffer.reset()
    stream2 = ["Pi is roughly 3.14159 in math. ", "Model size is 1.5B parameters."]
    out2 = []
    for chunk in stream2:
        out2.extend(buffer.feed(chunk))
    out2.extend(buffer.flush())

    passed_t2 = (len(out2) == 2) and ("3.14159" in out2[0]) and ("1.5B" in out2[1])
    record("T2_decimal_protection", passed_t2, {
        "summary": "Decimals 3.14159 and 1.5B protected from premature splitting",
        "sentences": out2,
        "count": len(out2)
    })

    # -------------------------------------------------------------------
    # TEST 3: Abbreviation Protection
    # -------------------------------------------------------------------
    buffer.reset()
    stream3 = ["Dr. Smith consulted with Prof. Jones yesterday. ", "Use e.g. this example."]
    out3 = []
    for chunk in stream3:
        out3.extend(buffer.feed(chunk))
    out3.extend(buffer.flush())

    passed_t3 = (len(out3) == 2) and ("Dr. Smith" in out3[0]) and ("e.g. this" in out3[1])
    record("T3_abbreviation_protection", passed_t3, {
        "summary": "Abbreviations Dr. and e.g. protected from splitting",
        "sentences": out3,
        "count": len(out3)
    })

    # -------------------------------------------------------------------
    # TEST 4: Multilingual & Varied Punctuation (. ! ? \n)
    # -------------------------------------------------------------------
    buffer.reset()
    stream4 = [
        "Aap kaisay hain? ",
        "Main bilkul theek hoon! ",
        "Weather update check karein.\n",
        "Subha 9 baje meeting hai."
    ]
    out4 = []
    for chunk in stream4:
        out4.extend(buffer.feed(chunk))
    out4.extend(buffer.flush())

    passed_t4 = (len(out4) == 4) and ("kaisay hain?" in out4[0]) and ("theek hoon!" in out4[1])
    record("T4_multilingual_punctuation", passed_t4, {
        "summary": "Correctly split on ?, !, \\n across Roman Urdu sentences",
        "sentences": out4,
        "count": len(out4)
    })

    # -------------------------------------------------------------------
    # TEST 5: Processing Latency Overhead (< 1 ms per chunk)
    # -------------------------------------------------------------------
    buffer.reset()
    bench_chunks = ["The ", "quick ", "brown ", "fox ", "jumps. ", "Over ", "the ", "lazy ", "dog! "] * 50
    times_us = []
    for chunk in bench_chunks:
        t0 = time.perf_counter()
        _ = buffer.feed(chunk)
        times_us.append((time.perf_counter() - t0) * 1_000_000)

    mean_us = sum(times_us) / len(times_us)
    p95_us = sorted(times_us)[int(len(times_us) * 0.95)]
    passed_t5 = (mean_us / 1000.0) < 1.0

    record("T5_latency_overhead", passed_t5, {
        "summary": f"Mean latency = {mean_us:.2f} µs ({mean_us/1000.0:.5f} ms), P95 = {p95_us:.2f} µs",
        "mean_overhead_us": round(mean_us, 2),
        "mean_overhead_ms": round(mean_us / 1000.0, 5),
        "p95_overhead_us": round(p95_us, 2),
        "chunks_tested": len(bench_chunks)
    })

    # -------------------------------------------------------------------
    # Save Report
    # -------------------------------------------------------------------
    total = len(results)
    passed_count = sum(1 for r in results.values() if r["passed"])
    all_passed = (passed_count == total)

    summary = {
        "subsystem": "sentence_buffer",
        "total_tests": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "all_passed": all_passed,
        "tests": results
    }

    os.makedirs("reports", exist_ok=True)
    out_json = os.path.join("reports", "phase8_sentence_buffer_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"  PHASE 8 SUMMARY: {passed_count}/{total} PASSED")
    print("=" * 65)
    return summary


if __name__ == "__main__":
    res = run_benchmark()
    sys.exit(0 if res["all_passed"] else 1)
