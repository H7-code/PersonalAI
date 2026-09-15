"""
Phase 7 Benchmark Suite — Incremental Streaming Output Guard Validation
Validates:
1. Zero reasoning leakage (0% leak across pathological chunk boundaries)
2. Processing overhead per chunk (< 1 ms requirement)
3. Markdown stripping (headers, bold, italics, bullets, code blocks)
4. Arabic/Urdu Unicode detection ([\u0600-\u06FF]) and phonetic transliteration / retry triggering
5. Length clamping (max 400 characters, max 3 sentences)
6. Single retry logic under script corruption
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.llm.output_guard import OutputGuard, GuardState, count_sentences

def run_tests():
    print("=" * 65)
    print("  PHASE 7 -- INCREMENTAL STREAMING OUTPUT GUARD VALIDATION")
    print("=" * 65)

    results = {}
    failures = []

    def record(test_id: str, passed: bool, data: dict):
        results[test_id] = {"passed": passed, **data}
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_id}: {data.get('summary', '')}")
        if not passed:
            failures.append(test_id)

    # -----------------------------------------------------------------------
    # TEST 1: Pathological Tag Splitting & 0% Leakage
    # -----------------------------------------------------------------------
    print("\n--- T1: <think> Suppression Across Pathological Chunk Splits ---")
    guard = OutputGuard()
    
    # Pathological token split sequence
    chunks = [
        "Hello! ",
        "<th", "ink", ">",
        "This is internal reasoning that must NEVER be spoken aloud. ",
        "Step 1: calculate distance. Step 2: format reply.",
        "</th", "ink>",
        "The sky is blue because of Rayleigh scattering."
    ]

    emitted = []
    for c in chunks:
        out = guard.process_chunk(c)
        if out:
            emitted.append(out)
    flush_out = guard.flush()
    if flush_out:
        emitted.append(flush_out)

    full_output = "".join(emitted).strip()
    has_leak = "<think>" in full_output.lower() or "internal reasoning" in full_output or "step 1" in full_output
    expected_output = "Hello! The sky is blue because of Rayleigh scattering."
    passed_t1 = (not has_leak) and ("Rayleigh scattering" in full_output) and ("Hello!" in full_output)

    record("T1_think_suppression", passed_t1, {
        "summary": "100% suppression of reasoning across split tokens",
        "full_output": full_output,
        "think_chars_suppressed": len(guard.think_buffer),
        "leak_detected": has_leak
    })

    # -----------------------------------------------------------------------
    # TEST 2: Processing Overhead Benchmark (< 1 ms per chunk)
    # -----------------------------------------------------------------------
    print("\n--- T2: Processing Latency Overhead Benchmark ---")
    guard.reset()
    test_stream = [
        "The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog. ",
        "Aria ", "is ", "a ", "voice ", "assistant. ", "It ", "runs ", "in ", "real-time! "
    ] * 20  # 360 chunks

    latencies_us = []
    for chunk in test_stream:
        t0 = time.perf_counter()
        _ = guard.process_chunk(chunk)
        lat = (time.perf_counter() - t0) * 1_000_000  # microseconds
        latencies_us.append(lat)

    mean_us = sum(latencies_us) / len(latencies_us)
    p95_us = sorted(latencies_us)[int(len(latencies_us) * 0.95)]
    mean_ms = mean_us / 1000.0
    passed_t2 = mean_ms < 1.0  # Must be strictly < 1 ms

    record("T2_processing_overhead", passed_t2, {
        "summary": f"Mean overhead = {mean_us:.1f} µs ({mean_ms:.4f} ms), P95 = {p95_us:.1f} µs",
        "mean_overhead_us": round(mean_us, 2),
        "mean_overhead_ms": round(mean_ms, 5),
        "p95_overhead_us": round(p95_us, 2),
        "threshold_ms": 1.0,
        "chunks_tested": len(test_stream)
    })

    # -----------------------------------------------------------------------
    # TEST 3: Markdown Stripping
    # -----------------------------------------------------------------------
    print("\n--- T3: Markdown Syntax Stripping ---")
    guard.reset()
    markdown_stream = [
        "## Important Notice\n",
        "Here is **bold** text, *italic* text, and `inline code`.\n",
        "- Bullet point 1\n",
        "- Bullet point 2\n",
        "Have a great day!"
    ]

    emitted = []
    for c in markdown_stream:
        out = guard.process_chunk(c)
        if out:
            emitted.append(out)
    emitted.append(guard.flush())
    clean_md = "".join(emitted)

    forbidden_symbols = ["##", "**", "*italic*", "`", "- Bullet"]
    has_markdown = any(sym in clean_md for sym in forbidden_symbols)
    passed_t3 = (not has_markdown) and ("Have a great day!" in clean_md)

    record("T3_markdown_stripping", passed_t3, {
        "summary": "All headers, bold, bullets, and code markers stripped",
        "raw_input_snippet": "".join(markdown_stream)[:60],
        "clean_output": clean_md,
        "has_markdown": has_markdown
    })

    # -----------------------------------------------------------------------
    # TEST 4: Arabic/Urdu Unicode Detection & Transliteration
    # -----------------------------------------------------------------------
    print("\n--- T4: Arabic/Urdu Unicode Detection & Sanitization ---")
    guard.reset()
    urdu_stream = [
        "Aap ko mera ",
        "سلام",  # Arabic script for Salam
        "! Umeed hai aap ",
        "ٹھیک",  # Arabic script for Theek
        " hon ge."
    ]

    emitted = []
    for c in urdu_stream:
        out = guard.process_chunk(c)
        if out:
            emitted.append(out)
    emitted.append(guard.flush())
    clean_urdu = "".join(emitted)

    contains_arabic = any('\u0600' <= ch <= '\u06FF' for ch in clean_urdu)
    transliterated = "Salam" in clean_urdu and "Theek" in clean_urdu
    passed_t4 = (not contains_arabic) and transliterated and guard.script_violation

    record("T4_script_sanitization", passed_t4, {
        "summary": "Arabic Unicode intercepted and safely mapped to Roman Urdu",
        "clean_output": clean_urdu,
        "contains_arabic": contains_arabic,
        "transliterated": transliterated,
        "script_violation_flag": guard.script_violation
    })

    # -----------------------------------------------------------------------
    # TEST 5: Script Corruption & Single Retry Trigger
    # -----------------------------------------------------------------------
    print("\n--- T5: Unmapped Script Corruption & Retry Signal ---")
    guard.reset()
    corrupt_stream = [
        "Pakistan ka daro sadar ",
        "حکومت پاکستان کی بنیاد",  # Unmapped complex Nastaliq phrase
        " hai."
    ]

    for c in corrupt_stream:
        guard.process_chunk(c)
    guard.flush()

    passed_t5 = guard.requires_retry and guard.script_violation
    record("T5_retry_trigger", passed_t5, {
        "summary": "Corrupt unmapped script triggers single retry requirement",
        "requires_retry": guard.requires_retry,
        "script_violation": guard.script_violation
    })

    # -----------------------------------------------------------------------
    # TEST 6: Maximum Character Clamp (<= 400 chars)
    # -----------------------------------------------------------------------
    print("\n--- T6: Response Length Clamping (Max 400 Characters) ---")
    guard.reset(max_chars=400, max_sentences=3)
    long_stream = ["This is a sentence that goes on and on with extra words. "] * 15  # ~855 chars

    emitted = []
    for c in long_stream:
        out = guard.process_chunk(c)
        if out:
            emitted.append(out)
    emitted.append(guard.flush())
    clamped_chars_text = "".join(emitted)

    passed_t6 = (len(clamped_chars_text) <= 400) and guard.is_clamped
    record("T6_char_clamping", passed_t6, {
        "summary": f"Clamped output to {len(clamped_chars_text)} chars (<= 400 chars limit)",
        "output_chars": len(clamped_chars_text),
        "is_clamped": guard.is_clamped,
        "limit": 400
    })

    # -----------------------------------------------------------------------
    # TEST 7: Maximum Sentence Clamp (<= 3 sentences)
    # -----------------------------------------------------------------------
    print("\n--- T7: Response Sentence Clamping (Max 3 Sentences) ---")
    guard.reset(max_chars=400, max_sentences=3)
    sentences_stream = [
        "First sentence is short. ",
        "Second sentence provides the core answer! ",
        "Third sentence completes the thought? ",
        "Fourth sentence should be completely ignored. ",
        "Fifth sentence should never be seen."
    ]

    emitted = []
    for c in sentences_stream:
        out = guard.process_chunk(c)
        if out:
            emitted.append(out)
    emitted.append(guard.flush())
    clamped_sents_text = "".join(emitted)

    sents = count_sentences(clamped_sents_text)
    passed_t7 = (len(sents) <= 3) and ("Fourth sentence" not in clamped_sents_text) and guard.is_clamped
    record("T7_sentence_clamping", passed_t7, {
        "summary": f"Clamped output to {len(sents)} sentences (<= 3 sentences limit)",
        "output_sentences": len(sents),
        "sentences_extracted": sents,
        "is_clamped": guard.is_clamped
    })

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    total = len(results)
    passed_count = sum(1 for r in results.values() if r["passed"])
    failed_count = total - passed_count
    print(f"  PHASE 7 SUMMARY: {passed_count}/{total} PASSED")
    print("=" * 65)

    summary = {
        "subsystem": "output_guard",
        "total_tests": total,
        "passed": passed_count,
        "failed": failed_count,
        "failures": failures,
        "all_passed": (failed_count == 0),
        "tests": results
    }

    os.makedirs("reports", exist_ok=True)
    out_json = os.path.join("reports", "phase7_output_guard_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 7 JSON report saved to: {out_json}")

    return summary


if __name__ == "__main__":
    res = run_tests()
    sys.exit(0 if res["all_passed"] else 1)
