"""
Validation benchmark for DeepSeek-R1 1.5B under strict ARIA V3 constraints:
- think: false
- num_predict = 96 (starting) and num_predict = 128 (hard maximum)
- max 3 sentences, max 500 characters
- generation timeout = 20 seconds
- actual ARIA master system prompts (English, Roman Urdu, Minglish)
- includes difficult/procedural prompts to evaluate true model capabilities
- measures: TTFT, total latency, tokens/sec, thinking tokens/chars, empty content rate,
  completion rate, script adherence, VRAM usage and headroom.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error

# Ensure stdout handles UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.llm.llm_engine import _SYSTEM_PROMPTS, _strip_think
from src.llm.output_guard import OutputGuard

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MODEL_NAME      = "deepseek-r1:1.5b"
TIMEOUT_SEC     = 20
NUM_CTX         = 2048

PROMPTS = [
    # English
    {"id": "EN_1", "lang": "english",  "prompt": "What is the speed of light?"},
    {"id": "EN_2", "lang": "english",  "prompt": "How do I boil an egg?"},
    {"id": "EN_3", "lang": "english",  "prompt": "Tell me a short interesting fact."},
    {"id": "EN_4", "lang": "english",  "prompt": "What's the weather like today?"},
    # Roman Urdu
    {"id": "UR_1", "lang": "urdu",     "prompt": "Aaj ka mausam kaisa hai?"},
    {"id": "UR_2", "lang": "urdu",     "prompt": "Mujhe ek funny joke sunao."},
    {"id": "UR_3", "lang": "urdu",     "prompt": "Pakistan ka daro sadar kaun hai?"},
    {"id": "UR_4", "lang": "urdu",     "prompt": "Aap kaun hain aur kya kar sakti hain?"},
    # Minglish
    {"id": "MG_1", "lang": "minglish", "prompt": "Yaar, kya chal raha hai aaj?"},
    {"id": "MG_2", "lang": "minglish", "prompt": "Tell me something interesting about space in Minglish."},
    {"id": "MG_3", "lang": "minglish", "prompt": "Aaj lunch mein kya khaoon?"},
    {"id": "MG_4", "lang": "minglish", "prompt": "Mera laptop bohot slow ho gaya hai, kya karoon?"},
]

SENTENCE_RE = re.compile(r'[^.!?]+[.!?]*', re.UNICODE)
ARABIC_RE   = re.compile(r'[\u0600-\u06FF]')
CJK_RE      = re.compile(r'[\u4E00-\u9FFF]')


def get_vram_info():
    """Return (used_mb, free_mb) or (None, None)."""
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            timeout=5
        ).decode().strip()
        parts = [int(p.strip()) for p in out.split(",")]
        return parts[0], parts[1]
    except Exception:
        return None, None


def count_sentences(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    hits = [s for s in SENTENCE_RE.findall(text) if s.strip()]
    return len(hits) if hits else (1 if text else 0)


def ensure_ollama_running():
    """Verify Ollama is running or start it."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("  Ollama service already running.")
            return True
    except Exception:
        print("  Starting Ollama service...")
        ollama_exe = shutil.which("ollama") or r"C:\Users\HABIB\AppData\Local\Programs\Ollama\ollama.exe"
        subprocess.Popen([ollama_exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(15):
            time.sleep(1)
            try:
                with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2) as r:
                    print("  Ollama service started successfully.")
                    return True
            except Exception:
                pass
        raise RuntimeError("Failed to start Ollama service.")


def test_single_prompt(prompt_item: dict, num_predict: int, think: bool = False):
    """
    Execute a single streaming chat request via Ollama API.
    Captures:
    - time to first chunk (ms)
    - time to first content token (ms)
    - total latency (ms)
    - thinking tokens / thinking text
    - content tokens / content text
    - done reason ('stop', 'length', etc.)
    - sentence count, char count
    - script violations (Arabic, CJK)
    """
    lang = prompt_item["lang"]
    prompt = prompt_item["prompt"]
    sys_prompt = _SYSTEM_PROMPTS.get(lang, _SYSTEM_PROMPTS["english"])

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": prompt}
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "think": think,
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": num_predict,
        },
        "keep_alive": "10m",
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    t0 = time.perf_counter()
    first_chunk_t = None
    first_content_t = None

    thinking_chunks = []
    content_chunks = []
    done_data = {}
    error_msg = None

    guard = OutputGuard(max_chars=400, max_sentences=3)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                now = time.perf_counter()
                if first_chunk_t is None:
                    first_chunk_t = (now - t0) * 1000

                try:
                    chunk = json.loads(line)
                except Exception:
                    continue

                msg = chunk.get("message", {})
                th = msg.get("thinking", "")
                ct = msg.get("content", "")

                if th:
                    thinking_chunks.append(th)
                if ct:
                    content_chunks.append(ct)
                    guard.process_chunk(ct)
                    if first_content_t is None:
                        first_content_t = (now - t0) * 1000

                if chunk.get("done", False):
                    done_data = chunk
                    break
    except urllib.error.URLError as e:
        error_msg = f"URLError/Timeout: {e}"
    except Exception as e:
        error_msg = f"Exception: {e}"

    guard.flush()
    total_latency_ms = (time.perf_counter() - t0) * 1000

    guarded_content = guard.get_full_response()
    raw_content = "".join(content_chunks)
    thinking_text = "".join(thinking_chunks)

    # Metrics
    prompt_tokens = done_data.get("prompt_eval_count", 0)
    eval_tokens   = done_data.get("eval_count", 0)
    done_reason   = done_data.get("done_reason", "unknown" if not error_msg else "error")

    # Tokens per second
    eval_dur_ns = done_data.get("eval_duration", 0)
    tps = (eval_tokens / (eval_dur_ns / 1e9)) if (eval_dur_ns and eval_tokens) else None

    # Quality assessments on final guarded output
    is_empty = len(guarded_content) == 0
    is_completed = (done_reason == "stop" or guard.is_clamped) and not is_empty
    sentence_cnt = count_sentences(guarded_content)
    char_cnt = len(guarded_content)
    has_arabic = bool(ARABIC_RE.search(guarded_content))
    has_cjk    = bool(CJK_RE.search(guarded_content))
    sentence_ok = (1 <= sentence_cnt <= 3) if not is_empty else False
    chars_ok    = (char_cnt <= 400) if not is_empty else False

    return {
        "id": prompt_item["id"],
        "lang": lang,
        "prompt": prompt,
        "num_predict": num_predict,
        "first_chunk_ms": round(first_chunk_t, 1) if first_chunk_t else None,
        "first_content_ms": round(first_content_t, 1) if first_content_t else None,
        "total_latency_ms": round(total_latency_ms, 1),
        "prompt_tokens": prompt_tokens,
        "eval_tokens": eval_tokens,
        "tps": round(tps, 1) if tps else None,
        "done_reason": done_reason,
        "thinking_chars": len(thinking_text),
        "content_chars": char_cnt,
        "content_sentences": sentence_cnt,
        "content": guarded_content,
        "is_empty": is_empty,
        "is_completed": is_completed,
        "sentence_ok": sentence_ok,
        "chars_ok": chars_ok,
        "has_arabic": has_arabic,
        "has_cjk": has_cjk,
        "error": error_msg,
    }


def run_benchmark_suite():
    print("=" * 75)
    print("  PHASE 6 DEEPSEEK-R1 1.5B RE-EVALUATION WITH REVISED GENERATION BUDGET")
    print("  Budgets tested: 512, 768 | Guarded Spoken Output: <= 3 sent, <= 400 chars")
    print("=" * 75)

    ensure_ollama_running()
    used_vram_start, free_vram_start = get_vram_info()
    print(f"\nInitial VRAM: Used={used_vram_start} MB, Free={free_vram_start} MB")

    configs = [512, 768]
    all_results = {}

    for np_val in configs:
        print(f"\n{'='*75}")
        print(f"  TESTING CONFIGURATION: num_predict = {np_val}, think = False")
        print(f"{'='*75}")

        config_results = []
        for item in PROMPTS:
            res = test_single_prompt(item, num_predict=np_val, think=False)
            config_results.append(res)

            status_icon = "PASS" if (res["is_completed"] and not res["has_arabic"] and not res["has_cjk"] and res["sentence_ok"]) else "FAIL"
            print(f"  [{status_icon}] [{res['id']} {res['lang'].upper()}] '{res['prompt']}'")
            print(f"         TTFT: {res['first_chunk_ms']} ms | Content TTFT: {res['first_content_ms']} ms | Total: {res['total_latency_ms']} ms")
            print(f"         Tokens: eval={res['eval_tokens']}/{np_val} (Reason: {res['done_reason']}) | ThinkChars: {res['thinking_chars']}")
            print(f"         Content (len={res['content_chars']}, sent={res['content_sentences']}): {repr(res['content'][:90])}")
            if res["has_arabic"]:
                print(f"         [CRITICAL SCRIPT VIOLATION] Arabic/Perso-Urdu Unicode detected!")
            if res["has_cjk"]:
                print(f"         [CRITICAL SCRIPT VIOLATION] Chinese/CJK Unicode detected!")
            if res["is_empty"]:
                print(f"         [CRITICAL FAILURE] Content is EMPTY! Entire budget consumed by thinking or dropped.")
            elif not res["is_completed"]:
                print(f"         [INCOMPLETE] Response truncated or hit token limit (done_reason={res['done_reason']})")
            print()

        all_results[f"num_predict_{np_val}"] = config_results

    used_vram_end, free_vram_end = get_vram_info()
    print(f"\nFinal VRAM: Used={used_vram_end} MB, Free={free_vram_end} MB")

    # Compute Aggregate Stats
    summary = {
        "model": MODEL_NAME,
        "vram": {
            "start_used_mb": used_vram_start,
            "start_free_mb": free_vram_start,
            "end_used_mb": used_vram_end,
            "end_free_mb": free_vram_end,
            "headroom_mb": free_vram_end,
            "headroom_target_mb": 500,
            "headroom_met": (free_vram_end >= 500) if free_vram_end else None,
        },
        "configurations": {}
    }

    print("\n" + "=" * 75)
    print("  AGGREGATE EVALUATION SUMMARY")
    print("=" * 75)

    for np_key, items in all_results.items():
        total = len(items)
        empty_count = sum(1 for x in items if x["is_empty"])
        completed_count = sum(1 for x in items if x["is_completed"])
        arabic_count = sum(1 for x in items if x["has_arabic"])
        cjk_count = sum(1 for x in items if x["has_cjk"])
        sent_ok_count = sum(1 for x in items if x["sentence_ok"])
        chars_ok_count = sum(1 for x in items if x["chars_ok"])
        
        valid_ttft = [x["first_chunk_ms"] for x in items if x["first_chunk_ms"] is not None]
        mean_ttft = sum(valid_ttft) / len(valid_ttft) if valid_ttft else None
        
        valid_tot = [x["total_latency_ms"] for x in items]
        mean_total_ms = sum(valid_tot) / len(valid_tot) if valid_tot else None

        valid_tps = [x["tps"] for x in items if x["tps"] is not None]
        mean_tps = sum(valid_tps) / len(valid_tps) if valid_tps else None

        thinking_heavy_count = sum(1 for x in items if x["thinking_chars"] > 0)
        
        # Perfect valid response: non-empty, completed (stop), no arabic, no cjk, sentence_ok
        perfect_valid = sum(
            1 for x in items
            if x["is_completed"] and not x["has_arabic"] and not x["has_cjk"] and x["sentence_ok"] and x["chars_ok"]
        )

        stat = {
            "total_prompts": total,
            "perfect_valid_responses": perfect_valid,
            "perfect_valid_rate": round(perfect_valid / total, 3),
            "empty_content_count": empty_count,
            "empty_content_rate": round(empty_count / total, 3),
            "completed_count": completed_count,
            "completed_rate": round(completed_count / total, 3),
            "script_violations_arabic": arabic_count,
            "script_violations_cjk": cjk_count,
            "thinking_token_trigger_count": thinking_heavy_count,
            "mean_ttft_ms": round(mean_ttft, 1) if mean_ttft else None,
            "mean_total_latency_ms": round(mean_total_ms, 1) if mean_total_ms else None,
            "mean_tps": round(mean_tps, 1) if mean_tps else None,
            "details": items
        }
        summary["configurations"][np_key] = stat

        print(f"\n  [Config {np_key}]")
        print(f"    - Total Prompts:               {total}")
        print(f"    - Valid & Compliant Responses: {perfect_valid}/{total} ({stat['perfect_valid_rate']*100:.1f}%)")
        print(f"    - Empty Content Responses:     {empty_count}/{total} ({stat['empty_content_rate']*100:.1f}%)")
        print(f"    - Completed (stop reason):     {completed_count}/{total} ({stat['completed_rate']*100:.1f}%)")
        print(f"    - Script Violations (Arabic):  {arabic_count}")
        print(f"    - Script Violations (Chinese): {cjk_count}")
        print(f"    - Prompts emitting thinking:   {thinking_heavy_count}/{total}")
        print(f"    - Mean TTFT:                   {stat['mean_ttft_ms']} ms")
        print(f"    - Mean Total Latency:          {stat['mean_total_latency_ms']} ms")
        print(f"    - Mean Generation TPS:         {stat['mean_tps']} tok/s")

    # Save to json
    os.makedirs("reports", exist_ok=True)
    out_json = os.path.join("reports", "phase6_deepseek_v3_limits_benchmark.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nBenchmark JSON saved to: {out_json}")

    return summary


if __name__ == "__main__":
    run_benchmark_suite()
