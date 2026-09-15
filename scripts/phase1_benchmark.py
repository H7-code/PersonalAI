"""
ARIA Phase 1 — Urdu/Minglish TTS Quality Gate Benchmark
Model  : facebook/mms-tts-urd-script_latin
License: CC BY-NC 4.0 (Personal/Non-Commercial use confirmed)
CPU-only: CUDA is explicitly disabled throughout.

Metrics measured:
  - Synthesis latency (ms) per sentence
  - Real-Time Factor (RTF = synth_ms / audio_duration_ms)
  - Peak RAM delta (MB) during synthesis
  - VRAM usage (must remain 0 MB delta — CPU only)
  - Audio output saved to WAV for human intelligibility review
  - Basic audio health check (RMS energy, clipping detection)

V3 benchmark sentences (20 total):
  - Roman Urdu (6)
  - Minglish (8)
  - English technical terms (3)
  - Naturalness/prosody tests (3)
"""

import os
import sys
import time
import json
import wave
import struct
import traceback
import subprocess
from pathlib import Path

import psutil

# ── Enforce CPU-only BEFORE any torch/transformers import ──────
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR    = PROJECT_ROOT / "models" / "mms"
OUTPUT_DIR   = PROJECT_ROOT / "phase1_outputs"
WAV_DIR      = OUTPUT_DIR / "wav"
REPORT_DIR   = OUTPUT_DIR / "reports"

WAV_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Phase 1 Acceptance Thresholds ─────────────────────────────
LATENCY_P95_TARGET_MS   = 5000   # ≤ 5 s latency for 95th percentile
RTF_TARGET              = 1.0    # RTF must be < 1.0 (real-time capable)
VRAM_DELTA_TARGET_MB    = 50     # VRAM delta must stay < 50 MB (CPU-only)
RAM_DELTA_LIMIT_MB      = 1500   # RAM growth during synthesis ≤ 1500 MB
AUDIO_ENERGY_FLOOR      = 0.001  # Minimum RMS (silence rejection)
CLIPPING_THRESHOLD      = 0.99   # Amplitude > 99% of max = clipping

# ── V3 Benchmark Sentences ─────────────────────────────────────
# 20 sentences covering Roman Urdu, Minglish, English technical terms, naturalness
BENCHMARK_SENTENCES = [
    # ── Roman Urdu (6) ────────────────────────────────────────
    {
        "id": "RU-01",
        "category": "roman_urdu",
        "text": "Aap ka din kaisa raha?",
        "expected_check": "greeting naturalness",
    },
    {
        "id": "RU-02",
        "category": "roman_urdu",
        "text": "Main aap ki madad karna chahta hoon.",
        "expected_check": "complete sentence fluency",
    },
    {
        "id": "RU-03",
        "category": "roman_urdu",
        "text": "Yeh bahut acha sawaal hai, mujhe socha dena hoga.",
        "expected_check": "thinking/reflection phrasing",
    },
    {
        "id": "RU-04",
        "category": "roman_urdu",
        "text": "Theek hai, ab baat karte hain.",
        "expected_check": "short affirmation",
    },
    {
        "id": "RU-05",
        "category": "roman_urdu",
        "text": "Mera naam ARIA hai, aur main aap ka assistant hoon.",
        "expected_check": "self-identification",
    },
    {
        "id": "RU-06",
        "category": "roman_urdu",
        "text": "Shukriya, aap ka yeh sawal bahut important hai.",
        "expected_check": "gratitude + emphasis",
    },

    # ── Minglish / Code-Switched (8) ──────────────────────────
    {
        "id": "MG-01",
        "category": "minglish",
        "text": "Aap ka system update ho gaya hai.",
        "expected_check": "tech action in Urdu syntax",
    },
    {
        "id": "MG-02",
        "category": "minglish",
        "text": "CPU usage thori zyada hai, check karo.",
        "expected_check": "technical metric mention",
    },
    {
        "id": "MG-03",
        "category": "minglish",
        "text": "File download complete ho gayi, ab open kar sakte ho.",
        "expected_check": "file operation code-switch",
    },
    {
        "id": "MG-04",
        "category": "minglish",
        "text": "Network connection stable hai, speed aachi hai.",
        "expected_check": "network status mixed",
    },
    {
        "id": "MG-05",
        "category": "minglish",
        "text": "Model load ho raha hai, thora wait karo.",
        "expected_check": "AI model reference",
    },
    {
        "id": "MG-06",
        "category": "minglish",
        "text": "Pehle backup lo, phir delete karo.",
        "expected_check": "imperative code-switch",
    },
    {
        "id": "MG-07",
        "category": "minglish",
        "text": "Yeh error message kuch theek nahi lag raha.",
        "expected_check": "error context",
    },
    {
        "id": "MG-08",
        "category": "minglish",
        "text": "App crash ho gayi thi, ab restart kar diya.",
        "expected_check": "past tense tech event",
    },

    # ── English Technical Terms (3) ───────────────────────────
    {
        "id": "EN-01",
        "category": "english_technical",
        "text": "CUDA, GPU, RAM, CPU, API, JSON, HTTP.",
        "expected_check": "acronym pronunciation",
    },
    {
        "id": "EN-02",
        "category": "english_technical",
        "text": "PyTorch, TensorFlow, transformer, neural network.",
        "expected_check": "ML library terms",
    },
    {
        "id": "EN-03",
        "category": "english_technical",
        "text": "Install the dependency and restart the server.",
        "expected_check": "plain English technical instruction",
    },

    # ── Naturalness / Prosody (3) ──────────────────────────────
    {
        "id": "NAT-01",
        "category": "naturalness",
        "text": "Okay! Bilkul theek hai.",
        "expected_check": "short energetic affirmation",
    },
    {
        "id": "NAT-02",
        "category": "naturalness",
        "text": "Mujhe yaqeen hai ke aap yeh kar sakte hain.",
        "expected_check": "encouragement prosody",
    },
    {
        "id": "NAT-03",
        "category": "naturalness",
        "text": "Ek second, main soch raha hoon.",
        "expected_check": "pause/thinking naturalness",
    },
]

# ── VRAM Query ─────────────────────────────────────────────────

def get_vram_used_mb() -> float:
    """Query VRAM via nvidia-smi. Returns -1 if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5
        ).decode().strip()
        return float(out.split("\n")[0].strip())
    except Exception:
        return -1.0


def get_ram_used_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1_048_576


# ── WAV Writer ─────────────────────────────────────────────────

def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Save float32 or int16 numpy array as 16-bit WAV."""
    if audio.dtype != np.int16:
        # Clip and convert float to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


# ── Audio Health Check ─────────────────────────────────────────

def check_audio_health(audio: np.ndarray) -> dict:
    """Check RMS energy, clipping, and silence."""
    if audio.dtype == np.int16:
        f = audio.astype(np.float32) / 32768.0
    else:
        f = audio.astype(np.float32)

    rms      = float(np.sqrt(np.mean(f ** 2)))
    peak     = float(np.max(np.abs(f)))
    clipping = bool(peak > CLIPPING_THRESHOLD)
    silence  = bool(rms < AUDIO_ENERGY_FLOOR)
    duration_s = len(f) / 16000  # MMS default sample rate

    return {
        "rms"         : round(rms, 5),
        "peak"        : round(peak, 5),
        "clipping"    : clipping,
        "silence"     : silence,
        "duration_s"  : round(duration_s, 3),
        "health_pass" : (not clipping and not silence),
    }


# ── Model Loader ───────────────────────────────────────────────

def load_mms_model(model_dir: Path):
    """Load VitsModel + VitsTokenizer from local directory (CPU only)."""
    import torch
    from transformers import VitsModel, AutoTokenizer

    print(f"  Loading tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    print(f"  Loading VitsModel from {model_dir} ...")
    model = VitsModel.from_pretrained(str(model_dir))
    model.eval()

    # Enforce CPU
    model = model.to("cpu")

    # Try INT8 quantization for speed (optional, graceful fallback)
    try:
        import torch.quantization
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("  INT8 dynamic quantization applied (linear layers).")
    except Exception as e:
        print(f"  INT8 quantization skipped: {e}")

    return model, tokenizer


# ── Single Sentence Synthesis ──────────────────────────────────

def synthesize(model, tokenizer, text: str) -> tuple[np.ndarray, int]:
    """
    Synthesize text to audio. Returns (audio_array_float32, sample_rate).
    Enforces no CUDA usage.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt")
    # Move to CPU explicitly
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    waveform = output.waveform.squeeze().cpu().numpy()
    sample_rate = model.config.sampling_rate
    return waveform, sample_rate


# ── Main Benchmark ─────────────────────────────────────────────

def run_benchmark():
    print("=" * 65)
    print("ARIA PHASE 1 — MMS-TTS QUALITY GATE BENCHMARK")
    print(f"Model : facebook/mms-tts-urd-script_latin")
    print(f"Device: CPU-only (CUDA_VISIBLE_DEVICES='')")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 65)

    # ── Pre-flight checks ──────────────────────────────────────
    if not MODEL_DIR.exists():
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        print("       Run scripts/download_mms_tts.py first.")
        return 1

    required = ["config.json", "tokenizer_config.json", "vocab.json"]
    weights  = [MODEL_DIR / "model.safetensors", MODEL_DIR / "pytorch_model.bin"]
    missing  = [f for f in required if not (MODEL_DIR / f).exists()]
    has_weights = any(w.exists() for w in weights)

    if missing:
        print(f"ERROR: Missing model files: {missing}")
        return 1
    if not has_weights:
        print("ERROR: No weight file found (model.safetensors or pytorch_model.bin)")
        return 1

    print(f"\nModel directory OK: {MODEL_DIR}")
    for f in MODEL_DIR.iterdir():
        print(f"  {f.name}  ({f.stat().st_size / 1_048_576:.2f} MB)")

    # ── Baseline measurements ──────────────────────────────────
    vram_baseline = get_vram_used_mb()
    ram_baseline  = get_ram_used_mb()
    print(f"\nBaseline VRAM: {vram_baseline:.1f} MB")
    print(f"Baseline RAM : {ram_baseline:.1f} MB")

    # ── Load model ─────────────────────────────────────────────
    print("\nLoading model...")
    t_load_start = time.perf_counter()
    try:
        model, tokenizer = load_mms_model(MODEL_DIR)
    except Exception as e:
        print(f"\nFATAL: Could not load MMS-TTS model:\n{traceback.format_exc()}")
        return 1
    t_load_end = time.perf_counter()
    load_time_s = t_load_end - t_load_start

    vram_after_load = get_vram_used_mb()
    ram_after_load  = get_ram_used_mb()
    vram_load_delta = (vram_after_load - vram_baseline) if vram_baseline >= 0 else -1
    ram_load_delta  = ram_after_load - ram_baseline

    print(f"Model loaded in {load_time_s:.2f}s")
    print(f"RAM  delta after load: +{ram_load_delta:.1f} MB (total: {ram_after_load:.1f} MB)")
    print(f"VRAM delta after load: {vram_load_delta:+.1f} MB  (MUST be ~0)")

    if vram_load_delta > VRAM_DELTA_TARGET_MB:
        print(f"WARNING: VRAM delta {vram_load_delta:.1f} MB exceeds {VRAM_DELTA_TARGET_MB} MB threshold — GPU may be in use!")

    # ── Benchmark loop ─────────────────────────────────────────
    results  = []
    failures = []

    print(f"\nRunning {len(BENCHMARK_SENTENCES)} benchmark sentences...\n")

    for i, sent in enumerate(BENCHMARK_SENTENCES):
        sid      = sent["id"]
        category = sent["category"]
        text     = sent["text"]
        print(f"  [{i+1:02d}/{len(BENCHMARK_SENTENCES)}] {sid} ({category})")
        print(f"         Text: {text[:70]}")

        vram_before = get_vram_used_mb()
        ram_before  = get_ram_used_mb()
        t0 = time.perf_counter()

        try:
            audio, sr = synthesize(model, tokenizer, text)
            t1 = time.perf_counter()

            latency_ms    = (t1 - t0) * 1000
            audio_dur_ms  = (len(audio) / sr) * 1000
            rtf           = latency_ms / max(audio_dur_ms, 1)

            vram_after  = get_vram_used_mb()
            ram_after   = get_ram_used_mb()
            vram_delta  = (vram_after - vram_before) if vram_before >= 0 else -1
            ram_delta   = ram_after - ram_before

            health = check_audio_health(audio)

            # Save WAV
            wav_path = WAV_DIR / f"{sid}.wav"
            save_wav(wav_path, audio, sr)

            # Gate checks
            latency_ok = latency_ms < LATENCY_P95_TARGET_MS
            rtf_ok     = rtf < RTF_TARGET
            vram_ok    = vram_delta < VRAM_DELTA_TARGET_MB if vram_delta >= 0 else True
            audio_ok   = health["health_pass"]
            gate_pass  = latency_ok and rtf_ok and vram_ok and audio_ok

            result = {
                "id"                : sid,
                "category"          : category,
                "text"              : text,
                "expected_check"    : sent["expected_check"],
                "latency_ms"        : round(latency_ms, 1),
                "audio_duration_ms" : round(audio_dur_ms, 1),
                "rtf"               : round(rtf, 4),
                "sample_rate"       : sr,
                "ram_delta_mb"      : round(ram_delta, 1),
                "vram_delta_mb"     : round(vram_delta, 1) if vram_delta >= 0 else "N/A",
                "audio_health"      : health,
                "wav_file"          : str(wav_path),
                "gate_checks"       : {
                    "latency_ok"    : latency_ok,
                    "rtf_ok"        : rtf_ok,
                    "vram_ok"       : vram_ok,
                    "audio_ok"      : audio_ok,
                },
                "gate_pass"         : gate_pass,
                "error"             : None,
            }

            tag = "PASS" if gate_pass else "FAIL"
            print(f"         -> {tag}  latency={latency_ms:.0f}ms  RTF={rtf:.3f}  "
                  f"dur={audio_dur_ms:.0f}ms  VRAM={vram_delta:+.1f}MB  "
                  f"RMS={health['rms']:.4f}  WAV saved")

        except Exception as e:
            t1 = time.perf_counter()
            error_str = traceback.format_exc()
            result = {
                "id"        : sid,
                "category"  : category,
                "text"      : text,
                "gate_pass" : False,
                "error"     : error_str,
                "latency_ms": round((t1 - t0) * 1000, 1),
            }
            failures.append(sid)
            print(f"         -> FAIL (exception): {e}")

        results.append(result)

    # ── Aggregate statistics ───────────────────────────────────
    passed    = [r for r in results if r.get("gate_pass")]
    failed    = [r for r in results if not r.get("gate_pass")]
    errors    = [r for r in results if r.get("error")]

    latencies = [r["latency_ms"] for r in results if "latency_ms" in r and not r.get("error")]
    rtfs      = [r["rtf"] for r in results if "rtf" in r and not r.get("error")]

    stats = {}
    if latencies:
        stats["latency_ms"] = {
            "min"    : round(min(latencies), 1),
            "max"    : round(max(latencies), 1),
            "mean"   : round(sum(latencies) / len(latencies), 1),
            "p95"    : round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
        }
    if rtfs:
        stats["rtf"] = {
            "min"    : round(min(rtfs), 4),
            "max"    : round(max(rtfs), 4),
            "mean"   : round(sum(rtfs) / len(rtfs), 4),
        }

    # Category breakdown
    cats = {}
    for r in results:
        cat = r.get("category", "unknown")
        cats.setdefault(cat, {"pass": 0, "fail": 0})
        if r.get("gate_pass"):
            cats[cat]["pass"] += 1
        else:
            cats[cat]["fail"] += 1

    # Phase 1 overall gate
    p95_latency    = stats.get("latency_ms", {}).get("p95", 9999)
    mean_rtf       = stats.get("rtf", {}).get("mean", 9.9)
    phase1_pass    = (
        len(failed) == 0
        and p95_latency <= LATENCY_P95_TARGET_MS
        and mean_rtf < RTF_TARGET
    )

    # ── Compile report ─────────────────────────────────────────
    report = {
        "phase"                    : 1,
        "title"                    : "MMS-TTS Urdu/Minglish Quality Gate",
        "model"                    : "facebook/mms-tts-urd-script_latin",
        "license"                  : "CC BY-NC 4.0",
        "deployment_mode"          : "PERSONAL_NON_COMMERCIAL",
        "device"                   : "CPU-only",
        "timestamp"                : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_load_time_s"        : round(load_time_s, 2),
        "baseline_vram_mb"         : vram_baseline,
        "vram_after_model_load_mb" : vram_after_load,
        "vram_load_delta_mb"       : round(vram_load_delta, 1) if isinstance(vram_load_delta, float) else -1,
        "baseline_ram_mb"          : round(ram_baseline, 1),
        "ram_after_load_mb"        : round(ram_after_load, 1),
        "acceptance_thresholds"    : {
            "latency_p95_ms"       : LATENCY_P95_TARGET_MS,
            "rtf_max"              : RTF_TARGET,
            "vram_delta_max_mb"    : VRAM_DELTA_TARGET_MB,
        },
        "aggregate_stats"          : stats,
        "category_breakdown"       : cats,
        "total_sentences"          : len(BENCHMARK_SENTENCES),
        "passed"                   : len(passed),
        "failed"                   : len(failed),
        "errors"                   : len(errors),
        "phase1_gate_pass"         : phase1_pass,
        "sentences"                : results,
    }

    report_path = REPORT_DIR / "phase1_mms_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print summary ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 1 BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  Total sentences    : {len(BENCHMARK_SENTENCES)}")
    print(f"  Passed             : {len(passed)}")
    print(f"  Failed             : {len(failed)}")
    print(f"  Errors             : {len(errors)}")
    if latencies:
        print(f"  Latency  mean/p95  : {stats['latency_ms']['mean']:.0f} ms / {stats['latency_ms']['p95']:.0f} ms")
        print(f"  RTF mean           : {stats['rtf']['mean']:.4f}  (target < {RTF_TARGET})")
    print(f"  VRAM delta (load)  : {vram_load_delta:+.1f} MB  (must be ~0)")
    print(f"\n  Category breakdown:")
    for cat, counts in cats.items():
        total_cat = counts["pass"] + counts["fail"]
        print(f"    {cat:<25} {counts['pass']}/{total_cat} passed")
    print()

    gate_str = "PASS" if phase1_pass else "FAIL"
    print(f"  PHASE 1 GATE: {gate_str}")
    if not phase1_pass:
        print("\n  Failed sentences:")
        for r in failed:
            chk = r.get("gate_checks", {})
            print(f"    {r['id']}: latency={r.get('latency_ms','?')}ms  "
                  f"checks={chk}  error={r.get('error','none')[:80] if r.get('error') else 'none'}")

    print(f"\n  Report: {report_path}")
    print(f"  WAVs  : {WAV_DIR}")
    print("=" * 65)

    return 0 if phase1_pass else 1


if __name__ == "__main__":
    sys.exit(run_benchmark())
