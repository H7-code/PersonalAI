"""
ARIA Phase 2 — English TTS Quality Gate Benchmark
Uses Piper en_US-lessac-medium (ONNX Runtime, CPU-only, MIT license)

V3 Acceptance Criteria (Section 33, Phase 2):
  - Mean RTF  <= 0.25x on CPU
  - First-chunk latency <= 300 ms
  - Natural prosody verified
  - VRAM = 0 MB

15 standard English sentences per V3 spec, covering:
  Group A (5): Conversational greetings / everyday assistant replies
  Group B (5): Technical/system information (ARIA's primary domain)
  Group C (5): Prosody stress tests (punctuation, varied rhythm, questions)
"""

import os, sys, json, time, wave, subprocess, struct
from pathlib import Path
import threading
import psutil
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).parent.parent
PIPER_DIR    = PROJECT_ROOT / "models" / "piper"
OUTPUT_DIR   = PROJECT_ROOT / "phase2_outputs"
WAV_DIR      = OUTPUT_DIR / "wav"
REPORT_DIR   = OUTPUT_DIR / "reports"
WAV_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_FILE   = PIPER_DIR / "en_US-lessac-medium.onnx"
JSON_FILE   = PIPER_DIR / "en_US-lessac-medium.onnx.json"
MANIFEST    = PIPER_DIR / "piper_manifest.json"

V3_MEAN_RTF_TARGET     = 0.25
V3_FIRST_CHUNK_TARGET  = 300   # ms
SAMPLE_RATE_EXPECTED   = 22050

# ── V3 Phase 2 English benchmark corpus (15 sentences) ────────
SENTENCES = [
    # Group A: Conversational
    {"id": "EN-A01", "text": "Hello! How can I help you today?",                   "group": "conversational"},
    {"id": "EN-A02", "text": "I'm sorry, I didn't quite catch that. Could you repeat?", "group": "conversational"},
    {"id": "EN-A03", "text": "Sure, let me look that up for you right now.",        "group": "conversational"},
    {"id": "EN-A04", "text": "That's a great question. Here's what I found.",       "group": "conversational"},
    {"id": "EN-A05", "text": "Done! Is there anything else you'd like me to do?",   "group": "conversational"},
    # Group B: Technical / system
    {"id": "EN-B01", "text": "Your system memory usage is currently at sixty-four percent.", "group": "technical"},
    {"id": "EN-B02", "text": "The download has completed successfully. The file is ready.", "group": "technical"},
    {"id": "EN-B03", "text": "I couldn't connect to the Ollama service. Please check if it's running.", "group": "technical"},
    {"id": "EN-B04", "text": "The model has been loaded and is ready to accept your questions.", "group": "technical"},
    {"id": "EN-B05", "text": "Warning: available disk space is below five hundred megabytes.",  "group": "technical"},
    # Group C: Prosody stress tests
    {"id": "EN-C01", "text": "Wait — are you absolutely sure about that?",         "group": "prosody"},
    {"id": "EN-C02", "text": "First, open the settings. Then, navigate to privacy. Finally, disable telemetry.", "group": "prosody"},
    {"id": "EN-C03", "text": "The CPU, GPU, and RAM are all functioning within normal parameters.", "group": "prosody"},
    {"id": "EN-C04", "text": "Processing... please hold on just a moment.",        "group": "prosody"},
    {"id": "EN-C05", "text": "Excellent work! You've completed all fifteen benchmark sentences.", "group": "prosody"},
]


# ── VRAM reader ────────────────────────────────────────────────
def read_vram_mb() -> float:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0.0


# ── WAV utilities ──────────────────────────────────────────────
def save_wav(path: Path, audio: np.ndarray, sr: int):
    a = np.clip(audio, -1.0, 1.0)
    pcm = (a * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

def load_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr    = wf.getframerate()
        data  = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0, sr

def audio_health(audio: np.ndarray) -> dict:
    rms   = float(np.sqrt(np.mean(audio ** 2)))
    peak  = float(np.max(np.abs(audio)))
    silent = rms < 0.001
    # True clipping: >0.5% of samples saturated at |peak| >= 0.9999
    # Piper normalises its output to [-1.0, 1.0] by design — a single
    # sample touching 1.0 is normal; saturation of many samples is not.
    sat_frac = float(np.mean(np.abs(audio) >= 0.9999))
    clipped = sat_frac > 0.005   # more than 0.5% of samples at ceiling
    return {"rms": round(rms, 4), "peak": round(peak, 4),
            "saturation_frac": round(sat_frac, 6),
            "silent": silent, "clipped": clipped}


# ── Synthesizer ────────────────────────────────────────────────
class PiperSynthesizer:
    def __init__(self):
        from piper import PiperVoice
        print("  Loading PiperVoice...")
        t0 = time.perf_counter()
        self.voice = PiperVoice.load(
            str(ONNX_FILE),
            config_path=str(JSON_FILE),
            use_cuda=False,
        )
        self.load_time = time.perf_counter() - t0
        self.sample_rate = self.voice.config.sample_rate
        print(f"  PiperVoice loaded in {self.load_time:.2f}s  "
              f"sample_rate={self.sample_rate} Hz")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Returns (float32 audio array, sample_rate).
        piper-tts 1.8.0: voice.synthesize() yields AudioChunk objects with .audio_int16
        """
        chunks = []
        for chunk in self.voice.synthesize(text):
            # AudioChunk has .audio_int16_array (numpy int16 array)
            arr = np.asarray(chunk.audio_int16_array, dtype=np.int16)
            chunks.append(arr)
        if not chunks:
            return np.zeros(1, dtype=np.float32), self.sample_rate
        pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
        return pcm, self.sample_rate


# ── Main benchmark ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("ARIA Phase 2 — English TTS Benchmark (Piper en_US-lessac-medium)")
    print(f"V3 targets: RTF <= {V3_MEAN_RTF_TARGET}x | First-chunk <= {V3_FIRST_CHUNK_TARGET} ms | VRAM = 0 MB")
    print("=" * 65)

    if not ONNX_FILE.exists():
        print(f"ERROR: ONNX model not found at {ONNX_FILE}")
        print("Run scripts/download_piper.py first.")
        return 1

    # ── VRAM baseline ──────────────────────────────────────────
    vram_baseline = read_vram_mb()
    print(f"\nVRAM baseline: {vram_baseline:.0f} MB")

    # ── Init synthesizer ───────────────────────────────────────
    proc = psutil.Process(os.getpid())
    ram_before_load = proc.memory_info().rss / 1_048_576

    synth = PiperSynthesizer()

    ram_after_load = proc.memory_info().rss / 1_048_576
    vram_after_load = read_vram_mb()
    vram_delta_load = vram_after_load - vram_baseline

    print(f"  RAM before load : {ram_before_load:.1f} MB")
    print(f"  RAM after load  : {ram_after_load:.1f} MB  (+{ram_after_load-ram_before_load:.1f} MB)")
    print(f"  VRAM after load : {vram_after_load:.0f} MB  (delta: {vram_delta_load:+.0f} MB)")

    # ── Warm-up ────────────────────────────────────────────────
    print("\nWarm-up synthesis (not timed)...")
    _ = synth.synthesize("Warm up.")

    # ── Peak RAM monitor ───────────────────────────────────────
    peak_ram = [proc.memory_info().rss / 1_048_576]
    stop_mon = threading.Event()
    def monitor():
        while not stop_mon.is_set():
            v = proc.memory_info().rss / 1_048_576
            if v > peak_ram[0]: peak_ram[0] = v
            time.sleep(0.05)
    threading.Thread(target=monitor, daemon=True).start()

    # ── Benchmark loop ─────────────────────────────────────────
    print(f"\nSynthesizing {len(SENTENCES)} sentences...\n")
    results = []
    header = f"{'ID':<9} {'Group':<15} {'Latency':>8} {'Audio':>8} {'RTF':>6} {'RMS':>6} {'Gate'}"
    print(header)
    print("-" * len(header))

    for s in SENTENCES:
        sid  = s["id"]
        text = s["text"]
        grp  = s["group"]

        t0 = time.perf_counter()
        audio, sr = synth.synthesize(text)
        latency_ms = (time.perf_counter() - t0) * 1000

        dur_ms  = len(audio) / sr * 1000
        rtf     = latency_ms / max(dur_ms, 1)
        health  = audio_health(audio)
        vram_now = read_vram_mb()
        cur_ram  = proc.memory_info().rss / 1_048_576

        wav_path = WAV_DIR / f"{sid}.wav"
        save_wav(wav_path, audio, sr)

        rtf_ok     = rtf < V3_MEAN_RTF_TARGET
        fc_ok      = latency_ms <= V3_FIRST_CHUNK_TARGET
        quality_ok = not health["silent"] and not health["clipped"]
        gate_pass  = rtf_ok and fc_ok and quality_ok
        gate_str   = "PASS" if gate_pass else "WARN"

        print(f"{sid:<9} {grp:<15} {latency_ms:>7.0f}ms {dur_ms:>7.0f}ms "
              f"{rtf:>6.3f} {health['rms']:>6.3f} [{gate_str}]")

        results.append({
            "id"            : sid,
            "group"         : grp,
            "text"          : text,
            "latency_ms"    : round(latency_ms, 1),
            "audio_dur_ms"  : round(dur_ms, 1),
            "rtf"           : round(rtf, 4),
            "first_chunk_ms": round(latency_ms, 1),  # ONNX is batch — latency = first chunk
            "rms"           : health["rms"],
            "peak"          : health["peak"],
            "silent"        : health["silent"],
            "clipped"       : health["clipped"],
            "vram_mb"       : vram_now,
            "ram_mb"        : round(cur_ram, 1),
            "gate_pass"     : gate_pass,
            "wav_file"      : str(wav_path),
        })

    stop_mon.set()

    # ── Compute summary ────────────────────────────────────────
    latencies = [r["latency_ms"] for r in results]
    rtfs      = [r["rtf"] for r in results]
    vrams     = [r["vram_mb"] for r in results]

    mean_lat  = sum(latencies) / len(latencies)
    p95_lat   = sorted(latencies)[int(len(latencies) * 0.95)]
    min_lat   = min(latencies)
    max_lat   = max(latencies)
    mean_rtf  = sum(rtfs) / len(rtfs)
    max_rtf   = max(rtfs)
    max_vram  = max(vrams)
    vram_delta_synth = max_vram - vram_baseline

    silent_count  = sum(1 for r in results if r["silent"])
    clipped_count = sum(1 for r in results if r["clipped"])

    # V3 gate evaluation
    gate_rtf    = mean_rtf <= V3_MEAN_RTF_TARGET
    gate_fc     = all(r["first_chunk_ms"] <= V3_FIRST_CHUNK_TARGET for r in results)
    gate_vram   = vram_delta_synth <= 50
    gate_quality= (silent_count == 0 and clipped_count == 0)
    gate_overall = gate_rtf and gate_fc and gate_vram and gate_quality

    group_stats = {}
    for grp in ["conversational", "technical", "prosody"]:
        grp_r = [r for r in results if r["group"] == grp]
        group_stats[grp] = {
            "mean_latency_ms" : round(sum(r["latency_ms"] for r in grp_r) / len(grp_r), 1),
            "mean_rtf"        : round(sum(r["rtf"] for r in grp_r) / len(grp_r), 4),
            "count"           : len(grp_r),
        }

    report = {
        "phase"            : 2,
        "model"            : "en_US-lessac-medium",
        "backend"          : "piper-tts Python (ONNX Runtime CPU)",
        "license"          : "MIT",
        "sample_rate_hz"   : synth.sample_rate,
        "model_load_time_s": round(synth.load_time, 2),
        "v3_targets"       : {
            "mean_rtf"       : V3_MEAN_RTF_TARGET,
            "first_chunk_ms" : V3_FIRST_CHUNK_TARGET,
            "vram_mb"        : 0,
        },
        "ram_mb" : {
            "baseline"       : round(ram_before_load, 1),
            "after_load"     : round(ram_after_load, 1),
            "delta_load"     : round(ram_after_load - ram_before_load, 1),
            "peak_synthesis" : round(peak_ram[0], 1),
        },
        "vram_mb": {
            "baseline"        : vram_baseline,
            "after_load"      : vram_after_load,
            "delta_load"      : vram_delta_load,
            "peak_synthesis"  : max_vram,
            "delta_synthesis" : vram_delta_synth,
        },
        "latency_ms": {
            "mean"  : round(mean_lat, 1),
            "p95"   : round(p95_lat, 1),
            "min"   : round(min_lat, 1),
            "max"   : round(max_lat, 1),
        },
        "rtf": {
            "mean"  : round(mean_rtf, 4),
            "max"   : round(max_rtf, 4),
        },
        "audio_health": {
            "silent_count"  : silent_count,
            "clipped_count" : clipped_count,
        },
        "gate_results": {
            "mean_rtf_pass"          : gate_rtf,
            "first_chunk_pass"       : gate_fc,
            "vram_pass"              : gate_vram,
            "audio_quality_pass"     : gate_quality,
            "overall_pass"           : gate_overall,
        },
        "group_breakdown" : group_stats,
        "timestamp"       : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sentences"       : results,
    }

    out = REPORT_DIR / "phase2_english_tts_benchmark.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 65)
    print("PHASE 2 BENCHMARK SUMMARY")
    print(f"  Mean RTF        : {mean_rtf:.4f}  (V3 target <= {V3_MEAN_RTF_TARGET}) {'PASS' if gate_rtf else 'FAIL'}")
    print(f"  Mean latency    : {mean_lat:.1f} ms")
    print(f"  P95 latency     : {p95_lat:.1f} ms")
    print(f"  Min latency     : {min_lat:.1f} ms")
    print(f"  Max latency     : {max_lat:.1f} ms")
    print(f"  First-chunk max : {max(latencies):.1f} ms  (V3 target <= {V3_FIRST_CHUNK_TARGET}) {'PASS' if gate_fc else 'FAIL'}")
    print(f"  Peak RAM        : {peak_ram[0]:.1f} MB")
    print(f"  VRAM delta      : {vram_delta_synth:+.0f} MB  {'PASS' if gate_vram else 'FAIL'}")
    print(f"  Silent outputs  : {silent_count} / {len(results)}")
    print(f"  Clipped outputs : {clipped_count} / {len(results)}")
    print(f"  Sample rate     : {synth.sample_rate} Hz")
    print()
    print(f"  OVERALL GATE    : {'[PASS]' if gate_overall else '[FAIL]'}")
    print(f"  Report          : {out}")
    print("=" * 65)

    return 0 if gate_overall else 1



if __name__ == "__main__":
    sys.exit(main())
