"""
ARIA Phase 2 — Piper English TTS Download & Setup
Downloads en_US-lessac-medium ONNX model + config.
Tries piper-tts Python package first; falls back to Piper Windows binary.
License: Piper voices are MIT licensed.
"""
import os, sys, json, hashlib, time, zipfile, shutil
from pathlib import Path
import urllib.request

PROJECT_ROOT = Path(__file__).parent.parent
PIPER_DIR    = PROJECT_ROOT / "models" / "piper"
BIN_DIR      = PIPER_DIR / "bin"
PIPER_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = "en_US-lessac-medium"
ONNX_FILE    = PIPER_DIR / f"{MODEL_NAME}.onnx"
JSON_FILE    = PIPER_DIR / f"{MODEL_NAME}.onnx.json"

# HuggingFace rhasspy/piper-voices — MIT license
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
ONNX_URL = f"{HF_BASE}/{MODEL_NAME}.onnx"
JSON_URL = f"{HF_BASE}/{MODEL_NAME}.onnx.json"

# Piper Windows binary — MIT license
PIPER_RELEASE = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
PIPER_EXE     = BIN_DIR / "piper.exe"


def download(url: str, dest: Path, label: str):
    if dest.exists():
        print(f"  Already exists: {dest.name} ({dest.stat().st_size // 1024} KB)")
        return
    print(f"  Downloading {label}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ARIA/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done  = 0
            while chunk := r.read(65536):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r    {pct}% ({done // 1024} / {total // 1024} KB)", end="", flush=True)
        print(f"\r    Done — {dest.stat().st_size // 1024} KB          ")
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed for {label}: {e}")


def try_pip_piper() -> bool:
    """Return True if piper-tts Python package is usable."""
    try:
        from piper import PiperVoice
        if not ONNX_FILE.exists():
            return False
        v = PiperVoice.load(str(ONNX_FILE), config_path=str(JSON_FILE), use_cuda=False)
        # Quick smoke test
        audio_bytes = b"".join(v.synthesize_stream_raw("test"))
        return len(audio_bytes) > 0
    except Exception as e:
        print(f"  piper-tts Python package test: FAIL ({e})")
        return False


def download_binary():
    """Download the official Piper Windows binary."""
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = PIPER_DIR / "piper_windows_amd64.zip"
    download(PIPER_RELEASE, zip_path, "Piper Windows binary")
    print("  Extracting binary...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            name = Path(member).name
            if name and not member.endswith("/"):
                target = BIN_DIR / name
                with z.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(f"  Binary extracted to: {BIN_DIR}")


def main():
    print("=" * 60)
    print("ARIA Phase 2 — Piper TTS Setup")
    print("Model: en_US-lessac-medium (MIT license)")
    print("=" * 60)

    # ── 1. Download ONNX model and config ──────────────────────
    print("\n[1] Downloading Piper ONNX model files...")
    download(ONNX_URL, ONNX_FILE, "en_US-lessac-medium.onnx (~63 MB)")
    download(JSON_URL, JSON_FILE, "en_US-lessac-medium.onnx.json")

    # ── 2. Determine synthesis backend ─────────────────────────
    print("\n[2] Checking synthesis backend...")

    backend = None
    if try_pip_piper():
        backend = "piper_python"
        print("  ✅ piper-tts Python package: AVAILABLE")
    else:
        print("  piper-tts Python package not available — using binary")
        if not PIPER_EXE.exists():
            print("\n[3] Downloading Piper Windows binary...")
            download_binary()
        if PIPER_EXE.exists():
            backend = "piper_binary"
            print(f"  ✅ Piper binary: {PIPER_EXE}")
        else:
            print("  ❌ No synthesis backend available")
            return 1

    # ── 3. License record ──────────────────────────────────────
    manifest = {
        "model"          : MODEL_NAME,
        "model_license"  : "MIT",
        "model_source"   : "rhasspy/piper-voices (HuggingFace)",
        "piper_license"  : "MIT",
        "piper_source"   : "rhasspy/piper (GitHub)",
        "backend"        : backend,
        "onnx_path"      : str(ONNX_FILE),
        "json_path"      : str(JSON_FILE),
        "binary_path"    : str(PIPER_EXE) if backend == "piper_binary" else None,
        "offline_runtime": True,
        "cuda_enabled"   : False,
        "timestamp"      : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = PIPER_DIR / "piper_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Piper setup complete. Backend: {backend}")
    print(f"   Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
