"""
ARIA Phase 1 — MMS-TTS Model Downloader
Model: facebook/mms-tts-urd-script_latin
License: CC BY-NC 4.0 (confirmed: Personal/Non-Commercial deployment)
Target dir: models/mms/
"""

import os
import sys
import time
import json
import urllib.request
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────
REPO_ID    = "facebook/mms-tts-urd-script_latin"
MODEL_DIR  = Path(__file__).parent.parent / "models" / "mms"

HF_FILES = [
    "config.json",
    "tokenizer_config.json",
    "vocab.json",
    "model.safetensors",
    "special_tokens_map.json",
]

HF_BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

# ── Helpers ────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "aria-phase1-downloader/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / 1_048_576
                        total_mb = total / 1_048_576
                        print(f"\r  {desc}: {mb:.1f} MB / {total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
            print()
        tmp.rename(dest)
        return True
    except Exception as e:
        print(f"\n  ERROR downloading {url}: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main():
    print("=" * 60)
    print("ARIA Phase 1 — MMS-TTS Model Download")
    print(f"Repo  : {REPO_ID}")
    print(f"Target: {MODEL_DIR}")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    all_ok  = True

    for fname in HF_FILES:
        dest = MODEL_DIR / fname
        if file_ok(dest):
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  [SKIP] {fname} already exists ({size_mb:.1f} MB)")
            results[fname] = "cached"
            continue

        url = f"{HF_BASE_URL}/{fname}"
        print(f"  [DL]   {fname}")
        ok = download_file(url, dest, fname)
        if ok:
            size_mb = dest.stat().st_size / 1_048_576
            print(f"         -> {size_mb:.1f} MB  OK")
            results[fname] = f"downloaded ({size_mb:.1f} MB)"
        else:
            if fname == "model.safetensors":
                fallback = "pytorch_model.bin"
                fallback_url = f"{HF_BASE_URL}/{fallback}"
                fallback_dest = MODEL_DIR / fallback
                print(f"  [FALLBACK] safetensors failed, trying {fallback}")
                ok2 = download_file(fallback_url, fallback_dest, fallback)
                if ok2:
                    size_mb = fallback_dest.stat().st_size / 1_048_576
                    print(f"         -> {size_mb:.1f} MB  OK")
                    results[fname] = f"downloaded as pytorch_model.bin ({size_mb:.1f} MB)"
                else:
                    results[fname] = "FAILED"
                    all_ok = False
            else:
                results[fname] = "FAILED"
                all_ok = False

    manifest_path = MODEL_DIR / "download_manifest.json"
    manifest = {
        "repo_id": REPO_ID,
        "license": "CC BY-NC 4.0",
        "deployment_mode": "PERSONAL_NON_COMMERCIAL",
        "download_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": results,
        "all_files_ok": all_ok,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 60)
    print("Download Summary:")
    for k, v in results.items():
        status = "OK" if "FAIL" not in v else "FAIL"
        print(f"  [{status}] {k}: {v}")
    print(f"\nAll files OK: {all_ok}")
    print(f"Manifest saved: {manifest_path}")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
