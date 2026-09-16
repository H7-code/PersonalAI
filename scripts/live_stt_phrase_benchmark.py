"""Capture four real microphone phrases and compare cached Faster-Whisper models."""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.capture import AudioCaptureManager
from src.audio.vad import SileroVAD
from src.stt.whisper_engine import WhisperEngine

PHRASES = [
    "Hello",
    "How are you?",
    "What is Python?",
    "My name is Hassan",
]
CAPTURE_SECONDS = 5.0


def capture_phrase() -> list[np.ndarray]:
    capture = AudioCaptureManager(device=AudioCaptureManager.find_wasapi_device())
    capture.start()
    frames = []
    deadline = time.perf_counter() + CAPTURE_SECONDS
    while time.perf_counter() < deadline:
        frame = capture.get_frame(timeout=0.2)
        if frame is not None:
            frames.append(frame)
    capture.stop()
    if not frames:
        raise RuntimeError("No microphone frames captured")
    return frames


def extract_speech(frames: list[np.ndarray]) -> np.ndarray:
    vad = SileroVAD()
    for frame in frames:
        _, _, completed = vad.step(frame)
        if completed is not None:
            return completed.audio.astype(np.float32)
    raise RuntimeError("Silero VAD did not emit a speech segment")


def main() -> None:
    print("Live STT benchmark: speak each phrase immediately after pressing Enter.")
    print(f"Each capture lasts {CAPTURE_SECONDS:.1f} seconds.\n")
    recordings = []
    for phrase in PHRASES:
        input(f"Press Enter, then say: {phrase!r}\n")
        frames = capture_phrase()
        audio = extract_speech(frames)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        print(f"Speech segment {len(audio) / 16000:.2f}s, rms={rms:.6f}, peak={peak:.6f}\n")
        recordings.append(audio)

    engines = {
        "tiny": WhisperEngine("tiny", download_root="models/whisper", beam_size=1),
        "base": WhisperEngine("base", download_root="models/whisper", beam_size=1),
    }
    for index, (expected, audio) in enumerate(zip(PHRASES, recordings), start=1):
        print(f"[{index}] expected={expected!r}")
        for name, engine in engines.items():
            result = engine.transcribe(audio, language="en")
            print(
                f"    {name}: text={result.text!r} raw={result.raw_text!r} "
                f"language={result.language} probability={result.language_probability:.4f} "
                f"latency_ms={result.latency_ms:.1f} rtf={result.rtf:.3f}"
            )


if __name__ == "__main__":
    main()
