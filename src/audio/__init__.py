"""
ARIA Audio Subsystem:
- Capture via sounddevice / PortAudio WASAPI
- Silero VAD v5 ONNX state machine with RMS energy pre-filter
"""
from src.audio.capture import AudioCaptureManager
from src.audio.vad import SileroVAD, VADState, SpeechSegment

__all__ = ["AudioCaptureManager", "SileroVAD", "VADState", "SpeechSegment"]
