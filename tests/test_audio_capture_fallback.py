import types

from src.audio import capture as capture_module
from src.audio.capture import AudioCaptureManager


def test_list_input_devices_prefers_soundcard_windows_backend(monkeypatch):
    fake_microphone = types.SimpleNamespace(
        name="Microphone Array (Intel® Smart Sound Technology (Intel® SST))",
        id="mic-1",
        channels=1,
    )
    fake_soundcard = types.SimpleNamespace(
        all_microphones=lambda include_loopback=False: [fake_microphone],
        default_microphone=lambda: fake_microphone,
    )

    monkeypatch.setattr(capture_module, "sc", fake_soundcard, raising=False)
    monkeypatch.setattr(capture_module, "sd", None, raising=False)

    devices = AudioCaptureManager.list_input_devices()

    assert devices[0]["name"] == fake_microphone.name
    assert devices[0]["hostapi"] == "WASAPI"
    assert devices[0]["max_input_channels"] == 1


def test_find_wasapi_device_uses_default_soundcard_mic(monkeypatch):
    fake_microphone = types.SimpleNamespace(name="Microphone Array (Intel® Smart Sound Technology (Intel® SST))", id="mic-1", channels=1)
    fake_soundcard = types.SimpleNamespace(
        all_microphones=lambda include_loopback=False: [fake_microphone],
        default_microphone=lambda: fake_microphone,
    )

    monkeypatch.setattr(capture_module, "sc", fake_soundcard, raising=False)
    monkeypatch.setattr(capture_module, "sd", None, raising=False)

    assert AudioCaptureManager.find_wasapi_device() == 0
