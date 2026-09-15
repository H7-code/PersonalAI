"""
ARIA V3 — Central AppStateManager & Barge-In Cancellation Controller
Implements Phase 11 Central State Machine, Concurrency Hardening & Echo Gate:
- Centralized thread-safe application state (AppState enum)
- Strict State Transition Matrix enforcement
- Monotonically increasing request_id (race-free turn cancellation)
- Independent subsystem activity flags (llm_active, tts_active, playback_active)
- Application-Level Echo Gate: Half-duplex turn gating + 200 ms echo holdoff
- Fast barge-in cancellation (< 50 ms response time)
- Tagged queue draining and stale chunk rejection
- Zero usage of thread-local storage
"""

import enum
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple

logger = logging.getLogger("ARIA.StateManager")


class AppState(enum.Enum):
    INITIALIZING = "INITIALIZING"
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING_STT = "PROCESSING_STT"
    PROCESSING_LLM = "PROCESSING_LLM"
    SPEAKING = "SPEAKING"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


# Valid State Transition Matrix (ARIA V3 Section 25)
VALID_TRANSITIONS: Dict[AppState, Set[AppState]] = {
    AppState.INITIALIZING: {AppState.IDLE, AppState.ERROR},
    AppState.IDLE: {AppState.LISTENING, AppState.ERROR},
    AppState.LISTENING: {AppState.PROCESSING_STT, AppState.IDLE, AppState.CANCELLED, AppState.ERROR},
    AppState.PROCESSING_STT: {AppState.PROCESSING_LLM, AppState.IDLE, AppState.CANCELLED, AppState.ERROR},
    AppState.PROCESSING_LLM: {AppState.SPEAKING, AppState.IDLE, AppState.CANCELLED, AppState.ERROR},
    AppState.SPEAKING: {AppState.IDLE, AppState.LISTENING, AppState.CANCELLED, AppState.ERROR},
    AppState.CANCELLED: {AppState.IDLE, AppState.LISTENING, AppState.ERROR},
    AppState.ERROR: {AppState.IDLE, AppState.INITIALIZING},
}


class AppStateManager:
    """
    Thread-safe central state manager for ARIA V3.
    Enforces monotonic request_id cancellation, State Transition Matrix,
    and half-duplex turn gating (Echo Gate) to eliminate acoustic feedback.
    """

    ECHO_HOLDOFF_SECONDS: float = 0.200  # 200 ms post-playback echo holdoff

    def __init__(self, echo_holdoff_seconds: float = ECHO_HOLDOFF_SECONDS):
        self._lock = threading.Lock()
        self.state = AppState.INITIALIZING
        self.current_request_id = 0
        self.cancel_event = threading.Event()

        # Independent subsystem activity flags (V3 Section 1 & 25)
        self.llm_active = False
        self.tts_active = False
        self.playback_active = False

        # Echo Gate tracking (V3 Section 13 & Phase 10 Objective)
        self.echo_holdoff_seconds = echo_holdoff_seconds
        self.last_playback_end_time: float = 0.0

        # Queues managed across pipeline stages
        self.audio_capture_queue = queue.Queue(maxsize=100)
        self.stt_queue = queue.Queue(maxsize=10)
        self.tts_input_queue = queue.Queue(maxsize=20)
        self.audio_playback_queue = queue.Queue(maxsize=20)

        # Optional hardware stop callback (e.g. sounddevice.stop)
        self.hardware_stop_cb: Optional[Callable[[], None]] = None

    def set_hardware_stop_callback(self, cb: Callable[[], None]):
        """Register audio output hardware halt callback."""
        self.hardware_stop_cb = cb

    def new_request(self) -> int:
        """Start a new turn, returning the new monotonic request_id."""
        with self._lock:
            self.current_request_id += 1
            self.cancel_event.clear()
            self.state = AppState.LISTENING
            return self.current_request_id

    def set_state(self, new_state: AppState):
        """Update current application state directly under lock."""
        with self._lock:
            self.state = new_state

    def transition_to(self, new_state: AppState) -> bool:
        """
        Transition application state strictly according to the State Transition Matrix.
        Raises ValueError if transition is invalid.
        """
        with self._lock:
            allowed = VALID_TRANSITIONS.get(self.state, set())
            if new_state not in allowed:
                err_msg = f"Invalid state transition: {self.state.value} -> {new_state.value}"
                logger.error(err_msg)
                raise ValueError(err_msg)
            self.state = new_state
            return True

    def get_state(self) -> AppState:
        with self._lock:
            return self.state

    def set_llm_active(self, active: bool):
        with self._lock:
            self.llm_active = active

    def set_tts_active(self, active: bool):
        with self._lock:
            self.tts_active = active

    def set_playback_active(self, active: bool):
        """
        Set playback active flag. When transitioning from True to False,
        records last_playback_end_time to enforce the 200 ms echo holdoff.
        """
        with self._lock:
            was_active = self.playback_active
            self.playback_active = active
            if was_active and not active:
                self.last_playback_end_time = time.perf_counter()

    def should_discard_audio_frame(self) -> bool:
        """
        Application-Level Echo Gate (Original Phase 10 Objective):
        Determines whether an incoming microphone frame must be discarded to prevent
        self-transcription and echo loops.
        Returns True if:
          1. playback_active is True, OR
          2. state == AppState.SPEAKING, OR
          3. Within the 200 ms post-playback echo holdoff window.
        """
        with self._lock:
            if self.playback_active or self.state == AppState.SPEAKING:
                return True
            if self.last_playback_end_time > 0:
                elapsed = time.perf_counter() - self.last_playback_end_time
                if elapsed < self.echo_holdoff_seconds:
                    return True
            return False

    def is_request_valid(self, request_id: int) -> bool:
        """Check if request_id matches active turn and is not cancelled."""
        with self._lock:
            return (request_id == self.current_request_id) and not self.cancel_event.is_set()

    def cancel_active_request(self, reason: str = "barge_in") -> Tuple[int, float]:
        """
        Deterministically cancel the active turn without early-reset races.
        Returns (cancelled_request_id, cancellation_latency_ms).
        """
        t0 = time.perf_counter()
        with self._lock:
            cancelled_id = self.current_request_id
            self.cancel_event.set()
            self.state = AppState.CANCELLED

        # 1. Terminate physical hardware output immediately
        if self.hardware_stop_cb:
            try:
                self.hardware_stop_cb()
            except Exception as e:
                logger.error(f"Hardware stop callback error: {e}")

        # 2. Drain all pending queues matching the cancelled request_id
        self._drain_queue(self.audio_capture_queue)
        self._drain_queue(self.stt_queue)
        self._drain_tagged_queue(self.tts_input_queue, cancelled_id)
        self._drain_tagged_queue(self.audio_playback_queue, cancelled_id)

        # 3. Increment request_id so late worker outputs are rejected
        with self._lock:
            self.current_request_id += 1
            self.llm_active = False
            self.tts_active = False
            self.playback_active = False
            self.last_playback_end_time = time.perf_counter()

            # Reset cancel_event only AFTER all cleanup is complete
            self.cancel_event.clear()
            self.state = AppState.LISTENING

        latency_ms = (time.perf_counter() - t0) * 1000
        return cancelled_id, latency_ms

    def get_snapshot(self) -> dict:
        """Thread-safe snapshot of state machine and queue depths for UI/Telemetry."""
        with self._lock:
            return {
                "state": self.state.value,
                "request_id": self.current_request_id,
                "llm_active": self.llm_active,
                "tts_active": self.tts_active,
                "playback_active": self.playback_active,
                "cancelled": self.cancel_event.is_set(),
                "echo_gate_active": (
                    self.playback_active
                    or self.state == AppState.SPEAKING
                    or ((time.perf_counter() - self.last_playback_end_time) < self.echo_holdoff_seconds
                        if self.last_playback_end_time > 0 else False)
                ),
                "audio_queue_depth": self.audio_capture_queue.qsize(),
                "stt_queue_depth": self.stt_queue.qsize(),
                "tts_queue_depth": self.tts_input_queue.qsize(),
                "playback_queue_depth": self.audio_playback_queue.qsize(),
            }

    @staticmethod
    def _drain_queue(q: queue.Queue) -> int:
        """Drain all items from an untagged queue."""
        count = 0
        while True:
            try:
                q.get_nowait()
                q.task_done()
                count += 1
            except (queue.Empty, ValueError):
                break
        return count

    @staticmethod
    def _drain_tagged_queue(q: queue.Queue, target_request_id: int) -> int:
        """
        Drain items matching target_request_id from a tagged queue ((req_id, item)).
        Retains items belonging to other requests.
        """
        retained = []
        drained_count = 0
        while True:
            try:
                item = q.get_nowait()
                if isinstance(item, tuple) and len(item) == 2 and item[0] == target_request_id:
                    drained_count += 1
                else:
                    retained.append(item)
                q.task_done()
            except (queue.Empty, ValueError):
                break

        for item in retained:
            q.put_nowait(item)

        return drained_count


# Module-level singleton
_state_manager_instance: Optional[AppStateManager] = None

def get_state_manager() -> AppStateManager:
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = AppStateManager()
    return _state_manager_instance
