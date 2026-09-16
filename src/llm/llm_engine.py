"""
ARIA v3 — LLM Engine
Model  : deepseek-r1:1.5b  (Q4_K_M GGUF via Ollama)
Device : GPU (CUDA) — no silent CPU fallback
Context: bounded sliding-window memory (max 6 turns)

Public API
----------
LLMEngine(config=None)          — instantiate (lazy model load)
engine.verify_connection()      — raise if Ollama unreachable
engine.check_vram_headroom()    — raise if free VRAM < 500 MB
engine.generate(text, lang)     — sync response str (strips <think>)
engine.generate_streaming(...)  — generator of token strings
engine.reset_memory()           — clear conversation history
engine.close()                  — release keep_alive (unload model)
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Generator, Optional

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL   = "http://127.0.0.1:11434"
MODEL_NAME        = "deepseek-r1:1.5b"
NUM_CTX           = 2048        # token context window
NUM_PREDICT       = 512         # internal LLM generation budget for DeepSeek-R1 reasoning
THINK_DISABLED    = True        # disable DeepSeek-R1 chain-of-thought (think:false)
KEEP_ALIVE        = "10m"       # keep model hot in VRAM
TIMEOUT_GENERATE  = 20          # V3 generation timeout (seconds)
MODEL_WARMUP_TIMEOUT = 180      # overall cold start warm-up budget (seconds)
MODEL_WARMUP_HTTP_TIMEOUT = 150 # one connected HTTP request for Ollama loading
MAX_HISTORY_TURNS = 6           # bounded sliding window (user+assistant pairs)
MIN_FREE_VRAM_MB  = 500         # headroom guard
MAX_SENTENCES     = 3           # V3 max spoken sentences
MAX_CHARS         = 400         # V3 max spoken characters

# Language-specific system prompts (ARIA V3 Master System Prompts)
_SYSTEM_PROMPTS: dict[str, str] = {
    "english": (
        "You are ARIA, a highly capable, concise, and natural voice assistant. "
        "The user is speaking to you via voice. Your responses will be read aloud by a text-to-speech engine. "
        "CONSTRAINTS: "
        "1. Speak in natural, grammatically correct English. "
        "2. Be extremely concise: reply in 1 to 2 sentences (maximum 3 sentences, under 500 characters). "
        "3. Never use markdown formatting, bullet points, asterisks, bold text, or numbered lists. "
        "4. Never output code blocks, URLs, or technical syntax unless specifically asked. "
        "5. Keep your tone warm, professional, and helpful. "
        "6. Do not include reasoning or planning text. Answer directly."
        " Never answer in Spanish or another unrelated language."
    ),
    "urdu": (
        "Aap ARIA hain, aik intahai qabil, seedhi aur natural bolnay wali voice assistant. "
        "User aap se voice ke zarye baat kar raha hai. "
        "ZAROORI HIDAYAAT: "
        "1. Sirf aur sirf ROMAN URDU (Latin script / English alphabet) mein baat karein. "
        "2. Arabic ya Nastaliq script ka istemal BILKUL MANA HAI. "
        "3. Jawab bohot mukhtasar hona chahiye: 1 se 2 jumlay (zyada se zyada 3 jumlay). "
        "4. Kisi qisam ki markdown, asterisks, bullets, ya headings ka istemal na karein. "
        "5. Seedha aur asan jawab dein. Koi sochne ya planning ka text shamil mat karen."
        " Sirf Roman Urdu mein jawab dein, Spanish ya kisi unrelated zaban mein nahi."
    ),
    "minglish": (
        "Aap ARIA hain, aik natural Pakistani multilingual voice assistant. "
        "User aap se Minglish (code-switched Roman Urdu aur English) mein baat kar raha hai. "
        "CONSTRAINTS & RULES: "
        "1. Jawab ROMAN URDU aur ENGLISH ke natural mix mein dein. "
        "2. Technical aur daily-use terms English mein hi rehne dein. "
        "3. Conversational grammar aur connecting words Roman Urdu mein hon. "
        "4. Arabic/Urdu script ka istemal SAKHT MANA HAI. Sirf Latin/English alphabet use karein. "
        "5. Jawab intehai mukhtasar rakhein (1 to 2 sentences, maximum 3 sentences). "
        "6. No markdown, no bullet points, no asterisks, no reasoning text."
        " Sirf natural English aur Roman Urdu use karein, Spanish ya kisi unrelated zaban mein nahi."
    ),
}
_DEFAULT_SYSTEM = _SYSTEM_PROMPTS["english"]

# Regex to strip DeepSeek-R1 <think>…</think> blocks (may span newlines)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Graceful conversational fallbacks for empty model output (e.g. thinking budget exhaustion)
FALLBACK_RESPONSES: dict[str, str] = {
    "english": "I'm sorry, I couldn't process that. Could you please say that again?",
    "urdu": "Mujhe samajh nahi aaya, barah-e-karam dobara kahiye.",
    "minglish": "Sorry, main samajh nahi paayi, please dobara repeat karein?",
}

def get_fallback_response(lang: str = "english") -> str:
    """Return graceful conversational fallback when model output is empty."""
    return FALLBACK_RESPONSES.get(lang.lower(), FALLBACK_RESPONSES["english"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    """Remove all <think>…</think> blocks and collapse whitespace."""
    cleaned = _THINK_RE.sub("", text)
    return cleaned.strip()


def _post_json(url: str, payload: dict, timeout: int = TIMEOUT_GENERATE) -> bytes:
    """HTTP POST with JSON payload; raises URLError on failure."""
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data    = data,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _get_json(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# LLMEngine
# ---------------------------------------------------------------------------

class LLMEngine:
    """
    Thin, dependency-free wrapper around the Ollama /api/chat endpoint
    for ARIA v3.  Uses CPU-free GPU inference; raises on CPU fallback.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.base_url    = cfg.get("base_url",    OLLAMA_BASE_URL)
        self.model       = cfg.get("model",        MODEL_NAME)
        self.num_ctx     = cfg.get("num_ctx",      NUM_CTX)
        self.num_predict = int(cfg.get("num_predict", NUM_PREDICT))
        self.keep_alive  = cfg.get("keep_alive",   KEEP_ALIVE)
        self.timeout     = cfg.get("timeout",      TIMEOUT_GENERATE)
        self.max_turns   = cfg.get("max_turns",    MAX_HISTORY_TURNS)
        # think=False disables DeepSeek-R1 CoT reasoning in production.
        # When True the model emits <think>...</think> before the answer;
        # with num_predict=150 it exhausted the budget inside the think block.
        self.think       = cfg.get("think",        not THINK_DISABLED)

        # Sliding-window conversation memory: deque of {"role":..., "content":...}
        # We store user+assistant pairs; max_turns*2 messages
        self._history: deque[dict] = deque(maxlen=self.max_turns * 2)
        self._lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._warmup_complete = threading.Event()

    # ------------------------------------------------------------------
    # Connection & VRAM checks
    # ------------------------------------------------------------------

    def verify_connection(self) -> dict:
        """
        Check Ollama is reachable and the target model is loaded.
        Returns the /api/tags response dict.
        Raises ConnectionError if not reachable or model absent.
        """
        try:
            data = _get_json(f"{self.base_url}/api/tags")
        except Exception as exc:
            raise ConnectionError(
                f"Ollama unreachable at {self.base_url}: {exc}"
            ) from exc

        models = [m["name"] for m in data.get("models", [])]
        # Accept both "deepseek-r1:1.5b" and tag variants like "deepseek-r1:1.5b-..."
        if not any(self.model.split(":")[0] in m for m in models):
            raise ConnectionError(
                f"Model '{self.model}' not found in Ollama. "
                f"Available: {models}"
            )
        return data

    def _model_status(self) -> Optional[dict]:
        """Return Ollama's status blob for the configured model, if present."""
        try:
            data = _get_json(f"{self.base_url}/api/ps", timeout=10)
        except Exception:
            return None

        for model in data.get("models", []):
            name = model.get("name", "")
            if self.model == name or self.model.split(":")[0] in name:
                return model
        return None

    def is_model_loading(self) -> bool:
        """True when Ollama is still warming or loading the model."""
        status = self._model_status()
        if status is None:
            return False
        state = str(status.get("status", "")).lower()
        return state in {
            "loading",
            "pending",
            "pulling",
            "creating",
            "starting",
            "unpacking",
            "warming",
        }

    def warmup(self, timeout_seconds: float = MODEL_WARMUP_TIMEOUT) -> bool:
        """Keep one minimal Ollama request connected until the cold model load completes."""
        if self._warmup_complete.is_set():
            return True

        with self._warmup_lock:
            if self._warmup_complete.is_set():
                return True

            http_timeout = min(MODEL_WARMUP_HTTP_TIMEOUT, max(1.0, timeout_seconds))
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "warmup"}],
                "stream": False,
                "think": False,
                "options": {"num_ctx": self.num_ctx, "num_predict": 1},
                "keep_alive": self.keep_alive,
            }
            try:
                _post_json(
                    f"{self.base_url}/api/chat",
                    payload,
                    timeout=http_timeout,
                )
            except urllib.error.URLError as exc:
                raise TimeoutError(
                    f"Ollama cold-start warm-up failed for {self.model} "
                    f"after {http_timeout:.1f}s"
                ) from exc
            self._warmup_complete.set()
            return True

    def ensure_ready_for_inference(self, warmup_timeout_seconds: float = MODEL_WARMUP_TIMEOUT) -> bool:
        """Warm the model specifically for cold-start; once ready, normal generation timeout remains in force."""
        if self._warmup_complete.is_set():
            return True
        return self.warmup(timeout_seconds=warmup_timeout_seconds)

    def check_vram_headroom(self) -> dict:
        """
        Query nvidia-smi for free VRAM.
        Returns {"free_mb": int, "used_mb": int, "total_mb": int}.
        Raises RuntimeError if free VRAM < MIN_FREE_VRAM_MB.
        Logs a warning (does not raise) if nvidia-smi is unavailable.
        """
        import subprocess, shutil
        result = {"free_mb": None, "used_mb": None, "total_mb": None,
                  "nvidia_smi_available": False}
        if not shutil.which("nvidia-smi"):
            print("[LLMEngine] WARNING: nvidia-smi not found; skipping VRAM check.")
            return result

        try:
            proc = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.free,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if proc.returncode != 0:
                print(f"[LLMEngine] nvidia-smi error: {proc.stderr.strip()}")
                return result

            parts = [int(x.strip()) for x in proc.stdout.strip().split(",")]
            free_mb, used_mb, total_mb = parts[0], parts[1], parts[2]
            result.update(free_mb=free_mb, used_mb=used_mb,
                          total_mb=total_mb, nvidia_smi_available=True)

            if free_mb < MIN_FREE_VRAM_MB:
                raise RuntimeError(
                    f"Insufficient free VRAM: {free_mb} MB < {MIN_FREE_VRAM_MB} MB required."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            print(f"[LLMEngine] VRAM check failed: {exc}")

        return result

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        """Clear all conversation history."""
        with self._lock:
            self._history.clear()

    def _build_messages(self, user_text: str, lang: str) -> list[dict]:
        """
        Assemble the full messages list:
          system prompt + bounded history + current user turn.
        """
        system_prompt = _SYSTEM_PROMPTS.get(lang.lower(), _DEFAULT_SYSTEM)
        messages = [{"role": "system", "content": system_prompt}]
        with self._lock:
            messages.extend(list(self._history))
        messages.append({"role": "user", "content": user_text})
        return messages

    def _record_turn(self, user_text: str, assistant_text: str) -> None:
        """Append user+assistant pair to sliding-window history."""
        with self._lock:
            self._history.append({"role": "user",      "content": user_text})
            self._history.append({"role": "assistant",  "content": assistant_text})

    # ------------------------------------------------------------------
    # Core generation — synchronous (non-streaming)
    # ------------------------------------------------------------------

    def generate(self, user_text: str, lang: str = "english") -> str:
        """
        Generate a response string.
        Strips <think>…</think> blocks.
        Records the turn in bounded memory.
        Raises on timeout or Ollama error.
        """
        messages = self._build_messages(user_text, lang)
        payload  = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "think":    self.think,   # False = disable DeepSeek-R1 CoT (V3 requirement)
            "options":  {
                "num_ctx":     self.num_ctx,
                "num_predict": self.num_predict,
            },
            "keep_alive": self.keep_alive,
        }

        t0 = time.perf_counter()
        try:
            raw = _post_json(
                f"{self.base_url}/api/chat",
                payload,
                timeout=self.timeout,
            )
        except urllib.error.URLError as exc:
            raise TimeoutError(
                f"LLM request timed out or connection failed: {exc}"
            ) from exc

        elapsed = time.perf_counter() - t0
        resp_data = json.loads(raw)

        raw_content = resp_data.get("message", {}).get("content", "")
        content     = _strip_think(raw_content)

        # Graceful fallback for empty model output (e.g. reasoning exhausted num_predict)
        if not content:
            content = get_fallback_response(lang)

        # Extract timing stats from response
        prompt_tokens    = resp_data.get("prompt_eval_count",    0)
        generated_tokens = resp_data.get("eval_count",           0)
        total_ns         = resp_data.get("total_duration",       0)
        first_token_ns   = resp_data.get("prompt_eval_duration", 0)
        gen_ns           = resp_data.get("eval_duration",        0)

        self._last_stats = {
            "elapsed_s":          elapsed,
            "prompt_tokens":      prompt_tokens,
            "generated_tokens":   generated_tokens,
            "total_duration_s":   total_ns   / 1e9 if total_ns   else elapsed,
            "first_token_s":      first_token_ns / 1e9 if first_token_ns else None,
            "gen_duration_s":     gen_ns / 1e9 if gen_ns else None,
            "tokens_per_sec":     (generated_tokens / (gen_ns / 1e9))
                                  if gen_ns and generated_tokens else None,
        }

        self._record_turn(user_text, content)
        return content

    # ------------------------------------------------------------------
    # Core generation — streaming
    # ------------------------------------------------------------------

    def generate_streaming(
        self,
        user_text: str,
        lang:      str = "english",
    ) -> Generator[str, None, None]:
        """
        Yield response tokens one at a time as they stream from Ollama.
        <think> blocks are buffered and stripped before yielding.
        Full response is recorded in memory after stream ends.
        """
        messages = self._build_messages(user_text, lang)
        payload  = {
            "model":    self.model,
            "messages": messages,
            "stream":   True,
            "think":    self.think,   # False = disable DeepSeek-R1 CoT (V3 requirement)
            "options":  {
                "num_ctx":     self.num_ctx,
                "num_predict": self.num_predict,
            },
            "keep_alive": self.keep_alive,
        }

        data    = json.dumps(payload).encode("utf-8")
        req     = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data    = data,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )

        accumulated = []
        think_depth = 0   # track open <think> nesting
        buffer      = ""

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("message", {}).get("content", "")
                if not token:
                    continue

                buffer += token

                # Process buffer: yield only non-think text
                while True:
                    if think_depth == 0:
                        open_idx = buffer.find("<think>")
                        if open_idx == -1:
                            # No think tag; yield everything except possible partial tag
                            safe_idx = max(0, len(buffer) - 7)
                            if safe_idx > 0:
                                out = buffer[:safe_idx]
                                accumulated.append(out)
                                yield out
                                buffer = buffer[safe_idx:]
                            break
                        else:
                            # Yield text before the <think>
                            if open_idx > 0:
                                out = buffer[:open_idx]
                                accumulated.append(out)
                                yield out
                            buffer = buffer[open_idx + 7:]
                            think_depth = 1
                    else:
                        close_idx = buffer.find("</think>")
                        if close_idx == -1:
                            break  # wait for more tokens
                        buffer = buffer[close_idx + 8:]
                        think_depth = 0

                if chunk.get("done", False):
                    break

        # Yield any remaining non-think text
        if buffer and think_depth == 0:
            accumulated.append(buffer)
            yield buffer

        full_response = _strip_think("".join(accumulated))
        # Graceful fallback for empty model output (e.g. reasoning exhausted num_predict)
        if not full_response:
            fallback = get_fallback_response(lang)
            accumulated.append(fallback)
            yield fallback
            full_response = fallback

        self._record_turn(user_text, full_response)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Ask Ollama to unload the model from VRAM immediately.
        Call when the application exits.
        """
        try:
            payload = {
                "model":      self.model,
                "keep_alive": "0",
                "messages":   [],
            }
            _post_json(f"{self.base_url}/api/chat", payload, timeout=5)
        except Exception:
            pass  # best-effort


# ---------------------------------------------------------------------------
# Module-level singleton helper (optional convenience)
# ---------------------------------------------------------------------------
_engine_instance: Optional[LLMEngine] = None

def get_engine(config: Optional[dict] = None) -> LLMEngine:
    """Return a module-level LLMEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LLMEngine(config)
    return _engine_instance
