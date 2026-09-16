// ARIA V3 — Vanilla JavaScript Interface Logic (Zero External CDNs)
(function () {
  let ws = null;
  let isPttActive = false;
  let isSpacePressed = false;
  let currentAssistantMsgEl = null;

  // DOM Elements
  const statusBadge = document.getElementById("status-badge");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const conversationFeed = document.getElementById("conversation-feed");
  const pttButton = document.getElementById("ptt-btn");
  const pttLabel = document.getElementById("ptt-label");
  const cancelBtn = document.getElementById("cancel-btn");
  
  // HUD Elements
  const hudState = document.getElementById("hud-state");
  const hudRequestId = document.getElementById("hud-request-id");
  const hudLang = document.getElementById("hud-lang");
  const hudTtft = document.getElementById("hud-ttft");
  const hudEchoGate = document.getElementById("hud-echo-gate");
  const hudLatency = document.getElementById("hud-latency");

  const STATE_COLORS = {
    INITIALIZING: "#64748b",
    WARMING: "#f59e0b",
    IDLE: "#64748b",
    LISTENING: "#10b981",
    PROCESSING_STT: "#f59e0b",
    PROCESSING_LLM: "#00d2ff",
    SPEAKING: "#9d4edd",
    CANCELLED: "#f43f5e",
    ERROR: "#ef4444",
  };

  function updateStatus(stateName) {
    const color = STATE_COLORS[stateName] || "#64748b";
    statusText.textContent = stateName;
    statusDot.style.background = color;
    statusDot.style.boxShadow = `0 0 8px ${color}`;
    if (hudState) hudState.textContent = stateName;
  }

  function appendMessage(role, text, langMode = "") {
    const card = document.createElement("div");
    card.className = `message-card message-${role}`;
    
    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = text;
    card.appendChild(body);

    const meta = document.createElement("div");
    meta.className = "message-meta";
    const now = new Date().toLocaleTimeString();
    meta.textContent = langMode ? `${now} • [${langMode.toUpperCase()}]` : now;
    card.appendChild(meta);

    conversationFeed.appendChild(card);
    conversationFeed.scrollTop = conversationFeed.scrollHeight;
    return body;
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("[ARIA UI] WebSocket connected to", wsUrl);
      updateStatus("IDLE");
      if (hudLatency) hudLatency.textContent = "< 5 ms";
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
      } catch (err) {
        console.error("[ARIA UI] Error parsing WS message:", err);
      }
    };

    ws.onclose = () => {
      console.log("[ARIA UI] WebSocket disconnected. Retrying in 1.5s...");
      updateStatus("INITIALIZING");
      setTimeout(connectWebSocket, 1500);
    };

    ws.onerror = (err) => {
      console.error("[ARIA UI] WebSocket error:", err);
    };
  }

  function handleServerMessage(msg) {
    switch (msg.type) {
      case "state_update":
        updateStatus(msg.state);
        if (msg.request_id && hudRequestId) {
          hudRequestId.textContent = `#${msg.request_id}`;
        }
        if (hudEchoGate) {
          hudEchoGate.textContent = msg.echo_gate_active ? "GATED (Active)" : "OPEN";
          hudEchoGate.style.color = msg.echo_gate_active ? "var(--accent-rose)" : "var(--accent-emerald)";
        }
        break;

      case "user_transcript":
        appendMessage("user", msg.text, msg.lang);
        if (hudLang && msg.lang) hudLang.textContent = msg.lang.toUpperCase();
        // Prepare assistant bubble for upcoming stream
        currentAssistantMsgEl = appendMessage("assistant", "...", msg.lang);
        break;

      case "assistant_chunk":
        if (!currentAssistantMsgEl) {
          currentAssistantMsgEl = appendMessage("assistant", msg.chunk);
        } else {
          if (currentAssistantMsgEl.textContent === "...") {
            currentAssistantMsgEl.textContent = msg.chunk;
          } else {
            currentAssistantMsgEl.textContent += msg.chunk;
          }
        }
        conversationFeed.scrollTop = conversationFeed.scrollHeight;
        break;

      case "assistant_final":
        if (currentAssistantMsgEl) {
          if (msg.text) currentAssistantMsgEl.textContent = msg.text;
          currentAssistantMsgEl = null;
        }
        break;

      case "telemetry":
        if (hudTtft && msg.ttft_ms) {
          hudTtft.textContent = `${msg.ttft_ms.toFixed(0)} ms`;
        }
        break;

      case "pong":
        break;
    }
  }

  function startPtt() {
    if (isPttActive) return;
    isPttActive = true;
    pttButton.classList.add("active");
    pttLabel.textContent = "Listening... (Release to Send)";
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ptt_start", timestamp: Date.now() }));
    }
  }

  function stopPtt() {
    if (!isPttActive) return;
    isPttActive = false;
    pttButton.classList.remove("active");
    pttLabel.textContent = "Hold [Space] or Click to Speak";
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ptt_stop", timestamp: Date.now() }));
    }
  }

  function sendCancel() {
    console.log("[ARIA UI] Sending turn cancellation / barge-in");
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "cancel", reason: "user_ui_cancel" }));
    }
    updateStatus("CANCELLED");
  }

  // Event Listeners: Mouse & Touch PTT
  pttButton.addEventListener("mousedown", (e) => {
    e.preventDefault();
    startPtt();
  });
  pttButton.addEventListener("mouseup", (e) => {
    e.preventDefault();
    stopPtt();
  });
  pttButton.addEventListener("mouseleave", () => {
    if (isPttActive) stopPtt();
  });

  pttButton.addEventListener("touchstart", (e) => {
    e.preventDefault();
    startPtt();
  });
  pttButton.addEventListener("touchend", (e) => {
    e.preventDefault();
    stopPtt();
  });

  // Event Listener: Cancel Button
  cancelBtn.addEventListener("click", (e) => {
    e.preventDefault();
    sendCancel();
  });

  // Keyboard Shortcuts: Spacebar (PTT) & Escape (Cancel)
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !e.repeat && !isSpacePressed) {
      // Don't trigger if user is focused on an input element
      if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
      e.preventDefault();
      isSpacePressed = true;
      startPtt();
    } else if (e.code === "Escape") {
      e.preventDefault();
      sendCancel();
    }
  });

  window.addEventListener("keyup", (e) => {
    if (e.code === "Space" && isSpacePressed) {
      e.preventDefault();
      isSpacePressed = false;
      stopPtt();
    }
  });

  // Keep-alive Ping every 5s
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ping" }));
    }
  }, 5000);

  // Initialize Connection
  connectWebSocket();
})();
