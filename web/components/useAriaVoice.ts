"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { AriaState } from "./Header";

export interface AriaVoiceHook {
  state: AriaState;
  userTranscript: string;
  assistantText: string;
  speakingSentence: string;
  langMode: string;
  selectedMode: string;
  setSelectedMode: (mode: string) => void;
  latencyMs: number | null;
  isPttActive: boolean;
  isConnected: boolean;
  startPtt: () => void;
  stopPtt: () => void;
  interrupt: () => void;
  simulateTurn: (text?: string, lang?: string) => void;
}

export function useAriaVoice(): AriaVoiceHook {
  const [state, setState] = useState<AriaState>("READY");
  const [userTranscript, setUserTranscript] = useState<string>("");
  const [assistantText, setAssistantText] = useState<string>("");
  const [speakingSentence, setSpeakingSentence] = useState<string>("");
  const [langMode, setLangMode] = useState<string>("ENGLISH");
  const [selectedMode, setSelectedMode] = useState<string>("auto");
  const [latencyMs, setLatencyMs] = useState<number | null>(142);
  const [isPttActive, setIsPttActive] = useState<boolean>(false);
  const [isConnected, setIsConnected] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const isSpacePressed = useRef<boolean>(false);
  const connectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isUnmountedRef = useRef<boolean>(false);
  const connectWsRef = useRef<(() => void) | null>(null);

  const connectWs = useCallback(() => {
    if (isUnmountedRef.current) return;

    try {
      const ws = new WebSocket("ws://127.0.0.1:8000/ws");
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("[ARIA UI] WebSocket connected to 127.0.0.1:8000/ws");
        setIsConnected(true);
        setState("READY");
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          if (msg.type === "state_update") {
            const rawState = msg.state as string;
            if (rawState === "INITIALIZING" || rawState === "IDLE") {
              setState("READY");
            } else {
              setState(rawState as AriaState);
            }
            if (msg.cancellation_latency_ms) {
              setLatencyMs(msg.cancellation_latency_ms);
            }
          } else if (msg.type === "user_transcript") {
            setUserTranscript(msg.text);
            if (msg.lang) setLangMode(msg.lang.toUpperCase());
            // Clear prior assistant stream
            setAssistantText("");
            setSpeakingSentence("");
          } else if (msg.type === "assistant_chunk") {
            setAssistantText((prev) => prev + msg.chunk);
          } else if (msg.type === "assistant_final") {
            if (msg.text) setAssistantText(msg.text);
            if (msg.lang) setLangMode(msg.lang.toUpperCase());
          } else if (msg.type === "sentence_speaking") {
            setSpeakingSentence(msg.is_speaking ? msg.sentence : "");
          } else if (msg.type === "telemetry") {
            if (msg.ttft_ms) setLatencyMs(msg.ttft_ms);
          }
        } catch (err) {
          console.error("[ARIA UI] Error parsing WS payload:", err);
        }
      };

      ws.onclose = () => {
        if (wsRef.current !== ws || isUnmountedRef.current) return;
        setIsConnected(false);
        wsRef.current = null;
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectTimeoutRef.current = null;
          connectWsRef.current?.();
        }, 2000);
      };

      ws.onerror = (err) => {
        if (wsRef.current !== ws || isUnmountedRef.current) return;
        console.warn("[ARIA UI] WebSocket error:", err);
      };
    } catch (e) {
      console.error("[ARIA UI] Could not open WebSocket:", e);
    }
  }, []);

  useEffect(() => {
    isUnmountedRef.current = false;
    connectWsRef.current = connectWs;
    connectTimeoutRef.current = setTimeout(() => {
      connectTimeoutRef.current = null;
      connectWs();
    }, 0);
    return () => {
      isUnmountedRef.current = true;
      connectWsRef.current = null;
      if (connectTimeoutRef.current) {
        clearTimeout(connectTimeoutRef.current);
        connectTimeoutRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (wsRef.current) wsRef.current.close();
    };
  }, [connectWs]);

  const startPtt = useCallback(() => {
    setIsPttActive(true);
    setState("LISTENING");
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "ptt_start", mode: selectedMode }));
    }
  }, [selectedMode]);

  const stopPtt = useCallback(() => {
    setIsPttActive(false);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "ptt_stop" }));
    }
  }, []);

  const interrupt = useCallback(() => {
    setState("INTERRUPTED");
    setSpeakingSentence("");
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "cancel" }));
    }
  }, []);

  const simulateTurn = useCallback((text?: string, lang?: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "simulate_turn",
          text: text || "Analyze quarterly cloud cost anomalies and give recommendations.",
          lang: lang || "english",
        })
      );
    }
  }, []);

  // Global Keyboard Shortcuts: Space (PTT) & Esc (Interrupt)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat && !isSpacePressed.current) {
        if (["INPUT", "TEXTAREA"].includes((document.activeElement as HTMLElement)?.tagName)) {
          return;
        }
        e.preventDefault();
        isSpacePressed.current = true;
        startPtt();
      } else if (e.code === "Escape") {
        e.preventDefault();
        interrupt();
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space" && isSpacePressed.current) {
        e.preventDefault();
        isSpacePressed.current = false;
        stopPtt();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [startPtt, stopPtt, interrupt]);

  return {
    state,
    userTranscript,
    assistantText,
    speakingSentence,
    langMode,
    selectedMode,
    setSelectedMode,
    latencyMs,
    isPttActive,
    isConnected,
    startPtt,
    stopPtt,
    interrupt,
    simulateTurn,
  };
}
