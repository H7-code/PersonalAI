"use client";

import React, { useEffect, useState } from "react";

interface NodeInfo {
  id: string;
  name: string;
  engine: string;
  hardware: string;
  vram_mb: number;
  status: string;
  role: string;
}

const DEFAULT_NODES: NodeInfo[] = [
  {
    id: "stt",
    name: "Speech-to-Text (STT)",
    engine: "Faster-Whisper tiny",
    hardware: "CPU (INT8, 4 threads)",
    vram_mb: 0,
    status: "ONLINE",
    role: "Multilingual transcription (EN, UR, MG), RTF 0.32x",
  },
  {
    id: "router",
    name: "Language Router",
    engine: "Deterministic Heuristic",
    hardware: "CPU (< 2 ms)",
    vram_mb: 0,
    status: "ONLINE",
    role: "English vs Roman Urdu vs Minglish classification",
  },
  {
    id: "llm",
    name: "LLM Reasoning Engine",
    engine: "DeepSeek-R1 1.5B (Q4_K_M)",
    hardware: "GPU (NVIDIA RTX 3050)",
    vram_mb: 1139,
    status: "ONLINE",
    role: "num_predict=512, think=false, token streaming",
  },
  {
    id: "output_guard",
    name: "Streaming Output Guard",
    engine: "Custom Regex State Machine",
    hardware: "CPU (< 1 ms)",
    vram_mb: 0,
    status: "ONLINE",
    role: "0% think leakage, space-preserving markdown strip, max 3 sents",
  },
  {
    id: "sentence_buffer",
    name: "Sentence Buffer",
    engine: "Lookahead Boundary Detector",
    hardware: "CPU (< 1 ms)",
    vram_mb: 0,
    status: "ONLINE",
    role: "Sentence boundary dispatch with decimal & abbreviation protection",
  },
  {
    id: "tts_en",
    name: "English TTS Engine",
    engine: "Piper (en_US-lessac-medium)",
    hardware: "CPU (ONNX Runtime, 22050 Hz)",
    vram_mb: 0,
    status: "ONLINE",
    role: "Warm latency ~128 ms, 0 MB VRAM",
  },
  {
    id: "tts_ur",
    name: "Urdu / Minglish TTS Engine",
    engine: "MMS-TTS Latin (PyTorch VITS)",
    hardware: "CPU (PyTorch, 16000 Hz)",
    vram_mb: 0,
    status: "ONLINE",
    role: "Warm latency ~1060 ms, 0 MB VRAM",
  },
  {
    id: "audio_vad",
    name: "Audio Capture & VAD",
    engine: "PortAudio WASAPI + Silero VAD v5",
    hardware: "CPU (Real-time thread)",
    vram_mb: 0,
    status: "ONLINE",
    role: "16 kHz mono, 32 ms frames, 200 ms Echo Gate holdoff",
  },
  {
    id: "state_manager",
    name: "Central State Manager",
    engine: "AppStateManager Concurrency Controller",
    hardware: "CPU (Lock-synchronized)",
    vram_mb: 0,
    status: "ONLINE",
    role: "Monotonic request_id cancellation, deadlock-free transitions",
  },
];

export default function NodesView() {
  const [nodes, setNodes] = useState<NodeInfo[]>(DEFAULT_NODES);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/nodes")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && data.nodes) setNodes(data.nodes);
      })
      .catch((e) => console.log("Using default node topology:", e));
  }, []);

  return (
    <div className="w-full max-w-5xl mx-auto py-6 px-4">
      <div className="bg-white/95 backdrop-blur-md rounded-2xl border border-slate-200/90 shadow-xs p-6">
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <div className="flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <h2 className="text-base font-semibold text-slate-900 font-sans">
              ARIA V3 Pipeline Nodes & Hardware Allocation
            </h2>
          </div>
          <span className="text-xs font-mono text-emerald-700 bg-emerald-50 border border-emerald-200 rounded px-2 py-0.5 font-semibold">
            All Nodes Offline Local
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-6">
          {nodes.map((node) => (
            <div
              key={node.id}
              className="p-4 rounded-xl bg-slate-50/80 border border-slate-100 hover:border-emerald-200 transition-all flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-semibold text-slate-900">
                    {node.name}
                  </span>
                  <span className="inline-flex items-center gap-1 text-[10px] font-mono font-semibold text-emerald-700 bg-emerald-100/70 rounded px-1.5 py-0.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    {node.status}
                  </span>
                </div>

                <div className="mt-2 text-xs text-slate-600 font-medium">
                  Engine: <span className="font-mono text-slate-800">{node.engine}</span>
                </div>

                <div className="mt-1 text-xs text-slate-500">
                  Role: <span className="text-slate-700">{node.role}</span>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-200/60 flex items-center justify-between text-[11px] font-mono text-slate-500">
                <span>{node.hardware}</span>
                <span className="font-semibold text-slate-700">
                  {node.vram_mb > 0 ? `${node.vram_mb} MB VRAM` : "0 MB VRAM"}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
