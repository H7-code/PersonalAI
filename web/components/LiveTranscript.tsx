"use client";

import React from "react";
import { AriaState } from "./Header";

interface LiveTranscriptProps {
  state: AriaState;
  userTranscript: string;
  assistantText: string;
  speakingSentence: string;
  langMode: string;
  latencyMs: number | null;
}

export default function LiveTranscript({
  state,
  userTranscript,
  assistantText,
  speakingSentence,
  langMode,
  latencyMs,
}: LiveTranscriptProps) {
  const isSpeaking = state === "SPEAKING" || Boolean(speakingSentence);
  const isGenerating = state === "GENERATING";

  return (
    <section
      aria-label="Conversation Subtitle Stream"
      className="w-full max-w-xl mx-auto mt-10"
    >
      <div className="bg-white/90 backdrop-blur-md rounded-2xl border border-slate-200/90 shadow-xs p-5 transition-all hover:border-emerald-200">
        {/* Header Bar */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-100 text-xs">
          <span className="font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                state === "IDLE" || state === "READY"
                  ? "bg-emerald-500"
                  : state === "LISTENING"
                  ? "bg-emerald-500 animate-pulse"
                  : state === "TRANSCRIBING"
                  ? "bg-amber-500 animate-pulse"
                  : state === "THINKING" || state === "GENERATING"
                  ? "bg-sky-500 animate-pulse"
                  : state === "SPEAKING"
                  ? "bg-purple-500 animate-pulse"
                  : "bg-rose-500"
              }`}
            />
            Live Transcript
            {langMode && (
              <span className="ml-1.5 px-1.5 py-0.2 rounded bg-slate-100 text-slate-600 font-semibold text-[10px]">
                {langMode.toUpperCase()}
              </span>
            )}
          </span>
          <span className="text-slate-400 font-mono text-[11px]">
            {latencyMs ? `Latency: ${latencyMs.toFixed(0)}ms` : "Latency: 142ms"}
          </span>
        </div>

        {/* Content Box */}
        <div className="pt-3.5 space-y-3 text-left">
          {/* User Query */}
          <div className="flex items-start gap-3">
            <span className="text-[11px] font-mono text-emerald-700 bg-emerald-50 border border-emerald-100 rounded px-1.5 py-0.5 mt-0.5 shrink-0 font-medium">
              USER
            </span>
            <p className="text-sm font-medium text-slate-800">
              {userTranscript
                ? `"${userTranscript}"`
                : '"Analyze quarterly cloud cost anomalies and give recommendations."'}
            </p>
          </div>

          {/* AI Voice Response Preview */}
          <div className="flex items-start gap-3 bg-slate-50/70 p-3 rounded-xl border border-slate-100">
            <span className="text-[11px] font-mono text-teal-800 bg-teal-50 border border-teal-200/70 rounded px-1.5 py-0.5 mt-0.5 shrink-0 font-medium">
              ARIA AI
            </span>
            <div className="text-xs md:text-sm text-slate-600 leading-relaxed w-full">
              <p>
                {assistantText ||
                  "Compute instances accounted for 62% of the spike. Terminating 4 idle worker nodes saves an estimated 18% monthly."}
                {isGenerating && (
                  <span className="inline-block w-1.5 h-3.5 ml-1 bg-sky-500 animate-pulse align-middle" />
                )}
              </p>

              {/* Spoken voice feedback indicator */}
              {isSpeaking && (
                <div className="mt-2 flex items-center gap-2 text-emerald-600 text-xs font-medium">
                  <svg
                    className="w-3.5 h-3.5 animate-pulse"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      clipRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                      fillRule="evenodd"
                    />
                  </svg>
                  <span>
                    {speakingSentence
                      ? `Speaking: "${speakingSentence}"`
                      : "Playing synthesized voice output"}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
