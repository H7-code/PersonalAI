"use client";

import React, { useState } from "react";

export type AriaState =
  | "IDLE"
  | "READY"
  | "LISTENING"
  | "TRANSCRIBING"
  | "THINKING"
  | "GENERATING"
  | "SPEAKING"
  | "INTERRUPTED"
  | "ERROR";

interface HeaderProps {
  state: AriaState;
  selectedMic: string;
  devices: Array<{ index: number; name: string }>;
  onSelectMic: (micName: string) => void;
  onToggleSettings?: () => void;
}

const STATE_CONFIG: Record<
  AriaState,
  { label: string; dotColor: string; pingColor: string; textColor: string }
> = {
  IDLE: {
    label: "READY",
    dotColor: "bg-emerald-500",
    pingColor: "bg-emerald-400",
    textColor: "text-emerald-800",
  },
  READY: {
    label: "READY",
    dotColor: "bg-emerald-500",
    pingColor: "bg-emerald-400",
    textColor: "text-emerald-800",
  },
  LISTENING: {
    label: "LISTENING",
    dotColor: "bg-emerald-500",
    pingColor: "bg-emerald-400",
    textColor: "text-emerald-800",
  },
  TRANSCRIBING: {
    label: "TRANSCRIBING",
    dotColor: "bg-amber-500",
    pingColor: "bg-amber-400",
    textColor: "text-amber-800",
  },
  THINKING: {
    label: "THINKING",
    dotColor: "bg-teal-500",
    pingColor: "bg-teal-400",
    textColor: "text-teal-800",
  },
  GENERATING: {
    label: "GENERATING",
    dotColor: "bg-sky-500",
    pingColor: "bg-sky-400",
    textColor: "text-sky-800",
  },
  SPEAKING: {
    label: "SPEAKING",
    dotColor: "bg-purple-500",
    pingColor: "bg-purple-400",
    textColor: "text-purple-800",
  },
  INTERRUPTED: {
    label: "INTERRUPTED",
    dotColor: "bg-rose-500",
    pingColor: "bg-rose-400",
    textColor: "text-rose-800",
  },
  ERROR: {
    label: "ERROR",
    dotColor: "bg-red-500",
    pingColor: "bg-red-400",
    textColor: "text-red-800",
  },
};

export default function Header({
  state,
  selectedMic,
  devices,
  onSelectMic,
  onToggleSettings,
}: HeaderProps) {
  const [showMicDropdown, setShowMicDropdown] = useState(false);
  const currentConfig = STATE_CONFIG[state] || STATE_CONFIG.READY;

  return (
    <header className="w-full border-b border-slate-200/70 bg-white/70 backdrop-blur-md sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Brand & Stage Identity */}
        <div className="flex items-center gap-3">
          <div
            aria-hidden="true"
            className="w-9 h-9 rounded-xl bg-emerald-50 border border-emerald-200/60 flex items-center justify-center text-emerald-600 shadow-xs"
          >
            {/* Multi-bar audio icon */}
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeWidth="2.2"
              viewBox="0 0 24 24"
            >
              <path d="M4 10v4" />
              <path d="M8 7v10" />
              <path d="M12 4v16" />
              <path d="M16 8v8" />
              <path d="M20 11v2" />
            </svg>
          </div>
          <div>
            <span className="text-base font-semibold tracking-tight text-slate-900 block leading-tight">
              ARIA
            </span>
            <span className="text-[11px] font-mono text-slate-400 font-normal">
              Adaptive Real-time Intelligent Assistant • Session #204
            </span>
          </div>
        </div>

        {/* Center Status Capsule */}
        <div className="flex items-center">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/90 border border-emerald-200 shadow-xs">
            <span className="relative flex h-2 w-2">
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full ${currentConfig.pingColor} opacity-75`}
              />
              <span
                className={`relative inline-flex rounded-full h-2 w-2 ${currentConfig.dotColor}`}
              />
            </span>
            <span
              className={`font-mono text-xs font-semibold tracking-widest ${currentConfig.textColor}`}
            >
              {currentConfig.label}
            </span>
          </div>
        </div>

        {/* Right Control Actions */}
        <div className="flex items-center gap-3">
          {/* Microphone Input Selector */}
          <div className="relative hidden sm:block">
            <button
              onClick={() => setShowMicDropdown(!showMicDropdown)}
              className="inline-flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-slate-600 bg-white border border-slate-200 rounded-lg hover:border-emerald-300 hover:text-slate-900 transition-colors shadow-xs cursor-pointer"
              type="button"
            >
              <svg
                className="w-3.5 h-3.5 text-emerald-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 02-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                />
              </svg>
              <span className="max-w-[130px] truncate">{selectedMic}</span>
              <svg
                className="w-3 h-3 text-slate-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  d="M19 9l-7 7-7-7"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                />
              </svg>
            </button>

            {showMicDropdown && (
              <div className="absolute right-0 mt-1 w-56 bg-white border border-slate-200 rounded-lg shadow-lg py-1 z-50 text-xs font-mono">
                {devices.length > 0 ? (
                  devices.map((d) => (
                    <button
                      key={d.index}
                      onClick={() => {
                        onSelectMic(d.name);
                        setShowMicDropdown(false);
                      }}
                      className={`w-full text-left px-3 py-2 hover:bg-emerald-50 hover:text-emerald-900 truncate block ${
                        selectedMic === d.name
                          ? "bg-emerald-50 text-emerald-800 font-semibold"
                          : "text-slate-700"
                      }`}
                    >
                      {d.name}
                    </button>
                  ))
                ) : (
                  <div className="px-3 py-2 text-slate-400">
                    Spatial Array (Default)
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Stage Parameter Settings Icon */}
          <button
            onClick={onToggleSettings}
            aria-label="Acoustic tuning settings"
            className="w-9 h-9 flex items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 hover:text-slate-900 hover:border-slate-300 transition-colors shadow-xs cursor-pointer"
            type="button"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
            </svg>
          </button>

          {/* User Profile Avatar Pill (Local User) */}
          <div
            title="Local Offline Mode"
            className="h-9 w-9 rounded-full bg-emerald-100 border border-emerald-300/80 flex items-center justify-center text-emerald-800 font-semibold text-xs shadow-xs cursor-pointer"
          >
            <svg
              className="w-4 h-4 text-emerald-700"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
              />
            </svg>
          </div>
        </div>
      </div>
    </header>
  );
}
