"use client";

import React from "react";

export type NavTab = "STAGE" | "LOGS" | "NODES";

interface FooterProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  dspSensitivity?: string;
  engineName?: string;
  isOnline?: boolean;
}

export default function Footer({
  activeTab,
  onSelectTab,
  dspSensitivity = "-18.4 dBFS",
  engineName = "DeepSeek-R1 + Piper/MMS",
  isOnline = true,
}: FooterProps) {
  return (
    <footer className="w-full border-t border-slate-200/80 bg-white/80 backdrop-blur-md py-3 px-6 z-20 sticky bottom-0">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Primary Navigation Tabs */}
        <nav aria-label="Stage Section Navigation" className="flex items-center gap-8">
          {/* Stage Tab */}
          <button
            type="button"
            onClick={() => onSelectTab("STAGE")}
            className={`inline-flex items-center gap-2 text-xs font-semibold font-mono tracking-wider transition-colors pb-1 -mb-1 cursor-pointer ${
              activeTab === "STAGE"
                ? "text-emerald-600 border-b-2 border-emerald-500"
                : "text-slate-400 hover:text-slate-700"
            }`}
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeWidth="2.2"
              viewBox="0 0 24 24"
            >
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
              <line x1="12" x2="12" y1="19" y2="22" />
            </svg>
            STAGE
          </button>

          {/* Logs Tab */}
          <button
            type="button"
            onClick={() => onSelectTab("LOGS")}
            className={`inline-flex items-center gap-2 text-xs font-semibold font-mono tracking-wider transition-colors pb-1 -mb-1 cursor-pointer ${
              activeTab === "LOGS"
                ? "text-emerald-600 border-b-2 border-emerald-500"
                : "text-slate-400 hover:text-slate-700"
            }`}
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <line x1="4" x2="20" y1="6" y2="6" />
              <line x1="4" x2="20" y1="12" y2="12" />
              <line x1="4" x2="20" y1="18" y2="18" />
            </svg>
            LOGS
          </button>

          {/* Nodes Tab */}
          <button
            type="button"
            onClick={() => onSelectTab("NODES")}
            className={`inline-flex items-center gap-2 text-xs font-semibold font-mono tracking-wider transition-colors pb-1 -mb-1 cursor-pointer ${
              activeTab === "NODES"
                ? "text-emerald-600 border-b-2 border-emerald-500"
                : "text-slate-400 hover:text-slate-700"
            }`}
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <circle cx="18" cy="5" r="3" />
              <circle cx="6" cy="12" r="3" />
              <circle cx="18" cy="19" r="3" />
              <line x1="8.59" x2="15.42" y1="13.51" y2="17.49" />
              <line x1="15.41" x2="8.59" y1="6.51" y2="10.49" />
            </svg>
            NODES
          </button>
        </nav>

        {/* Technical Telemetry Readout */}
        <div className="text-[11px] font-mono text-slate-400 flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5 text-emerald-600 font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            ACOUSTIC CORE {isOnline ? "ONLINE" : "CONNECTING"}
          </span>
          <span className="hidden lg:inline text-slate-300">|</span>
          <span className="hidden lg:inline">DSP SENSITIVITY: {dspSensitivity}</span>
          <span className="hidden lg:inline text-slate-300">|</span>
          <span className="hidden sm:inline">ENGINE: {engineName}</span>
        </div>
      </div>
    </footer>
  );
}
