"use client";

import React from "react";
import { AriaState } from "./Header";

interface MicButtonProps {
  state: AriaState;
  isPttActive: boolean;
  onPttStart: () => void;
  onPttStop: () => void;
  onInterrupt: () => void;
}

export default function MicButton({
  state,
  isPttActive,
  onPttStart,
  onPttStop,
  onInterrupt,
}: MicButtonProps) {
  const isListening = state === "LISTENING" || isPttActive;
  const isGeneratingOrSpeaking = state === "GENERATING" || state === "SPEAKING" || state === "THINKING";

  const handleClick = () => {
    if (isGeneratingOrSpeaking) {
      // If currently generating or speaking, clicking interrupts
      onInterrupt();
    } else if (isListening) {
      onPttStop();
    } else {
      onPttStart();
    }
  };

  const getSubtitle = () => {
    if (isListening) return "LISTENING • RELEASE TO SEND";
    if (state === "TRANSCRIBING") return "TRANSCRIBING SPEECH...";
    if (state === "THINKING") return "THINKING...";
    if (state === "GENERATING") return "STREAMING TOKENS (ESC TO STOP)";
    if (state === "SPEAKING") return "SPEAKING (ESC TO INTERRUPT)";
    if (state === "INTERRUPTED") return "TURN CANCELLED";
    return "HANDS-FREE ACTIVE";
  };

  return (
    <div className="flex flex-col items-center">
      {/* Action Prompt Caption */}
      <p className="text-sm md:text-base font-normal text-slate-500 tracking-normal mb-8 select-none">
        Tap to speak or hold{" "}
        <kbd className="px-2 py-0.5 text-xs font-semibold text-slate-600 bg-white border border-slate-300 rounded shadow-xs font-mono">
          Space
        </kbd>
        {isGeneratingOrSpeaking && (
          <span className="ml-2 text-xs text-rose-500 font-medium">
            (or press <kbd className="px-1.5 py-0.5 text-xs font-semibold text-rose-700 bg-rose-50 border border-rose-200 rounded font-mono">Esc</kbd> to interrupt)
          </span>
        )}
      </p>

      {/* Interactive Push-to-Talk Mic Target */}
      <div className="flex flex-col items-center">
        <button
          id="mic-action-btn"
          aria-label={isGeneratingOrSpeaking ? "Interrupt Assistant" : "Toggle voice input"}
          type="button"
          onClick={handleClick}
          onMouseDown={(e) => {
            if (!isGeneratingOrSpeaking) {
              e.preventDefault();
              onPttStart();
            }
          }}
          onMouseUp={(e) => {
            if (!isGeneratingOrSpeaking && isPttActive) {
              e.preventDefault();
              onPttStop();
            }
          }}
          onTouchStart={(e) => {
            if (!isGeneratingOrSpeaking) {
              e.preventDefault();
              onPttStart();
            }
          }}
          onTouchEnd={(e) => {
            if (!isGeneratingOrSpeaking && isPttActive) {
              e.preventDefault();
              onPttStop();
            }
          }}
          className={`group relative flex items-center justify-center w-20 h-20 md:w-22 md:h-22 rounded-full text-white transition-all duration-200 cursor-pointer outline-none focus:ring-4 ${
            isGeneratingOrSpeaking
              ? "bg-gradient-to-tr from-rose-600 via-rose-500 to-amber-500 mic-button-halo-interrupted hover:scale-105 active:scale-95 focus:ring-rose-300"
              : isListening
              ? "bg-gradient-to-tr from-emerald-600 via-emerald-500 to-teal-400 mic-button-halo-listening ring-4 ring-emerald-400 scale-105 focus:ring-emerald-300"
              : "bg-gradient-to-tr from-emerald-600 via-emerald-500 to-teal-400 mic-button-halo hover:scale-105 active:scale-95 focus:ring-emerald-300"
          }`}
        >
          {isGeneratingOrSpeaking ? (
            /* Stop/Interrupt square icon */
            <svg
              className="w-7 h-7 drop-shadow-xs transition-transform duration-200 group-hover:scale-110"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
          ) : (
            /* Inner SVG microphone */
            <svg
              className="w-8 h-8 drop-shadow-xs transition-transform duration-200 group-hover:scale-110"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2.2"
              viewBox="0 0 24 24"
            >
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
              <line x1="12" x2="12" y1="19" y2="22" />
            </svg>
          )}
        </button>

        {/* Subtitle State Label */}
        <span className="mt-4 font-mono text-[11px] md:text-xs font-medium tracking-[0.25em] text-slate-400 uppercase select-none">
          {getSubtitle()}
        </span>
      </div>
    </div>
  );
}
