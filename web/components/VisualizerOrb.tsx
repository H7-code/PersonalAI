"use client";

import React from "react";
import { AriaState } from "./Header";

interface VisualizerOrbProps {
  state: AriaState;
}

export default function VisualizerOrb({ state }: VisualizerOrbProps) {
  const isSpeaking = state === "SPEAKING";
  const isListening = state === "LISTENING";
  const isThinkingOrGenerating = state === "THINKING" || state === "GENERATING";
  const isInterrupted = state === "INTERRUPTED";

  // Dynamic animation style applied to bars
  const getBarClass = (index: number) => {
    if (isSpeaking) {
      return "animate-wave-bar origin-center";
    }
    if (isListening) {
      return index % 2 === 0 ? "animate-pulse" : "";
    }
    if (isThinkingOrGenerating) {
      return "animate-pulse-subtle";
    }
    return "";
  };

  const getDelay = (index: number) => {
    if (isSpeaking) {
      const delays = [
        "0.1s", "0.25s", "0.4s", "0.15s", "0.3s", "0.5s", "0.2s", "0.35s", "0.45s", "0.1s", "0.2s", "0.3s"
      ];
      return { animationDelay: delays[index % delays.length] };
    }
    return {};
  };

  return (
    <div
      className="flex flex-col items-center justify-center w-full max-w-2xl mx-auto text-center"
      data-purpose="sound-visualizer-container"
    >
      {/* Central Waveform Visualizer Orb */}
      <div className="relative flex items-center justify-center mb-6">
        {/* Outermost soft ambient decorative glow ring */}
        <div
          className={`w-80 h-80 md:w-96 md:h-96 rounded-full bg-gradient-to-b from-white/90 to-emerald-50/40 p-4 border border-emerald-100/80 orb-halo-shadow flex items-center justify-center transition-all duration-500 ${
            isListening ? "ring-2 ring-emerald-300 scale-102" : ""
          } ${isInterrupted ? "ring-2 ring-rose-300" : ""}`}
        >
          {/* Mid-layer architectural ring */}
          <div className="w-full h-full rounded-full bg-white border border-slate-100/90 shadow-inner flex items-center justify-center relative">
            {/* Concentric delicate circle outline */}
            <div className="absolute inset-4 rounded-full border border-slate-100 pointer-events-none" />

            {/* Waveform Bar Array matching emerald tones */}
            <div
              aria-label="Dynamic audio frequency visualizer"
              className="flex items-center justify-center gap-1.5 md:gap-2 h-28 px-6"
            >
              {/* Bar 0 */}
              <div
                style={getDelay(0)}
                className={`w-1.5 h-3 bg-emerald-300 rounded-full opacity-60 transition-transform ${getBarClass(0)}`}
              />
              {/* Bar 1 */}
              <div
                style={getDelay(1)}
                className={`w-1.5 h-6 bg-emerald-400 rounded-full transition-transform ${getBarClass(1)}`}
              />
              {/* Bar 2 */}
              <div
                style={getDelay(2)}
                className={`w-1.5 h-12 bg-emerald-500 rounded-full transition-transform ${getBarClass(2)}`}
              />
              {/* Bar 3 (Center Amplitude) */}
              <div
                style={getDelay(3)}
                className={`w-1.5 md:w-2 h-16 bg-emerald-600 rounded-full shadow-xs transition-transform ${getBarClass(3)}`}
              />
              {/* Bar 4 (Center Hero) */}
              <div
                style={getDelay(4)}
                className={`w-2 md:w-2.5 h-24 bg-gradient-to-t from-emerald-600 to-teal-400 rounded-full shadow-[0_0_12px_rgba(16,185,129,0.3)] transition-transform ${getBarClass(4)}`}
              />
              {/* Bar 5 (Max Peak Hero) */}
              <div
                style={getDelay(5)}
                className={`w-2 md:w-2.5 h-28 bg-gradient-to-t from-emerald-700 via-emerald-500 to-emerald-300 rounded-full shadow-[0_0_15px_rgba(16,185,129,0.45)] transition-transform ${getBarClass(5)}`}
              />
              {/* Bar 6 (Center Right) */}
              <div
                style={getDelay(6)}
                className={`w-1.5 md:w-2 h-20 bg-gradient-to-t from-emerald-600 to-teal-400 rounded-full shadow-xs transition-transform ${getBarClass(6)}`}
              />
              {/* Bar 7 */}
              <div
                style={getDelay(7)}
                className={`w-1.5 h-14 bg-emerald-500 rounded-full transition-transform ${getBarClass(7)}`}
              />
              {/* Bar 8 */}
              <div
                style={getDelay(8)}
                className={`w-1.5 h-8 bg-emerald-400 rounded-full transition-transform ${getBarClass(8)}`}
              />
              {/* Bar 9 */}
              <div
                style={getDelay(9)}
                className={`w-1.5 h-5 bg-emerald-300 rounded-full opacity-80 transition-transform ${getBarClass(9)}`}
              />
              {/* Bar 10 */}
              <div
                style={getDelay(10)}
                className={`w-1.5 h-2 bg-emerald-300 rounded-full opacity-50 transition-transform ${getBarClass(10)}`}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
