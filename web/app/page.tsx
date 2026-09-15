"use client";

import React, { useState, useEffect } from "react";
import Header from "../components/Header";
import VisualizerOrb from "../components/VisualizerOrb";
import MicButton from "../components/MicButton";
import LiveTranscript from "../components/LiveTranscript";
import Footer, { NavTab } from "../components/Footer";
import LogsView from "../components/LogsView";
import NodesView from "../components/NodesView";
import { useAriaVoice } from "../components/useAriaVoice";

export default function Home() {
  const [activeTab, setActiveTab] = useState<NavTab>("STAGE");
  const [selectedMic, setSelectedMic] = useState<string>("Spatial Array (Default)");
  const [devices, setDevices] = useState<Array<{ index: number; name: string }>>([]);
  const [showSettings, setShowSettings] = useState<boolean>(false);

  const {
    state,
    userTranscript,
    assistantText,
    speakingSentence,
    langMode,
    latencyMs,
    isPttActive,
    isConnected,
    startPtt,
    stopPtt,
    interrupt,
    simulateTurn,
  } = useAriaVoice();

  // Fetch audio input devices from backend
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/devices")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && data.devices && data.devices.length > 0) {
          setDevices(data.devices);
          setSelectedMic(data.devices[0].name);
        }
      })
      .catch(() => {
        // Fallback default
        setDevices([
          { index: 0, name: "Spatial Array (Default)" },
          { index: 1, name: "Realtek Audio Microphone" },
        ]);
      });
  }, []);

  return (
    <div className="min-h-screen flex flex-col justify-between overflow-x-hidden">
      {/* Header */}
      <Header
        state={state}
        selectedMic={selectedMic}
        devices={devices}
        onSelectMic={setSelectedMic}
        onToggleSettings={() => setShowSettings(!showSettings)}
      />

      {/* Main View Switching based on Tab */}
      {activeTab === "STAGE" && (
        <main className="flex-1 flex flex-col items-center justify-center px-4 py-8 relative">
          <VisualizerOrb state={state} />

          <MicButton
            state={state}
            isPttActive={isPttActive}
            onPttStart={startPtt}
            onPttStop={stopPtt}
            onInterrupt={interrupt}
          />

          <LiveTranscript
            state={state}
            userTranscript={userTranscript}
            assistantText={assistantText}
            speakingSentence={speakingSentence}
            langMode={langMode}
            latencyMs={latencyMs}
          />

          {/* Quick Simulation controls for testing voice turns */}
          <div className="mt-6 flex items-center gap-2">
            <button
              onClick={() =>
                simulateTurn(
                  "Analyze quarterly cloud cost anomalies and give recommendations.",
                  "english"
                )
              }
              className="text-[11px] font-mono px-2.5 py-1 rounded bg-white/70 border border-slate-200 text-slate-500 hover:text-emerald-700 hover:border-emerald-300 transition-colors shadow-2xs cursor-pointer"
            >
              Simulate English Turn
            </button>
            <button
              onClick={() =>
                simulateTurn(
                  "Koshish karein ke mujhe server status ka pata chal jaye.",
                  "urdu"
                )
              }
              className="text-[11px] font-mono px-2.5 py-1 rounded bg-white/70 border border-slate-200 text-slate-500 hover:text-emerald-700 hover:border-emerald-300 transition-colors shadow-2xs cursor-pointer"
            >
              Simulate Urdu Turn
            </button>
          </div>
        </main>
      )}

      {activeTab === "LOGS" && <LogsView />}

      {activeTab === "NODES" && <NodesView />}

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-slate-900/30 backdrop-blur-xs flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-xl max-w-md w-full p-6">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <h3 className="text-base font-semibold text-slate-900">
                ARIA Acoustic Settings
              </h3>
              <button
                onClick={() => setShowSettings(false)}
                className="text-slate-400 hover:text-slate-600 text-lg font-mono cursor-pointer"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4 pt-4 text-xs font-mono text-slate-600">
              <div className="flex justify-between items-center">
                <span>Silero VAD Threshold:</span>
                <span className="font-semibold text-slate-900">0.50</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Echo Holdoff Timer:</span>
                <span className="font-semibold text-slate-900">200 ms</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Whisper STT Quantization:</span>
                <span className="font-semibold text-emerald-700">INT8 (CPU)</span>
              </div>
              <div className="flex justify-between items-center">
                <span>DeepSeek-R1 Generation Budget:</span>
                <span className="font-semibold text-slate-900">512 tokens</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Operating Mode:</span>
                <span className="font-semibold text-emerald-700">100% Offline Local</span>
              </div>
            </div>

            <button
              onClick={() => setShowSettings(false)}
              className="mt-6 w-full py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl font-medium text-xs transition-colors cursor-pointer"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Bottom Navigation & Telemetry Readout */}
      <Footer
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        isOnline={isConnected}
        dspSensitivity="-18.4 dBFS"
        engineName="DeepSeek-R1 + Piper/MMS"
      />
    </div>
  );
}
