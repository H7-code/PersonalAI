"use client";

import React, { useEffect, useState } from "react";

interface LogEntry {
  timestamp: string;
  epoch: number;
  level: string;
  logger: string;
  message: string;
  request_id?: number;
  latency_ms?: number;
  extra?: Record<string, unknown>;
}

export default function LogsView() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filterLevel, setFilterLevel] = useState<string>("ALL");
  const [loading, setLoading] = useState<boolean>(true);

  const fetchLogs = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/logs?limit=100");
      if (res.ok) {
        const data = await res.json();
        setLogs(data.logs || []);
      }
    } catch (e) {
      console.error("Failed to fetch logs:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 3000);
    return () => clearInterval(interval);
  }, []);

  const filtered = logs.filter((l) => {
    if (filterLevel === "ALL") return true;
    return l.level === filterLevel;
  });

  return (
    <div className="w-full max-w-5xl mx-auto py-6 px-4">
      <div className="bg-white/95 backdrop-blur-md rounded-2xl border border-slate-200/90 shadow-xs p-6">
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <div className="flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <h2 className="text-base font-semibold text-slate-900 font-sans">
              Session Event Logs
            </h2>
            <span className="text-xs font-mono text-slate-400">
              (JSONL • Max 10 MB, 5 Backups)
            </span>
          </div>

          <div className="flex items-center gap-2">
            {["ALL", "INFO", "WARNING", "ERROR"].map((lvl) => (
              <button
                key={lvl}
                onClick={() => setFilterLevel(lvl)}
                className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition-colors cursor-pointer ${
                  filterLevel === lvl
                    ? "bg-emerald-600 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                {lvl}
              </button>
            ))}
            <button
              onClick={fetchLogs}
              className="px-2.5 py-1 rounded text-xs font-mono font-medium bg-slate-100 text-slate-600 hover:bg-slate-200 cursor-pointer ml-2"
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Log records list */}
        <div className="mt-4 space-y-2 max-h-[520px] overflow-y-auto font-mono text-xs">
          {loading && logs.length === 0 ? (
            <div className="text-center py-12 text-slate-400">
              Loading session logs...
            </div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-12 text-slate-400">
              No matching log records found.
            </div>
          ) : (
            filtered.map((log, idx) => (
              <div
                key={idx}
                className="p-2.5 rounded-lg bg-slate-50/80 border border-slate-100 hover:bg-slate-100/70 transition-colors flex items-start gap-3"
              >
                <span className="text-slate-400 shrink-0 select-none">
                  {log.timestamp ? log.timestamp.split("T")[1]?.slice(0, 8) : "--:--:--"}
                </span>

                <span
                  className={`px-1.5 py-0.2 rounded text-[10px] font-semibold shrink-0 ${
                    log.level === "ERROR"
                      ? "bg-rose-100 text-rose-800"
                      : log.level === "WARNING"
                      ? "bg-amber-100 text-amber-800"
                      : "bg-emerald-100 text-emerald-800"
                  }`}
                >
                  {log.level}
                </span>

                <span className="text-slate-500 shrink-0 font-semibold truncate max-w-[140px]">
                  [{log.logger}]
                </span>

                <span className="text-slate-800 flex-1 break-all">
                  {log.message}
                  {log.request_id !== undefined && (
                    <span className="ml-2 text-slate-400">
                      (req #{log.request_id})
                    </span>
                  )}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
