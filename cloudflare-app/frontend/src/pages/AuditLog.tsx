import { useEffect, useState, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
  ResponsiveContainer,
} from "recharts";
import {
  ArrowPathIcon,
  FunnelIcon,
  ArrowDownTrayIcon,
} from "@heroicons/react/24/outline";
import { PageLoader, ErrorState, InlineSpinner } from "../components/LoadingSpinner";
import { api, fmt$$ } from "../lib/api";
import type { AuditEntry, AuditStatsResponse } from "../types";

function formatTimestamp(ts: string): string {
  try {
    return new Date(ts).toLocaleString("en-US", {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return ts;
  }
}

function downloadCSV(entries: AuditEntry[]) {
  const header = ["Timestamp", "ZIP", "State", "Year", "Event", "Predicted Income", "Actual Income", "State Effect", "Year Effect"];
  const rows   = entries.map((e) => [
    e.timestamp,
    e.query?.zip   ?? "",
    e.query?.state ?? "",
    e.query?.year  ?? "",
    e.event        ?? "",
    e.predicted?.income_mean?.toFixed(2) ?? "",
    e.actual_avg_income?.toFixed(2) ?? "",
    e.decomposition?.state_effect?.toFixed(4) ?? "",
    e.decomposition?.year_effect?.toFixed(4)  ?? "",
  ]);

  const csv = [header, ...rows].map((r) => r.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = `income_model_audit_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function AuditLog() {
  const [entries, setEntries] = useState<AuditEntry[] | null>(null);
  const [stats,   setStats]   = useState<AuditStatsResponse | null>(null);
  const [error,   setError]   = useState<string | null>(null);
  const [total,   setTotal]   = useState(0);
  const [loading, setLoading] = useState(false);

  const [filterState, setFilterState] = useState("");
  const [filterYear,  setFilterYear]  = useState("");
  const [filterEvent, setFilterEvent] = useState("");
  const [limit, setLimit] = useState(50);

  const load = useCallback(() => {
    setLoading(true);
    Promise.all([api.auditLog(200), api.auditStats()])
      .then(([log, s]) => {
        setEntries(log.entries);
        setTotal(log.total);
        setStats(s);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  const filtered = entries?.filter((e) => {
    const matchState = !filterState || (e.query?.state ?? "").toLowerCase().includes(filterState.toLowerCase());
    const matchYear  = !filterYear  || String(e.query?.year).includes(filterYear);
    const matchEvent = !filterEvent || (e.event ?? "").toLowerCase().includes(filterEvent.toLowerCase());
    return matchState && matchYear && matchEvent;
  }) ?? [];

  const displayEntries = filtered.slice(0, limit);

  if (error && !entries) return <ErrorState message={error} onRetry={load} />;

  const stateChartData = stats
    ? Object.entries(stats.byState)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([state, count]) => ({ state, count }))
    : [];

  const yearChartData = stats
    ? Object.entries(stats.byYear)
        .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
        .map(([year, count]) => ({ year: parseInt(year), count }))
    : [];

  return (
    <div className="space-y-6 pb-20 lg:pb-0">

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Prediction Audit Log</h1>
          <p className="text-sm text-slate-500 mt-1">
            Full history of all model predictions · {total} total entries
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={load}
            disabled={loading}
            className="btn-secondary"
          >
            {loading ? <InlineSpinner /> : <ArrowPathIcon className="w-4 h-4" />}
            Refresh
          </button>
          {entries && entries.length > 0 && (
            <button onClick={() => downloadCSV(filtered)} className="btn-secondary">
              <ArrowDownTrayIcon className="w-4 h-4" />
              Export CSV
            </button>
          )}
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="card p-4">
            <p className="label mb-1">Total Predictions</p>
            <p className="text-3xl font-bold text-amber-400 font-mono">{stats.total}</p>
          </div>

          {stateChartData.length > 0 && (
            <div className="card p-4">
              <p className="label mb-2">By State (Top 10)</p>
              <ResponsiveContainer width="100%" height={80}>
                <BarChart data={stateChartData} margin={{ top: 0, right: 0, left: -30, bottom: 0 }}>
                  <XAxis dataKey="state" tick={{ fill: "#64748b", fontSize: 9 }} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 9 }} />
                  <Tooltip
                    contentStyle={{ background: "#111827", border: "1px solid rgba(30,45,74,0.8)", fontSize: "12px", borderRadius: "6px" }}
                  />
                  <Bar dataKey="count" name="Queries" radius={[2, 2, 0, 0]}>
                    {stateChartData.map((_, i) => (
                      <Cell key={i} fill="#3b82f6" opacity={0.7} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {yearChartData.length > 0 && (
            <div className="card p-4">
              <p className="label mb-2">By Year</p>
              <ResponsiveContainer width="100%" height={80}>
                <BarChart data={yearChartData} margin={{ top: 0, right: 0, left: -30, bottom: 0 }}>
                  <XAxis dataKey="year" tick={{ fill: "#64748b", fontSize: 9 }} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 9 }} />
                  <Tooltip
                    contentStyle={{ background: "#111827", border: "1px solid rgba(30,45,74,0.8)", fontSize: "12px", borderRadius: "6px" }}
                  />
                  <Bar dataKey="count" name="Queries" radius={[2, 2, 0, 0]}>
                    {yearChartData.map((_, i) => (
                      <Cell key={i} fill="#10b981" opacity={0.7} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Filters */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <FunnelIcon className="w-4 h-4 text-slate-500" />
          <span className="label">Filters</span>
          {(filterState || filterYear || filterEvent) && (
            <button
              onClick={() => { setFilterState(""); setFilterYear(""); setFilterEvent(""); }}
              className="text-xs text-blue-400 hover:text-blue-300 ml-auto"
            >
              Clear all
            </button>
          )}
        </div>
        <div className="flex flex-wrap gap-3">
          <input
            type="text"
            placeholder="Filter by state…"
            value={filterState}
            onChange={(e) => setFilterState(e.target.value)}
            className="input-field py-1.5 text-sm w-40"
          />
          <input
            type="text"
            placeholder="Filter by year…"
            value={filterYear}
            onChange={(e) => setFilterYear(e.target.value)}
            className="input-field py-1.5 text-sm w-36"
          />
          <input
            type="text"
            placeholder="Filter by event…"
            value={filterEvent}
            onChange={(e) => setFilterEvent(e.target.value)}
            className="input-field py-1.5 text-sm w-44"
          />
          <div className="flex items-center gap-2 ml-auto">
            <span className="text-xs text-slate-500">Show:</span>
            {[25, 50, 100, 200].map((n) => (
              <button
                key={n}
                onClick={() => setLimit(n)}
                className={clsx(
                  "text-xs px-2 py-1 rounded border transition-colors",
                  limit === n
                    ? "bg-blue-600/20 border-blue-600/40 text-blue-300"
                    : "bg-navy-800 border-navy-600/40 text-slate-400 hover:text-slate-200"
                )}
              >
                {n}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-3">
          <p className="section-title">Prediction History</p>
          <span className="text-xs text-slate-500">
            {filtered.length} matching · showing {Math.min(displayEntries.length, limit)}
          </span>
        </div>

        {!entries ? (
          <PageLoader message="Loading audit log…" />
        ) : displayEntries.length === 0 ? (
          <div className="text-center py-12 text-slate-500">
            <p>No predictions match your filters.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-navy-600/40">
                  <th className="text-left py-2 px-3 label">Timestamp</th>
                  <th className="text-left py-2 px-3 label">ZIP</th>
                  <th className="text-left py-2 px-3 label">State</th>
                  <th className="text-right py-2 px-3 label">Year</th>
                  <th className="text-right py-2 px-3 label">Predicted</th>
                  <th className="text-right py-2 px-3 label hidden md:table-cell">Actual</th>
                  <th className="text-right py-2 px-3 label hidden lg:table-cell">State Eff.</th>
                  <th className="text-right py-2 px-3 label hidden lg:table-cell">Year Eff.</th>
                  <th className="text-left py-2 px-3 label hidden xl:table-cell">Event</th>
                </tr>
              </thead>
              <tbody>
                {displayEntries.map((entry, i) => (
                  <tr key={i} className="border-b border-navy-700/30 table-row-hover">
                    <td className="py-2.5 px-3 text-xs text-slate-500 font-mono whitespace-nowrap">
                      {formatTimestamp(entry.timestamp)}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-slate-200">{entry.query?.zip ?? "—"}</td>
                    <td className="py-2.5 px-3">
                      <span className="badge badge-gray">{entry.query?.state ?? "—"}</span>
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-300">{entry.query?.year ?? "—"}</td>
                    <td className="py-2.5 px-3 text-right font-mono font-semibold text-emerald-400">
                      {entry.predicted?.income_mean ? fmt$$(entry.predicted.income_mean) : "—"}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-amber-400 hidden md:table-cell">
                      {entry.actual_avg_income ? fmt$$(entry.actual_avg_income) : "—"}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-xs hidden lg:table-cell"
                        style={{ color: (entry.decomposition?.state_effect ?? 0) >= 0 ? "#10b981" : "#ef4444" }}>
                      {entry.decomposition?.state_effect != null
                        ? `${entry.decomposition.state_effect >= 0 ? "+" : ""}${entry.decomposition.state_effect.toFixed(4)}`
                        : "—"}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-xs hidden lg:table-cell"
                        style={{ color: (entry.decomposition?.year_effect ?? 0) >= 0 ? "#10b981" : "#ef4444" }}>
                      {entry.decomposition?.year_effect != null
                        ? `${entry.decomposition.year_effect >= 0 ? "+" : ""}${entry.decomposition.year_effect.toFixed(4)}`
                        : "—"}
                    </td>
                    <td className="py-2.5 px-3 hidden xl:table-cell">
                      <span className="badge badge-blue text-[10px]">{entry.event ?? "—"}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {filtered.length > limit && (
          <div className="mt-4 text-center">
            <button onClick={() => setLimit(l => l + 50)} className="btn-secondary text-sm">
              Load more ({filtered.length - limit} remaining)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function clsx(...classes: (string | boolean | undefined)[]): string {
  return classes.filter(Boolean).join(" ");
}
