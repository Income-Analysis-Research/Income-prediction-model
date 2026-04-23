import { useEffect, useState, useMemo } from "react";
import {
  ComposableMap, Geographies, Geography, ZoomableGroup,
} from "react-simple-maps";
import { scaleLinear } from "d3-scale";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
  ResponsiveContainer, CartesianGrid,
} from "recharts";
import { MagnifyingGlassIcon } from "@heroicons/react/24/outline";
import { PageLoader, ErrorState } from "../components/LoadingSpinner";
import { api } from "../lib/api";
import type { StateEntry } from "../types";

const GEO_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json";

// FIPS → State abbreviation mapping
const FIPS_TO_STATE: Record<string, string> = {
  "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE",
  "11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA",
  "20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN",
  "28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM",
  "36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
  "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA",
  "54":"WV","55":"WI","56":"WY",
};

export default function Geographic() {
  const [states, setStates]   = useState<StateEntry[] | null>(null);
  const [error,  setError]    = useState<string | null>(null);
  const [view,   setView]     = useState<"map" | "chart">("map");
  const [search, setSearch]   = useState("");
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<StateEntry | null>(null);
  const [sort,   setSort]     = useState<"mean" | "state">("mean");
  const [order,  setOrder]    = useState<"asc" | "desc">("desc");

  useEffect(() => {
    api.states(sort, order)
      .then((r) => setStates(r.states))
      .catch((e: Error) => setError(e.message));
  }, [sort, order]);

  const stateMap = useMemo(() => {
    if (!states) return {};
    return Object.fromEntries(states.map((s) => [s.state, s]));
  }, [states]);

  const colorScale = useMemo(() => {
    if (!states) return null;
    const vals = states.map((s) => s.mean);
    return scaleLinear<string>()
      .domain([Math.min(...vals), 0, Math.max(...vals)])
      .range(["#ef4444", "#374151", "#10b981"]);
  }, [states]);

  const filtered = useMemo(() => {
    if (!states) return [];
    return states.filter((s) =>
      s.state.toLowerCase().includes(search.toLowerCase())
    );
  }, [states, search]);

  const hoveredState = hovered ? stateMap[hovered] : null;

  if (error) return <ErrorState message={error} />;
  if (!states) return <PageLoader message="Loading state effects…" />;

  const top5    = [...states].sort((a, b) => b.mean - a.mean).slice(0, 5);
  const bottom5 = [...states].sort((a, b) => a.mean - b.mean).slice(0, 5);

  return (
    <div className="space-y-6 pb-20 lg:pb-0">

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Geographic Analysis</h1>
          <p className="text-sm text-slate-500 mt-1">
            State-level random intercepts after controlling for income composition · 94% HDI
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setView("map")}
            className={view === "map" ? "btn-primary text-xs py-1.5 px-3" : "btn-secondary text-xs py-1.5 px-3"}
          >
            US Map
          </button>
          <button
            onClick={() => setView("chart")}
            className={view === "chart" ? "btn-primary text-xs py-1.5 px-3" : "btn-secondary text-xs py-1.5 px-3"}
          >
            Bar Chart
          </button>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="card p-4">
          <p className="label mb-1">Highest Effect</p>
          <p className="text-2xl font-bold text-emerald-400 font-mono">{top5[0]?.state}</p>
          <p className="text-xs text-slate-500 font-mono mt-1">+{top5[0]?.mean.toFixed(4)}</p>
        </div>
        <div className="card p-4">
          <p className="label mb-1">Lowest Effect</p>
          <p className="text-2xl font-bold text-red-400 font-mono">{bottom5[0]?.state}</p>
          <p className="text-xs text-slate-500 font-mono mt-1">{bottom5[0]?.mean.toFixed(4)}</p>
        </div>
        <div className="card p-4">
          <p className="label mb-1">σ_state</p>
          <p className="text-2xl font-bold text-amber-400 font-mono">0.160</p>
          <p className="text-xs text-slate-500 mt-1">Structural heterogeneity</p>
        </div>
        <div className="card p-4">
          <p className="label mb-1">States Modeled</p>
          <p className="text-2xl font-bold text-blue-400 font-mono">{states.length}</p>
          <p className="text-xs text-slate-500 mt-1">50 states + DC</p>
        </div>
      </div>

      {/* Map / Chart view */}
      {view === "map" ? (
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <p className="section-title">State Effect Choropleth</p>
            <div className="flex items-center gap-2 text-xs">
              <span className="flex items-center gap-1"><span className="w-4 h-2 rounded bg-red-500" /> Negative</span>
              <span className="flex items-center gap-1"><span className="w-4 h-2 rounded bg-gray-600" /> Neutral</span>
              <span className="flex items-center gap-1"><span className="w-4 h-2 rounded bg-emerald-500" /> Positive</span>
            </div>
          </div>

          <div className="relative">
            {hoveredState && (
              <div className="absolute top-2 right-2 z-10 card p-3 text-sm min-w-[180px]">
                <p className="font-bold text-slate-100">{hoveredState.state}</p>
                <p className="font-mono text-xs mt-1">
                  Effect:{" "}
                  <span className={hoveredState.mean >= 0 ? "text-emerald-400" : "text-red-400"}>
                    {hoveredState.mean >= 0 ? "+" : ""}{hoveredState.mean.toFixed(4)}
                  </span>
                </p>
                <p className="text-xs text-slate-500">
                  94% HDI [{hoveredState.hdi_3.toFixed(3)}, {hoveredState.hdi_97.toFixed(3)}]
                </p>
              </div>
            )}

            <ComposableMap projection="geoAlbersUsa" style={{ width: "100%", height: "auto" }}>
              <ZoomableGroup>
                <Geographies geography={GEO_URL}>
                  {({ geographies }) =>
                    geographies.map((geo) => {
                      const fips = geo.id.toString().padStart(2, "0");
                      const abbr  = FIPS_TO_STATE[fips];
                      const entry = abbr ? stateMap[abbr] : null;
                      const fill  = entry && colorScale
                        ? colorScale(entry.mean)
                        : "#1a2744";

                      return (
                        <Geography
                          key={geo.rsmKey}
                          geography={geo}
                          fill={fill}
                          stroke="#0d1b35"
                          strokeWidth={0.8}
                          style={{
                            default: { outline: "none" },
                            hover:   { outline: "none", opacity: 0.85, cursor: "pointer" },
                            pressed: { outline: "none" },
                          }}
                          onMouseEnter={() => abbr && setHovered(abbr)}
                          onMouseLeave={() => setHovered(null)}
                          onClick={() => abbr && entry && setSelected(entry)}
                        />
                      );
                    })
                  }
                </Geographies>
              </ZoomableGroup>
            </ComposableMap>
          </div>

          {selected && (
            <div className="mt-4 pt-4 border-t border-navy-600/40 flex items-start justify-between gap-4">
              <div>
                <p className="font-bold text-slate-100 text-lg">{selected.state}</p>
                <p className="text-xs text-slate-500 mt-0.5">
                  After controlling for income composition, this state has a structural income{" "}
                  {selected.mean >= 0 ? "advantage" : "disadvantage"} of{" "}
                  <span className={selected.mean >= 0 ? "text-emerald-400 font-mono" : "text-red-400 font-mono"}>
                    {selected.mean >= 0 ? "+" : ""}{selected.mean.toFixed(4)}
                  </span>{" "}
                  in log-income.
                </p>
              </div>
              <div className="flex gap-4 text-center flex-shrink-0">
                <div>
                  <p className="label text-[10px]">Mean</p>
                  <p className={`font-mono font-bold ${selected.mean >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {selected.mean.toFixed(4)}
                  </p>
                </div>
                <div>
                  <p className="label text-[10px]">Std</p>
                  <p className="font-mono text-slate-300">{selected.sd.toFixed(4)}</p>
                </div>
                <div>
                  <p className="label text-[10px]">HDI 3%</p>
                  <p className="font-mono text-slate-400">{selected.hdi_3.toFixed(4)}</p>
                </div>
                <div>
                  <p className="label text-[10px]">HDI 97%</p>
                  <p className="font-mono text-slate-400">{selected.hdi_97.toFixed(4)}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="card p-5">
          <p className="section-title mb-4">State Effects (All {states.length})</p>
          <ResponsiveContainer width="100%" height={states.length * 20 + 40}>
            <BarChart data={filtered} layout="vertical" margin={{ top: 0, right: 60, left: 36, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(30,45,74,0.8)" />
              <XAxis type="number" tick={{ fill: "#64748b", fontSize: 10 }} />
              <YAxis type="category" dataKey="state" tick={{ fill: "#94a3b8", fontSize: 10 }} width={32} />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const s = payload[0].payload as StateEntry;
                  return (
                    <div className="tooltip-base">
                      <p className="font-bold">{s.state}</p>
                      <p className="font-mono">{s.mean >= 0 ? "+" : ""}{s.mean.toFixed(4)}</p>
                      <p className="text-xs text-slate-400">HDI [{s.hdi_3.toFixed(3)}, {s.hdi_97.toFixed(3)}]</p>
                    </div>
                  );
                }}
              />
              <Bar dataKey="mean" radius={[0, 3, 3, 0]}>
                {filtered.map((s, i) => (
                  <Cell key={i} fill={s.mean >= 0 ? "#10b981" : "#ef4444"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Full table */}
      <div className="card p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <p className="section-title">State Effects Table</p>
          <div className="flex gap-2 flex-wrap">
            <div className="relative">
              <MagnifyingGlassIcon className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                type="text"
                placeholder="Filter state…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="input-field pl-9 py-1.5 text-sm w-36"
              />
            </div>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as "mean" | "state")}
              className="input-field py-1.5 text-sm w-auto"
            >
              <option value="mean">Sort: Effect</option>
              <option value="state">Sort: State</option>
            </select>
            <select
              value={order}
              onChange={(e) => setOrder(e.target.value as "asc" | "desc")}
              className="input-field py-1.5 text-sm w-auto"
            >
              <option value="desc">Desc</option>
              <option value="asc">Asc</option>
            </select>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-navy-600/40">
                <th className="text-left py-2 px-3 label">Rank</th>
                <th className="text-left py-2 px-3 label">State</th>
                <th className="text-right py-2 px-3 label">Effect (Mean)</th>
                <th className="text-right py-2 px-3 label">Std</th>
                <th className="text-right py-2 px-3 label">HDI 3%</th>
                <th className="text-right py-2 px-3 label">HDI 97%</th>
                <th className="text-right py-2 px-3 label hidden sm:table-cell">Direction</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s, i) => (
                <tr
                  key={s.state}
                  onClick={() => setSelected(selected?.state === s.state ? null : s)}
                  className={[
                    "border-b border-navy-700/30 table-row-hover cursor-pointer",
                    selected?.state === s.state ? "bg-navy-700/40" : "",
                  ].join(" ")}
                >
                  <td className="py-2.5 px-3 text-slate-500 font-mono text-xs">{i + 1}</td>
                  <td className="py-2.5 px-3 font-semibold text-slate-200">{s.state}</td>
                  <td
                    className="py-2.5 px-3 text-right font-mono font-semibold"
                    style={{ color: s.mean >= 0 ? "#10b981" : "#ef4444" }}
                  >
                    {s.mean >= 0 ? "+" : ""}{s.mean.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-400">{s.sd.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-400">{s.hdi_3.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-400">{s.hdi_97.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right hidden sm:table-cell">
                    <span className={s.mean >= 0 ? "badge-green badge" : "badge-red badge"}>
                      {s.mean >= 0 ? "Premium" : "Discount"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
