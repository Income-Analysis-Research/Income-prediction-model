import { useState, useEffect, useRef, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
  ResponsiveContainer, CartesianGrid,
} from "recharts";
import {
  MagnifyingGlassIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
} from "@heroicons/react/24/outline";
import { InlineSpinner } from "../components/LoadingSpinner";
import { api, fmt$$, fmtLabel, betaColor } from "../lib/api";
import type { PredictionResult, ZipSearchResult } from "../types";
import clsx from "clsx";

const YEARS = Array.from({ length: 12 }, (_, i) => 2011 + i);

function useDebounce<T>(value: T, ms: number): T {
  const [dv, setDv] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDv(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return dv;
}

function DecompositionChart({ result }: { result: PredictionResult }) {
  const { alpha, beta_dot_x, state_effect, year_effect } = result.decomposition;
  const total = alpha + beta_dot_x + state_effect + year_effect;

  const items = [
    { name: "Intercept (α)",    value: alpha,        color: "#94a3b8" },
    { name: "Feature Effects",  value: beta_dot_x,   color: "#3b82f6" },
    { name: "State Effect",     value: state_effect, color: state_effect >= 0 ? "#10b981" : "#ef4444" },
    { name: "Year Effect",      value: year_effect,  color: year_effect  >= 0 ? "#6ee7b7" : "#fca5a5" },
  ];

  return (
    <div className="space-y-3">
      <div className="flex justify-between text-xs text-slate-500 font-mono">
        <span>Contribution to log(income)</span>
        <span>Σ = {total.toFixed(4)}</span>
      </div>
      {items.map((item) => {
        const barPct = Math.abs(item.value) / Math.abs(alpha) * 100;
        return (
          <div key={item.name} className="flex items-center gap-3">
            <span className="text-xs text-slate-400 w-32 flex-shrink-0">{item.name}</span>
            <div className="flex-1 bg-navy-900 rounded h-5 overflow-hidden relative">
              <div
                className="h-full rounded transition-all duration-700"
                style={{
                  width: `${Math.min(barPct, 100)}%`,
                  backgroundColor: item.color,
                  opacity: 0.75,
                }}
              />
            </div>
            <span className="text-xs font-mono w-16 text-right flex-shrink-0" style={{ color: item.color }}>
              {item.value >= 0 ? "+" : ""}{item.value.toFixed(4)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function FeatureChart({ result }: { result: PredictionResult }) {
  const data = Object.entries(result.feature_vector).map(([key, val]) => ({
    name:  fmtLabel(key),
    value: val,
  }));

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, data.length * 28)}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 50, left: 110, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(30,45,74,0.8)" />
        <XAxis type="number" domain={[0, 1]} tick={{ fill: "#64748b", fontSize: 10 }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
        <YAxis type="category" dataKey="name" tick={{ fill: "#94a3b8", fontSize: 10 }} width={105} />
        <Tooltip
          formatter={(v: number) => [`${(v * 100).toFixed(2)}%`, "Share of AGI"]}
          contentStyle={{ background: "#111827", border: "1px solid rgba(30,45,74,0.8)", borderRadius: "8px", fontSize: "12px" }}
        />
        <Bar dataKey="value" radius={[0, 3, 3, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill="#3b82f6" opacity={0.7} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function CIBar({ lo, mean, hi }: { lo: number; mean: number; hi: number }) {
  const width = hi - lo;
  const meanPct  = ((mean - lo) / width) * 100;
  return (
    <div className="relative h-2 bg-navy-900 rounded-full overflow-visible mt-2 mb-1">
      <div
        className="absolute h-full bg-blue-500/30 rounded-full"
        style={{ left: 0, right: 0 }}
      />
      <div
        className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-blue-400 border-2 border-blue-600 shadow-glow"
        style={{ left: `calc(${meanPct}% - 6px)` }}
      />
    </div>
  );
}

export default function Predictor() {
  const [zip,     setZip]     = useState("");
  const [year,    setYear]    = useState(2019);
  const [result,  setResult]  = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const [suggestions, setSuggestions]     = useState<ZipSearchResult[]>([]);
  const [showSuggest, setShowSuggest]     = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const debouncedZip = useDebounce(zip, 300);

  // ZIP search autocomplete
  useEffect(() => {
    if (debouncedZip.length < 2) { setSuggestions([]); return; }
    setSearchLoading(true);
    api.zipSearch(debouncedZip, year)
      .then((r) => { setSuggestions(r.results); setShowSuggest(true); })
      .catch(() => setSuggestions([]))
      .finally(() => setSearchLoading(false));
  }, [debouncedZip, year]);

  const predict = useCallback(async (z?: string, y?: number) => {
    const qzip  = z ?? zip;
    const qyear = y ?? year;
    if (!qzip || qzip.length < 3) { setError("Enter a valid ZIP code"); return; }

    setLoading(true);
    setError(null);
    setShowSuggest(false);

    try {
      const r = await api.predict(qzip.padStart(5, "0"), qyear);
      setResult(r);
      setZip(r.query.zip);
    } catch (e: unknown) {
      const err = e as Error & { hint?: string; fallback_year?: number };
      setError(err.message + (err.hint ? `\n${err.hint}` : ""));
    } finally {
      setLoading(false);
    }
  }, [zip, year]);

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") predict();
    if (e.key === "Escape") setShowSuggest(false);
  };

  const pickSuggestion = (s: ZipSearchResult) => {
    setZip(s.zip);
    setSuggestions([]);
    setShowSuggest(false);
    predict(s.zip, year);
  };

  return (
    <div className="space-y-6 pb-20 lg:pb-0">

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Income Predictor</h1>
        <p className="text-sm text-slate-500 mt-1">
          Bayesian posterior mean prediction with 94% credible interval
        </p>
      </div>

      {/* Input panel */}
      <div className="card p-6">
        <p className="section-title mb-5">Query Parameters</p>
        <div className="flex flex-col sm:flex-row gap-4 items-end">

          {/* ZIP input with autocomplete */}
          <div className="flex-1 relative">
            <label className="label mb-2 block">ZIP Code</label>
            <div className="relative">
              <MagnifyingGlassIcon className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                ref={inputRef}
                type="text"
                inputMode="numeric"
                placeholder="e.g. 10001"
                value={zip}
                maxLength={5}
                onChange={(e) => { setZip(e.target.value.replace(/\D/g, "")); setError(null); }}
                onKeyDown={handleKey}
                onFocus={() => suggestions.length > 0 && setShowSuggest(true)}
                className="input-field pl-9"
              />
              {searchLoading && (
                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                  <InlineSpinner />
                </div>
              )}
            </div>

            {/* Suggestions dropdown */}
            {showSuggest && suggestions.length > 0 && (
              <div className="absolute top-full left-0 right-0 z-50 mt-1 card-sm overflow-hidden shadow-xl">
                {suggestions.slice(0, 10).map((s) => (
                  <button
                    key={s.zip}
                    onClick={() => pickSuggestion(s)}
                    className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-navy-700/60 transition-colors text-left"
                  >
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-sm text-slate-200">{s.zip}</span>
                      <span className="badge badge-gray">{s.state}</span>
                    </div>
                    <span className="text-xs text-slate-500 font-mono">{fmt$$(s.avg_income)}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Year selector */}
          <div className="sm:w-44">
            <label className="label mb-2 block">Tax Year</label>
            <select
              value={year}
              onChange={(e) => setYear(parseInt(e.target.value))}
              className="input-field"
            >
              {YEARS.map((y) => (
                <option key={y} value={y}>{y}{y === 2020 ? " (COVID)" : ""}</option>
              ))}
            </select>
          </div>

          {/* Submit */}
          <button
            onClick={() => predict()}
            disabled={loading || !zip}
            className="btn-primary sm:w-36 h-[42px]"
          >
            {loading ? <InlineSpinner /> : <MagnifyingGlassIcon className="w-4 h-4" />}
            {loading ? "Predicting…" : "Predict"}
          </button>
        </div>

        {/* Quick examples */}
        <div className="mt-4 flex flex-wrap gap-2">
          <span className="text-xs text-slate-500">Examples:</span>
          {[["10002", 2019, "Manhattan"], ["60010", 2019, "Barrington IL"], ["75039", 2019, "Irving TX"], ["01011", 2019, "Chester MA"]].map(([z, y, name]) => (
            <button
              key={z}
              onClick={() => { setZip(String(z)); setYear(Number(y)); predict(String(z), Number(y)); }}
              className="text-xs px-2.5 py-1 rounded-md bg-navy-700 hover:bg-navy-600 text-slate-300 border border-navy-600/50 transition-colors font-mono"
            >
              {z} <span className="text-slate-500">({name})</span>
            </button>
          ))}
        </div>

        {/* Error */}
        {error && (
          <div className="mt-4 flex gap-2 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
            <ExclamationCircleIcon className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-300 whitespace-pre-line">{error}</p>
          </div>
        )}
      </div>

      {/* Result */}
      {result && (
        <div className="space-y-4 animate-fade-in">

          {/* Primary result card */}
          <div className="card p-6 glow-blue">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <CheckCircleIcon className="w-5 h-5 text-emerald-400" />
                  <p className="label">Prediction Result</p>
                </div>
                <p className="text-4xl font-bold text-emerald-400 font-mono tabular-nums">
                  {fmt$$(result.predicted.income_mean)}
                </p>
                <p className="text-sm text-slate-500 mt-1">
                  Posterior mean average income for{" "}
                  <span className="font-mono text-slate-300">{result.query.zip}</span>
                  {" "}({result.query.state}) in {result.query.year}
                </p>
              </div>

              <div className="text-right">
                <p className="label mb-1">Actual (IRS)</p>
                <p className="text-2xl font-semibold text-amber-400 font-mono">
                  {fmt$$(result.actual_avg_income)}
                </p>
                <p className="text-xs text-slate-500">
                  Error: {fmt$$(Math.abs(result.predicted.income_mean - result.actual_avg_income))}
                </p>
              </div>
            </div>

            {/* CI bar */}
            <div className="mt-6">
              <div className="flex justify-between text-xs text-slate-500 font-mono">
                <span>{fmt$$(result.predicted.income_lo)}</span>
                <span className="text-slate-400">94% Credible Interval</span>
                <span>{fmt$$(result.predicted.income_hi)}</span>
              </div>
              <CIBar lo={result.predicted.income_lo} mean={result.predicted.income_mean} hi={result.predicted.income_hi} />
            </div>

            {/* Quick stats */}
            <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-navy-900 rounded-lg p-3 text-center">
                <p className="label text-[10px] mb-1">log(income)</p>
                <p className="font-mono text-sm text-slate-200">{result.predicted.log_income_mean.toFixed(4)}</p>
              </div>
              <div className="bg-navy-900 rounded-lg p-3 text-center">
                <p className="label text-[10px] mb-1">State</p>
                <p className="font-mono text-sm text-slate-200">{result.query.state}</p>
              </div>
              <div className="bg-navy-900 rounded-lg p-3 text-center">
                <p className="label text-[10px] mb-1">ZIP</p>
                <p className="font-mono text-sm text-slate-200">{result.query.zip}</p>
              </div>
              <div className="bg-navy-900 rounded-lg p-3 text-center">
                <p className="label text-[10px] mb-1">Year</p>
                <p className="font-mono text-sm text-slate-200">{result.query.year}</p>
              </div>
            </div>
          </div>

          {/* Decomposition + Features */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

            {/* Decomposition */}
            <div className="card p-5">
              <p className="section-title mb-4">Prediction Decomposition</p>
              <DecompositionChart result={result} />
              <div className="mt-4 pt-4 border-t border-navy-600/40 grid grid-cols-2 gap-3">
                {Object.entries(result.decomposition).map(([k, v]) => (
                  <div key={k} className="bg-navy-900 rounded-lg p-3">
                    <p className="label text-[10px] mb-1">
                      {k === "alpha" ? "α (Intercept)" :
                       k === "beta_dot_x" ? "β·X (Features)" :
                       k === "state_effect" ? "u_state" : "γ_year"}
                    </p>
                    <p className={clsx(
                      "font-mono text-sm font-semibold",
                      k === "alpha" ? "text-slate-300" :
                      v >= 0 ? "text-emerald-400" : "text-red-400"
                    )}>
                      {v >= 0 ? "+" : ""}{v.toFixed(4)}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Feature vector */}
            <div className="card p-5">
              <p className="section-title mb-4">Income Composition (Feature Vector)</p>
              <FeatureChart result={result} />
            </div>
          </div>

          {/* New prediction button */}
          <div className="flex justify-end">
            <button
              onClick={() => { setResult(null); setError(null); setZip(""); inputRef.current?.focus(); }}
              className="btn-secondary"
            >
              <ArrowPathIcon className="w-4 h-4" />
              New Prediction
            </button>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div className="card p-10 text-center">
          <div className="w-16 h-16 rounded-full bg-blue-600/10 border border-blue-600/20 flex items-center justify-center mx-auto mb-4">
            <MagnifyingGlassIcon className="w-8 h-8 text-blue-400" />
          </div>
          <p className="text-slate-300 font-medium">Enter a ZIP code to generate a prediction</p>
          <p className="text-sm text-slate-500 mt-2 max-w-sm mx-auto">
            The model uses 4,000 posterior draws to compute the income estimate and credible interval.
          </p>
        </div>
      )}
    </div>
  );
}
