import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell,
} from "recharts";
import {
  ArrowRightIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from "@heroicons/react/24/outline";
import StatCard from "../components/StatCard";
import { PageLoader, ErrorState } from "../components/LoadingSpinner";
import { api, fmt$$, fmtLabel, betaColor } from "../lib/api";
import type { ModelSummary, StateEntry, YearEntry, BetaEntry, AuditEntry } from "../types";

interface DashData {
  summary: ModelSummary;
  states: StateEntry[];
  years: YearEntry[];
  betas: BetaEntry[];
  audit: AuditEntry[];
}

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value: number; color: string; name: string }>;
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip-base">
      <p className="font-semibold mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(4) : p.value}
        </p>
      ))}
    </div>
  );
};

export default function Dashboard() {
  const [data, setData]   = useState<DashData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.summary(),
      api.states("mean", "desc"),
      api.years(),
      api.betas(),
      api.auditLog(10),
    ]).then(([summary, statesR, yearsR, betasR, auditR]) => {
      setData({
        summary,
        states: statesR.states.slice(0, 10),
        years:  yearsR.years,
        betas:  betasR.betas,
        audit:  auditR.entries,
      });
    }).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} onRetry={() => window.location.reload()} />;
  if (!data)  return <PageLoader message="Loading model overview…" />;

  const { summary, states, years, betas, audit } = data;
  const maxRhat = Math.max(...betas.map((b) => b.r_hat ?? 1), summary.alpha?.r_hat ?? 1);
  const converged = maxRhat < 1.05;

  return (
    <div className="space-y-6 pb-20 lg:pb-0">

      {/* ── Header ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">
            Model Dashboard
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Hierarchical Bayesian Panel Regression · IRS SOI 2011–2022
          </p>
        </div>
        <div className="flex items-center gap-2">
          {converged ? (
            <span className="badge badge-green">
              <CheckCircleIcon className="w-3.5 h-3.5 mr-1" />
              Converged
            </span>
          ) : (
            <span className="badge badge-gold">
              <ExclamationTriangleIcon className="w-3.5 h-3.5 mr-1" />
              Check Convergence
            </span>
          )}
          <span className="badge badge-blue">
            Max R-hat: {maxRhat.toFixed(4)}
          </span>
        </div>
      </div>

      {/* ── KPI Grid ── */}
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3">
        <StatCard label="R² Score"         value={summary.fitMetrics.r2.toFixed(3)}        variant="gold"     sub="In-sample fit" />
        <StatCard label="RMSE (log scale)" value={summary.fitMetrics.rmse.toFixed(4)}      variant="blue"     sub="Residual error" />
        <StatCard label="Posterior Draws"  value={summary.nPosteriorDraws.toLocaleString()} variant="default"  sub="2 chains × draws" />
        <StatCard label="Training ZIPs"    value={summary.nTrainingZips.toLocaleString()}   variant="default"  sub="Stratified by state" />
        <StatCard label="Predictors"       value={summary.featureNames.length}              variant="default"  sub="Engineered features" />
        <StatCard
          label="Year Range"
          value={`${summary.yearRange[0]}–${summary.yearRange[1]}`}
          variant="default"
          sub="12-year panel"
        />
      </div>

      {/* ── Charts Row ── */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">

        {/* Year effects */}
        <div className="card p-5 lg:col-span-3">
          <div className="flex items-center justify-between mb-4">
            <p className="section-title">Year Effects (Macro Trend)</p>
            <Link to="/temporal" className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
              Full view <ArrowRightIcon className="w-3 h-3" />
            </Link>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={years} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="yearGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}   />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,45,74,0.8)" />
              <XAxis dataKey="year" tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="mean"
                name="Effect"
                stroke="#3b82f6"
                strokeWidth={2}
                fill="url(#yearGrad)"
                dot={{ r: 3, fill: "#3b82f6", strokeWidth: 0 }}
              />
            </AreaChart>
          </ResponsiveContainer>
          <p className="text-xs text-slate-500 mt-2 text-center">
            2020 spike (+0.318) = COVID stimulus + stock market recovery
          </p>
        </div>

        {/* Top states */}
        <div className="card p-5 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <p className="section-title">State Effects (Top 10)</p>
            <Link to="/geographic" className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
              All states <ArrowRightIcon className="w-3 h-3" />
            </Link>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={states} layout="vertical" margin={{ top: 0, right: 8, left: 24, bottom: 0 }}>
              <XAxis type="number" tick={{ fill: "#64748b", fontSize: 10 }} />
              <YAxis type="category" dataKey="state" tick={{ fill: "#94a3b8", fontSize: 11 }} width={28} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="mean" name="Effect" radius={[0, 3, 3, 0]}>
                {states.map((s, i) => (
                  <Cell key={i} fill={s.mean >= 0 ? "#10b981" : "#ef4444"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Beta Summary + Recent Audit ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        {/* Beta coefficients mini */}
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <p className="section-title">Feature Coefficients (β)</p>
            <Link to="/model" className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
              Full analysis <ArrowRightIcon className="w-3 h-3" />
            </Link>
          </div>
          <div className="space-y-2">
            {[...betas].sort((a, b) => b.mean - a.mean).map((b) => {
              const pct = Math.abs(b.mean) / 0.15;
              return (
                <div key={b.feature} className="flex items-center gap-3">
                  <span className="text-xs text-slate-400 w-36 truncate font-mono flex-shrink-0">
                    {fmtLabel(b.feature)}
                  </span>
                  <div className="flex-1 h-5 bg-navy-900 rounded overflow-hidden relative">
                    <div
                      className="h-full rounded transition-all duration-500"
                      style={{
                        width: `${Math.min(pct * 100, 100)}%`,
                        backgroundColor: betaColor(b.mean),
                        opacity: 0.8,
                      }}
                    />
                  </div>
                  <span
                    className="text-xs font-mono w-16 text-right flex-shrink-0"
                    style={{ color: betaColor(b.mean) }}
                  >
                    {b.mean >= 0 ? "+" : ""}{b.mean.toFixed(4)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Recent predictions */}
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <p className="section-title">Recent Predictions</p>
            <Link to="/audit" className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1">
              Full log <ArrowRightIcon className="w-3 h-3" />
            </Link>
          </div>
          {audit.length === 0 ? (
            <div className="text-center py-8 text-slate-500 text-sm">
              No predictions yet.{" "}
              <Link to="/predictor" className="text-blue-400 hover:underline">
                Run a prediction →
              </Link>
            </div>
          ) : (
            <div className="space-y-1">
              {audit.map((entry, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between py-2 px-3 rounded-lg table-row-hover"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="font-mono text-sm text-slate-200">{entry.query.zip}</span>
                    <span className="badge badge-gray">{entry.query.state}</span>
                    <span className="text-xs text-slate-500">{entry.query.year}</span>
                  </div>
                  <div className="text-right flex-shrink-0 ml-2">
                    <p className="text-sm font-semibold text-emerald-400 font-mono">
                      {entry.predicted?.income_mean
                        ? fmt$$(entry.predicted.income_mean)
                        : "—"}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
          <div className="mt-4 pt-3 border-t border-navy-600/40">
            <Link to="/predictor" className="btn-primary w-full">
              New Prediction
            </Link>
          </div>
        </div>
      </div>

      {/* ── Model spec card ── */}
      <div className="card p-5">
        <p className="section-title mb-4">Model Specification</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-navy-900 rounded-lg p-4 col-span-1 md:col-span-2">
            <p className="label mb-2">Likelihood</p>
            <code className="text-xs text-amber-400 font-mono leading-relaxed block">
              log(1 + AVG_INCOME<sub>z,t</sub>) = α + β·X<sub>z,t</sub> + u<sub>state[z]</sub> + γ<sub>t</sub> + ε<sub>z,t</sub>
            </code>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-400">
              <div><span className="text-slate-300">α</span> = global intercept</div>
              <div><span className="text-slate-300">β</span> = {summary.featureNames.length} feature coefficients</div>
              <div><span className="text-slate-300">u_state</span> = hierarchical state effect</div>
              <div><span className="text-slate-300">γ_t</span> = year fixed effect</div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="bg-navy-900 rounded-lg p-3">
              <p className="label text-[10px] mb-1">Global Intercept (α)</p>
              <p className="font-mono text-sm text-amber-400">{summary.alpha.mean.toFixed(4)}</p>
              <p className="text-xs text-slate-500">94% HDI [{summary.alpha.hdi_3.toFixed(3)}, {summary.alpha.hdi_97.toFixed(3)}]</p>
            </div>
            <div className="bg-navy-900 rounded-lg p-3">
              <p className="label text-[10px] mb-1">σ_obs (Residual)</p>
              <p className="font-mono text-sm text-slate-300">{summary.sigmaObs.mean.toFixed(4)}</p>
              <p className="text-xs text-slate-500">94% HDI [{summary.sigmaObs.hdi_3.toFixed(3)}, {summary.sigmaObs.hdi_97.toFixed(3)}]</p>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
