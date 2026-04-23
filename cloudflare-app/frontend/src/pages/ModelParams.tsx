import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
  ResponsiveContainer, CartesianGrid, ErrorBar,
} from "recharts";
import { InformationCircleIcon } from "@heroicons/react/24/outline";
import { PageLoader, ErrorState } from "../components/LoadingSpinner";
import StatCard from "../components/StatCard";
import { api, fmtLabel } from "../lib/api";
import type { ModelSummary, BetaEntry } from "../types";

interface Data { summary: ModelSummary; betas: BetaEntry[] }

const FEATURE_DESC: Record<string, string> = {
  share_wages:           "Fraction of AGI from wages/salaries. Wage-dependent ZIPs are structurally lower-income.",
  share_business:        "Fraction from self-employment/business income.",
  share_capital_gains:   "Fraction from net capital gains. Strongest positive predictor — high-income ZIPs are investment-driven.",
  share_interest:        "Fraction from taxable interest (savings income).",
  share_dividends:       "Fraction from ordinary dividends. Wealth-derived passive income.",
  share_unemployment:    "Fraction from unemployment compensation. Strongest negative — structural low-income marker.",
  share_social_security: "Fraction from Social Security benefits. Retirement/low-income indicator.",
  income_stability_index:"Year-over-year volatility of income composition. High = boom/bust cycles.",
  income_diversity_index:"Shannon entropy of income sources. High = diversified economy.",
  shock_response_index:  "2019→2020 income change. Encodes ZIP structural resilience to COVID shock.",
};

function TooltipCustom({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value: number[]; name: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const [lo, hi] = payload[0].value as unknown as number[];
  return (
    <div className="tooltip-base max-w-xs">
      <p className="font-semibold text-amber-400 mb-1">{label}</p>
      <p className="text-slate-300">Mean: <span className="font-mono">{(lo + hi) / 2}</span></p>
      <p className="text-slate-400 text-[10px] mt-1">{FEATURE_DESC[label ?? ""] ?? ""}</p>
    </div>
  );
}

export default function ModelParams() {
  const [data, setData]   = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view,  setView]  = useState<"chart" | "table">("chart");

  useEffect(() => {
    Promise.all([api.summary(), api.betas()])
      .then(([summary, betasR]) => setData({ summary, betas: betasR.betas }))
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!data)  return <PageLoader message="Loading model parameters…" />;

  const { summary, betas } = data;

  const sortedBetas = [...betas].sort((a, b) => b.mean - a.mean);
  const chartData = sortedBetas.map((b) => ({
    name:   fmtLabel(b.feature),
    mean:   b.mean,
    hdi_lo: b.hdi_3,
    hdi_hi: b.hdi_97,
    r_hat:  b.r_hat,
  }));

  const globalParams = [
    { key: "alpha",      label: "α (Intercept)",       p: summary.alpha },
    { key: "sigmaState", label: "σ_state",              p: summary.sigmaState },
    { key: "sigmaYear",  label: "σ_year",               p: summary.sigmaYear },
    { key: "sigmaObs",   label: "σ_obs (Residual)",     p: summary.sigmaObs },
  ];

  return (
    <div className="space-y-6 pb-20 lg:pb-0">

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Model Parameters</h1>
        <p className="text-sm text-slate-500 mt-1">
          Posterior summaries from {summary.nPosteriorDraws.toLocaleString()} MCMC draws · 94% HDI reported
        </p>
      </div>

      {/* Fit metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard label="R² (in-sample)"  value={summary.fitMetrics.r2.toFixed(3)}            variant="gold"     sub="Variance explained" />
        <StatCard label="RMSE (log scale)" value={summary.fitMetrics.rmse.toFixed(4)}          variant="blue"     sub="~26% income scale" />
        <StatCard label="MAE (log scale)"  value={summary.fitMetrics.mae.toFixed(4)}           variant="default"  sub="Median abs error" />
        <StatCard label="Gini (residuals)" value={summary.fitMetrics.gini_residuals.toFixed(3)} variant="default"  sub="Residual inequality" />
      </div>

      {/* Global params */}
      <div className="card p-5">
        <p className="section-title mb-4">Global Parameters</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-navy-600/40">
                <th className="text-left py-2 px-3 label">Parameter</th>
                <th className="text-right py-2 px-3 label">Mean</th>
                <th className="text-right py-2 px-3 label">Std</th>
                <th className="text-right py-2 px-3 label">HDI 3%</th>
                <th className="text-right py-2 px-3 label">HDI 97%</th>
                <th className="text-right py-2 px-3 label">R-hat</th>
              </tr>
            </thead>
            <tbody>
              {globalParams.map(({ key, label, p }) => (
                <tr key={key} className="border-b border-navy-700/30 table-row-hover">
                  <td className="py-2.5 px-3 font-mono text-amber-400">{label}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-200">{p.mean.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-400">{p.sd.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-400">{p.hdi_3.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-400">{p.hdi_97.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right">
                    <span className={(p.r_hat ?? 1) < 1.05 ? "badge-green badge" : "badge-gold badge"}>
                      {(p.r_hat ?? 1).toFixed(4)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Beta coefficients */}
      <div className="card p-5">
        <div className="flex items-center justify-between mb-4">
          <p className="section-title">
            Feature Coefficients (β) — {betas.length} predictors
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => setView("chart")}
              className={view === "chart" ? "btn-primary text-xs py-1.5 px-3" : "btn-secondary text-xs py-1.5 px-3"}
            >
              Forest Plot
            </button>
            <button
              onClick={() => setView("table")}
              className={view === "table" ? "btn-primary text-xs py-1.5 px-3" : "btn-secondary text-xs py-1.5 px-3"}
            >
              Table
            </button>
          </div>
        </div>

        {view === "chart" ? (
          <>
            <div className="flex gap-4 mb-3">
              <span className="flex items-center gap-1.5 text-xs text-emerald-400">
                <span className="w-3 h-3 rounded bg-emerald-500" /> Positive effect
              </span>
              <span className="flex items-center gap-1.5 text-xs text-red-400">
                <span className="w-3 h-3 rounded bg-red-500" /> Negative effect
              </span>
            </div>
            <ResponsiveContainer width="100%" height={Math.max(260, betas.length * 38)}>
              <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 80, left: 110, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(30,45,74,0.8)" />
                <XAxis
                  type="number"
                  domain={["dataMin - 0.02", "dataMax + 0.02"]}
                  tick={{ fill: "#64748b", fontSize: 11 }}
                  tickLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                  width={105}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload as typeof chartData[0];
                    return (
                      <div className="tooltip-base max-w-xs">
                        <p className="font-semibold text-amber-400">{d.name}</p>
                        <p className="text-slate-300 text-xs mt-1">Mean: <span className="font-mono">{d.mean.toFixed(4)}</span></p>
                        <p className="text-slate-400 text-xs">94% HDI: [{d.hdi_lo.toFixed(4)}, {d.hdi_hi.toFixed(4)}]</p>
                        <p className="text-slate-400 text-xs">R-hat: {d.r_hat.toFixed(4)}</p>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="mean" radius={[0, 3, 3, 0]}>
                  {chartData.map((d, i) => (
                    <Cell key={i} fill={d.mean >= 0 ? "#10b981" : "#ef4444"} />
                  ))}
                  <ErrorBar
                    dataKey={(d: typeof chartData[0]) => [
                      Math.abs(d.mean - d.hdi_lo),
                      Math.abs(d.hdi_hi - d.mean),
                    ]}
                    width={4}
                    strokeWidth={2}
                    stroke="#94a3b8"
                    direction="x"
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-navy-600/40">
                  <th className="text-left py-2 px-3 label">Feature</th>
                  <th className="text-right py-2 px-3 label">Mean</th>
                  <th className="text-right py-2 px-3 label">Std</th>
                  <th className="text-right py-2 px-3 label">HDI 3%</th>
                  <th className="text-right py-2 px-3 label">HDI 97%</th>
                  <th className="text-right py-2 px-3 label">R-hat</th>
                  <th className="text-left py-2 px-3 label hidden lg:table-cell">Direction</th>
                </tr>
              </thead>
              <tbody>
                {sortedBetas.map((b) => (
                  <tr key={b.feature} className="border-b border-navy-700/30 table-row-hover">
                    <td className="py-2.5 px-3 font-mono text-xs text-slate-300">{b.feature}</td>
                    <td className="py-2.5 px-3 text-right font-mono" style={{ color: b.mean >= 0 ? "#10b981" : "#ef4444" }}>
                      {b.mean >= 0 ? "+" : ""}{b.mean.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-400">{b.sd.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-400">{b.hdi_3.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-400">{b.hdi_97.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right">
                      <span className={b.r_hat < 1.05 ? "badge-green badge" : "badge-gold badge"}>
                        {b.r_hat.toFixed(4)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 hidden lg:table-cell">
                      <span className={b.mean >= 0 ? "badge-green badge" : "badge-red badge"}>
                        {b.mean >= 0 ? "Positive" : "Negative"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Interpretation note */}
        <div className="mt-4 pt-4 border-t border-navy-600/40 flex gap-2">
          <InformationCircleIcon className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-slate-500 leading-relaxed">
            All features are standardised (mean=0, std=1) before entering the model.
            Coefficients represent effect of a 1-standard-deviation increase on log(1+income).
            94% Highest Density Interval (HDI) is a Bayesian credible interval.
          </p>
        </div>
      </div>
    </div>
  );
}
