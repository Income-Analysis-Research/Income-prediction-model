import { useEffect, useState } from "react";
import {
  ComposedChart, Area, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine,
  Legend, ErrorBar,
} from "recharts";
import { PageLoader, ErrorState } from "../components/LoadingSpinner";
import StatCard from "../components/StatCard";
import { api } from "../lib/api";
import type { YearEntry } from "../types";

const ANNOTATIONS: Record<number, { label: string; color: string }> = {
  2012: { label: "Post-GFC Recovery", color: "#64748b" },
  2017: { label: "Tax Cuts & Jobs Act", color: "#f59e0b" },
  2020: { label: "COVID-19 + Stimulus", color: "#ef4444" },
  2021: { label: "Post-Stimulus Elevated", color: "#f97316" },
};

const EVENT_CARDS = [
  {
    year: "2011–2019",
    title: "Post-GFC Recovery Phase",
    color: "border-slate-600",
    desc: "Sustained negative effects reflect the post-2008 recovery lag. Real income growth was slow despite nominal gains.",
  },
  {
    year: "2020",
    title: "COVID Shock & Response",
    color: "border-red-500",
    desc: "+0.318 spike driven by fiscal stimulus (CARES Act) and stock market recovery boosting capital gains for high-income filers.",
  },
  {
    year: "2021–2022",
    title: "Post-Stimulus Elevation",
    color: "border-amber-500",
    desc: "Effects remain elevated above pre-pandemic levels (+0.15 to +0.18), reflecting sustained fiscal expansion and asset price inflation.",
  },
];

function TooltipContent({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value: number; name: string; color: string }>;
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  const year = label;
  const ann = year ? ANNOTATIONS[year] : null;
  return (
    <div className="tooltip-base min-w-[200px]">
      <p className="font-bold text-amber-400">{year}</p>
      {ann && <p className="text-xs text-slate-400 mb-2">{ann.label}</p>}
      {payload.map((p, i) => (
        <p key={i} className="text-sm font-mono" style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? (p.value >= 0 ? "+" : "") + p.value.toFixed(4) : p.value}
        </p>
      ))}
    </div>
  );
}

export default function Temporal() {
  const [years, setYears] = useState<YearEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.years()
      .then((r) => setYears(r.years))
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <ErrorState message={error} />;
  if (!years) return <PageLoader message="Loading temporal effects…" />;

  const covidYear  = years.find((y) => y.year === 2020);
  const precovidMean = years.filter((y) => y.year < 2020 && y.year >= 2015).reduce((a, b) => a + b.mean, 0) / 5;
  const postcovidMean = years.filter((y) => y.year > 2020).reduce((a, b) => a + b.mean, 0) / years.filter((y) => y.year > 2020).length;

  const chartData = years.map((y) => ({
    year:   y.year,
    mean:   y.mean,
    hdi_lo: y.hdi_3,
    hdi_hi: y.hdi_97,
    upper_band: y.hdi_97,
    lower_band: y.hdi_3,
  }));

  return (
    <div className="space-y-6 pb-20 lg:pb-0">

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Temporal Analysis</h1>
        <p className="text-sm text-slate-500 mt-1">
          Year fixed effects absorb inflation, policy shocks, and macro trends · 94% HDI bands shown
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard
          label="COVID Spike (2020)"
          value={`+${covidYear?.mean.toFixed(4) ?? "—"}`}
          variant="negative"
          sub="Largest year effect"
        />
        <StatCard
          label="Pre-COVID Avg (2015–19)"
          value={precovidMean.toFixed(4)}
          variant="default"
          sub="Baseline level"
        />
        <StatCard
          label="Post-COVID Avg (2021–22)"
          value={`+${postcovidMean.toFixed(4)}`}
          variant="positive"
          sub="Elevated trajectory"
        />
        <StatCard
          label="σ_year"
          value="0.163"
          variant="blue"
          sub="Year heterogeneity"
        />
      </div>

      {/* Main chart */}
      <div className="card p-5">
        <p className="section-title mb-4">Year Effects with 94% Credible Interval Bands</p>
        <ResponsiveContainer width="100%" height={340}>
          <ComposedChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.15} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,45,74,0.8)" />
            <XAxis
              dataKey="year"
              tick={{ fill: "#64748b", fontSize: 12 }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 11 }}
              tickLine={false}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <Tooltip content={<TooltipContent />} />
            <Legend
              wrapperStyle={{ fontSize: "12px", color: "#94a3b8", paddingTop: "8px" }}
            />

            {/* HDI bands */}
            <Area
              type="monotone"
              dataKey="upper_band"
              name="HDI Upper (97%)"
              stroke="none"
              fill="url(#bandGrad)"
              legendType="none"
            />
            <Area
              type="monotone"
              dataKey="lower_band"
              name="HDI Lower (3%)"
              stroke="none"
              fill="none"
              legendType="none"
            />

            {/* Mean line */}
            <Line
              type="monotone"
              dataKey="mean"
              name="Posterior Mean"
              stroke="#3b82f6"
              strokeWidth={2.5}
              dot={{ r: 4, fill: "#3b82f6", strokeWidth: 0 }}
              activeDot={{ r: 6 }}
            />

            {/* Reference lines */}
            <ReferenceLine y={0} stroke="#374151" strokeDasharray="4 4" strokeWidth={1} />
            <ReferenceLine
              x={2020}
              stroke="#ef4444"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{ value: "COVID-19", position: "insideTopLeft", fill: "#ef4444", fontSize: 11 }}
            />
            <ReferenceLine
              y={precovidMean}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              strokeWidth={1}
              label={{ value: "Pre-COVID avg", position: "insideLeft", fill: "#f59e0b", fontSize: 10 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Event cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {EVENT_CARDS.map((ev) => (
          <div key={ev.year} className={`card p-5 border-l-2 ${ev.color}`}>
            <p className="text-xs text-slate-500 font-mono mb-1">{ev.year}</p>
            <p className="font-semibold text-slate-200 mb-2">{ev.title}</p>
            <p className="text-sm text-slate-400 leading-relaxed">{ev.desc}</p>
          </div>
        ))}
      </div>

      {/* Data table */}
      <div className="card p-5">
        <p className="section-title mb-4">Year-by-Year Posterior Estimates</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-navy-600/40">
                <th className="text-left py-2 px-3 label">Year</th>
                <th className="text-right py-2 px-3 label">Mean</th>
                <th className="text-right py-2 px-3 label">Std</th>
                <th className="text-right py-2 px-3 label">HDI 3%</th>
                <th className="text-right py-2 px-3 label">HDI 97%</th>
                <th className="text-left py-2 px-3 label hidden sm:table-cell">Event</th>
              </tr>
            </thead>
            <tbody>
              {years.map((y) => {
                const ann = ANNOTATIONS[y.year];
                return (
                  <tr
                    key={y.year}
                    className={[
                      "border-b border-navy-700/30 table-row-hover",
                      y.year === 2020 ? "bg-red-500/5" : "",
                    ].join(" ")}
                  >
                    <td className="py-2.5 px-3 font-semibold text-slate-200 font-mono">{y.year}</td>
                    <td
                      className="py-2.5 px-3 text-right font-mono font-semibold"
                      style={{ color: y.mean >= 0 ? "#10b981" : "#ef4444" }}
                    >
                      {y.mean >= 0 ? "+" : ""}{y.mean.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-400">{y.sd.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-400">{y.hdi_3.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right font-mono text-slate-400">{y.hdi_97.toFixed(4)}</td>
                    <td className="py-2.5 px-3 hidden sm:table-cell">
                      {ann ? (
                        <span className="badge badge-gold">{ann.label}</span>
                      ) : (
                        <span className="text-slate-600 text-xs">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
