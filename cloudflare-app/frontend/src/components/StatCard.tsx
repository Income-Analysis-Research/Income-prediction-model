import type { ReactNode } from "react";
import clsx from "clsx";

interface StatCardProps {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  delta?: number | null;
  variant?: "default" | "gold" | "positive" | "negative" | "blue";
  loading?: boolean;
  className?: string;
  icon?: ReactNode;
}

export default function StatCard({
  label,
  value,
  sub,
  delta,
  variant = "default",
  loading = false,
  className,
  icon,
}: StatCardProps) {
  const accentClass = {
    default:  "border-navy-600/40",
    gold:     "border-amber-500/30",
    positive: "border-emerald-500/30",
    negative: "border-red-500/30",
    blue:     "border-blue-500/30",
  }[variant];

  const valueClass = {
    default:  "text-slate-100",
    gold:     "text-amber-400",
    positive: "text-emerald-400",
    negative: "text-red-400",
    blue:     "text-blue-400",
  }[variant];

  if (loading) {
    return (
      <div className={clsx("card p-5", className)}>
        <div className="animate-pulse space-y-2">
          <div className="h-3 w-24 bg-navy-600 rounded" />
          <div className="h-8 w-32 bg-navy-600 rounded mt-1" />
          <div className="h-3 w-20 bg-navy-700 rounded" />
        </div>
      </div>
    );
  }

  return (
    <div
      className={clsx(
        "card p-5 border-t-2 transition-all duration-200 hover:border-opacity-70",
        accentClass,
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <p className="label mb-2 truncate">{label}</p>
          <p className={clsx("value-lg", valueClass)}>{value}</p>
          {(sub || delta != null) && (
            <div className="flex items-center gap-2 mt-1.5">
              {sub && <p className="text-xs text-slate-500">{sub}</p>}
              {delta != null && (
                <span
                  className={clsx(
                    "text-xs font-semibold",
                    delta >= 0 ? "text-emerald-400" : "text-red-400"
                  )}
                >
                  {delta >= 0 ? "▲" : "▼"} {Math.abs(delta).toFixed(2)}
                </span>
              )}
            </div>
          )}
        </div>
        {icon && (
          <div className="ml-3 flex-shrink-0 opacity-40">{icon}</div>
        )}
      </div>
    </div>
  );
}
