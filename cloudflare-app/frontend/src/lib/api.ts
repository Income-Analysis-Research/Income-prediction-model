import type {
  ModelSummary,
  BetasResponse,
  StatesResponse,
  YearsResponse,
  ZipSearchResponse,
  PredictionResult,
  AuditLogResponse,
  AuditStatsResponse,
  HealthResponse,
  ApiError,
} from "../types";

const BASE: string = import.meta.env.VITE_API_URL ?? "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as ApiError;
    throw new Error(body.error ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    const err = data as ApiError;
    const e = new Error(err.error ?? `HTTP ${res.status}`) as Error & { hint?: string; fallback_year?: number };
    e.hint         = err.hint;
    e.fallback_year = err.fallback_year;
    throw e;
  }
  return data as T;
}

export const api = {
  health:     () => get<HealthResponse>("/api/health"),
  summary:    () => get<ModelSummary>("/api/model/summary"),
  betas:      () => get<BetasResponse>("/api/model/betas"),
  states:     (sort = "mean", order = "desc") =>
    get<StatesResponse>(`/api/model/states?sort=${sort}&order=${order}`),
  years:      () => get<YearsResponse>("/api/model/years"),
  zipSearch:  (q: string, year: number) =>
    get<ZipSearchResponse>(`/api/zips/search?q=${encodeURIComponent(q)}&year=${year}`),
  predict:    (zip: string, year: number) =>
    post<PredictionResult>("/api/predict", { zip, year }),
  auditLog:   (limit = 50) =>
    get<AuditLogResponse>(`/api/audit-log?limit=${limit}`),
  auditStats: () => get<AuditStatsResponse>("/api/audit-log/stats"),
};

// Formatting utilities
export function fmt$$(n: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(n);
}

export function fmtPct(n: number, decimals = 1): string {
  return `${(n * 100).toFixed(decimals)}%`;
}

export function fmtNum(n: number, decimals = 4): string {
  return n.toFixed(decimals);
}

export function fmtLabel(feature: string): string {
  return feature
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace("Share ", "")
    .replace("Income ", "");
}

export function effectColor(value: number): string {
  if (value > 0.15)  return "#10b981";
  if (value > 0.05)  return "#34d399";
  if (value > 0)     return "#6ee7b7";
  if (value > -0.05) return "#fca5a5";
  if (value > -0.15) return "#f87171";
  return "#ef4444";
}

export function betaColor(value: number): string {
  return value >= 0 ? "#10b981" : "#ef4444";
}
