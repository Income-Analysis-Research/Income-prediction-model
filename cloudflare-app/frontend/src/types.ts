export interface GlobalParam {
  mean: number;
  sd: number;
  hdi_3: number;
  hdi_97: number;
  r_hat?: number;
}

export interface BetaEntry {
  feature: string;
  mean: number;
  sd: number;
  hdi_3: number;
  hdi_97: number;
  r_hat: number;
}

export interface StateEntry {
  state: string;
  mean: number;
  sd: number;
  hdi_3: number;
  hdi_97: number;
}

export interface YearEntry {
  year: number;
  mean: number;
  sd: number;
  hdi_3: number;
  hdi_97: number;
}

export interface FitMetrics {
  r2: number;
  rmse: number;
  mae: number;
  gini_residuals: number;
}

export interface ModelSummary {
  alpha: GlobalParam;
  sigmaState: GlobalParam;
  sigmaYear: GlobalParam;
  sigmaObs: GlobalParam;
  fitMetrics: FitMetrics;
  featureNames: string[];
  nPosteriorDraws: number;
  nTrainingZips: number;
  yearRange: [number, number];
  generatedAt: string | null;
}

export interface BetasResponse {
  betas: BetaEntry[];
  featureNames: string[];
}

export interface StatesResponse {
  states: StateEntry[];
  count: number;
}

export interface YearsResponse {
  years: YearEntry[];
}

export interface ZipSearchResult {
  zip: string;
  state: string;
  avg_income: number;
}

export interface ZipSearchResponse {
  results: ZipSearchResult[];
  query: string;
  year: number;
}

export interface PredictionResult {
  query: {
    zip: string;
    year: number;
    state: string;
  };
  predicted: {
    income_mean: number;
    income_lo: number;
    income_hi: number;
    log_income_mean: number;
  };
  decomposition: {
    alpha: number;
    beta_dot_x: number;
    state_effect: number;
    year_effect: number;
  };
  feature_vector: Record<string, number>;
  actual_avg_income: number;
  timestamp: string;
}

export interface AuditEntry {
  event: string;
  query: {
    zip: string;
    year: number;
    state: string;
  };
  predicted: {
    income_mean: number;
    income_lo: number;
    income_hi: number;
    log_income_mean: number;
  };
  decomposition: {
    alpha: number;
    beta_dot_x: number;
    state_effect: number;
    year_effect: number;
  };
  feature_vector: Record<string, number>;
  actual_avg_income: number;
  timestamp: string;
}

export interface AuditLogResponse {
  entries: AuditEntry[];
  total: number;
}

export interface AuditStatsResponse {
  total: number;
  byState: Record<string, number>;
  byYear: Record<number, number>;
}

export interface HealthResponse {
  status: string;
  model: string;
  version: string;
  features: number;
  posterior_draws: number;
  zip_year_pairs: number;
  data_generated_at: string | null;
  timestamp: string;
}

export type ApiError = {
  error: string;
  hint?: string;
  fallback_year?: number;
};
