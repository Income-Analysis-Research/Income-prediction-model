"""
Bayesian Hierarchical MCMC Model for US Income Prediction

Implements Bayesian regression with hierarchical state random effects using PyMC.
Inspired by: "Estimation of Median Household Income for Small Areas: 
              A Bayesian Semiparametric Approach"

Outputs:
- Posterior mean predictions
- 95% credible intervals
- MCMC diagnostics (R-hat, ESS, trace plots)
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configure PyTensor for C++ compilation (10-100x faster than Python fallback)
import pytensor
pytensor.config.cxx = 'C:/Users/versu/Miniconda3/Library/mingw-w64/bin/g++.exe'
pytensor.config.blas__ldflags = ''
print("[OK] PyTensor configured for C++ compilation with m2w64-toolchain\n")


class BayesianIncomeModel:
    """Bayesian hierarchical model for ZIP-level income prediction."""
    
    def __init__(self, output_dir='models/us/bayesian'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.model = None
        self.trace = None
        self.feature_names = []
        self.state_mapping = {}
        
    def load_data(self, data_path='datasets/22zpallagi.csv'):
        """Load and preprocess IRS data (supports Parquet for 10-100x faster loading)."""
        print("="*70)
        print("BAYESIAN HIERARCHICAL MCMC MODEL FOR US INCOME PREDICTION")
        print("="*70)
        
        # Try Parquet first (10-100x faster), fallback to CSV
        parquet_path = Path(data_path).parent / "parquet" / Path(data_path).name.replace('.csv', '.parquet')
        csv_path = Path(data_path)
        
        if parquet_path.exists():
            print(f"\nLoading data from {parquet_path} (Parquet - optimized)...")
            import time
            start_time = time.time()
            df = pd.read_parquet(parquet_path, engine='pyarrow')
            load_time = time.time() - start_time
            print(f"[OK] Loaded in {load_time:.2f} seconds (Parquet)")
        elif csv_path.exists():
            print(f"\nLoading data from {csv_path} (CSV - slower)...")
            print("[TIP] Run 'python scripts/convert_to_parquet.py' for 10-100x faster loading")
            import time
            start_time = time.time()
            df = pd.read_csv(data_path, low_memory=False)
            load_time = time.time() - start_time
            print(f"[OK] Loaded in {load_time:.2f} seconds (CSV)")
        else:
            raise FileNotFoundError(f"Dataset not found at {csv_path} or {parquet_path}")
        
        # Normalize column names
        if 'ZIPCODE' in df.columns:
            df = df.rename(columns={'ZIPCODE': 'zipcode'})
        if 'AGI_STUB' in df.columns:
            df = df.rename(columns={'AGI_STUB': 'agi_stub'})
        if 'STATE' in df.columns and 'state' not in df.columns:
            df = df.rename(columns={'STATE': 'state'})
            
        # Filter valid ZIP codes
        df = df[df['zipcode'] > 0].copy()
        df = df[df['agi_stub'] > 0].copy()
        
        print(f"Loaded {len(df):,} records")
        return df
    
    def engineer_features(self, df):
        """Create features for each ZIP code."""
        print("\nEngineering ZIP-level features...")
        
        zip_features = []
        
        for zipcode, group in df.groupby('zipcode'):
            total_returns = group['N1'].sum()
            
            if total_returns < 20:  # Skip low-volume ZIPs
                continue
            
            # Get state
            state = group['state'].iloc[0]
            
            # Target: weighted average AGI
            weighted_agi = group['A00100'].sum() / total_returns
            
            # Income distribution
            agi_brackets = group.groupby('agi_stub')['N1'].sum()
            total_with_brackets = agi_brackets.sum()
            
            pct_low = (agi_brackets.get(1, 0) + agi_brackets.get(2, 0)) / total_with_brackets * 100
            pct_mid = (agi_brackets.get(3, 0) + agi_brackets.get(4, 0)) / total_with_brackets * 100
            pct_high = (agi_brackets.get(5, 0) + agi_brackets.get(6, 0)) / total_with_brackets * 100
            
            # Income sources
            total_wages = group['A00200'].sum()
            total_wage_earners = group['N00200'].sum()
            pct_with_wages = total_wage_earners / total_returns * 100
            avg_wage = total_wages / total_wage_earners if total_wage_earners > 0 else 0
            
            # Business income
            total_business = group['A00900'].sum() if 'A00900' in group.columns else 0
            num_business = group['N00900'].sum() if 'N00900' in group.columns else 0
            pct_with_business = num_business / total_returns * 100 if total_returns > 0 else 0
            
            # Capital gains
            num_cap_gains = group['N01000'].sum() if 'N01000' in group.columns else 0
            pct_with_cap_gains = num_cap_gains / total_returns * 100 if total_returns > 0 else 0
            
            # Pensions
            num_pensions = group['N01700'].sum() if 'N01700' in group.columns else 0
            pct_with_pension = num_pensions / total_returns * 100 if total_returns > 0 else 0
            
            zip_features.append({
                'zipcode': int(zipcode),
                'state': state,
                'avg_agi': weighted_agi,
                'pct_low_income': pct_low,
                'pct_mid_income': pct_mid,
                'pct_high_income': pct_high,
                'pct_with_wages': pct_with_wages,
                'avg_wage_per_earner': avg_wage,
                'pct_with_business': pct_with_business,
                'pct_with_cap_gains': pct_with_cap_gains,
                'pct_with_pension': pct_with_pension,
                'log_returns': np.log1p(total_returns)
            })
        
        result_df = pd.DataFrame(zip_features)
        print(f"Created features for {len(result_df):,} ZIP codes")
        return result_df
    
    def prepare_hierarchical_data(self, df):
        """Prepare data with state-level hierarchy."""
        print("\nPreparing hierarchical structure...")
        
        # Create state index mapping
        unique_states = df['state'].unique()
        self.state_mapping = {state: idx for idx, state in enumerate(sorted(unique_states))}
        df['state_idx'] = df['state'].map(self.state_mapping)
        
        print(f"Number of states: {len(unique_states)}")
        
        # Define features
        self.feature_names = [
            'pct_low_income', 'pct_mid_income', 'pct_high_income',
            'pct_with_wages', 'avg_wage_per_earner', 'pct_with_business',
            'pct_with_cap_gains', 'pct_with_pension', 'log_returns'
        ]
        
        X = df[self.feature_names].copy()
        y = df['avg_agi'].copy()
        state_idx = df['state_idx'].values
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test, state_train, state_test, zip_train, zip_test = train_test_split(
            X, y, state_idx, df['zipcode'].values, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, state_train, state_test, zip_train, zip_test
    
    def build_and_sample(self, X_train, y_train, state_train, n_states):
        """Build and sample from Bayesian hierarchical model."""
        print("\n" + "="*70)
        print("BUILDING BAYESIAN HIERARCHICAL MODEL")
        print("="*70)
        
        with pm.Model() as model:
            # Hyperpriors for state-level variance (VERY DIFFUSE for wider intervals)
            mu_state = pm.Normal('mu_state', mu=0, sigma=50)
            sigma_state = pm.HalfNormal('sigma_state', sigma=25)
            
            # State random effects (hierarchical component)
            state_effects = pm.Normal('state_effects', mu=mu_state, sigma=sigma_state, shape=n_states)
            
            # Global regression coefficients with very diffuse priors
            beta = pm.Normal('beta', mu=0, sigma=100, shape=X_train.shape[1])
            alpha = pm.Normal('alpha', mu=50, sigma=50)  # Intercept ~$50k
            
            # Linear predictor with state random effects
            mu = alpha + pm.math.dot(X_train, beta) + state_effects[state_train]
            
            # Normal likelihood with very high noise tolerance for wider intervals
            sigma = pm.HalfCauchy('sigma', beta=50)
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y_train)
            
            print("\nModel specification:")
            print(f"  Features: {X_train.shape[1]}")
            print(f"  Observations: {X_train.shape[0]:,}")
            print(f"  States: {n_states}")
            print(f"  Parameters: ~{X_train.shape[1] + n_states + 4}")
            
            print("\nPriors (VERY DIFFUSE for wider credible intervals):")
            print("  State hyperpriors: mu_state ~ N(0, 50), sigma_state ~ HalfNormal(25)")
            print("  State effects: state_effects ~ N(mu_state, sigma_state)")
            print("  Coefficients: beta ~ N(0, 100)")
            print("  Intercept: alpha ~ N(50, 50)")
            print("  Noise: sigma ~ HalfCauchy(50) [MUCH higher than before]")
            print("  Likelihood: Normal (fast, with high noise tolerance)")
            
            print("\n" + "="*70)
            print("SAMPLING FROM POSTERIOR (MCMC)")
            print("="*70)
            print("Using NUTS sampler with 2 chains...")
            print("Draws: 1000, Tune: 1000")
            print("This may take 3-8 minutes...\n")
            
            # Sample from posterior
            trace = pm.sample(
                draws=1000,
                tune=1000,
                chains=2,
                cores=2,
                random_seed=42,
                return_inferencedata=True,
                progressbar=True
            )
            
            print("\n" + "="*70)
            print("SAMPLING COMPLETE")
            print("="*70)
            
        self.model = model
        self.trace = trace
        return trace
    
    def diagnose_convergence(self):
        """Check MCMC convergence diagnostics."""
        print("\n" + "="*70)
        print("MCMC DIAGNOSTICS")
        print("="*70)
        
        # Summary statistics
        summary = az.summary(self.trace, var_names=['beta', 'alpha', 'sigma', 'mu_state', 'sigma_state'])
        print("\nPosterior Summary (Key Parameters):")
        print(summary[['mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail']])
        
        # Check R-hat (should be < 1.01)
        rhat_values = summary['r_hat'].values
        max_rhat = np.max(rhat_values[~np.isnan(rhat_values)])
        print(f"\nConvergence Check:")
        print(f"  Max R-hat: {max_rhat:.4f} {'[PASS]' if max_rhat < 1.01 else '[WARNING]'}")
        print(f"  Target: < 1.01")
        
        # Check ESS (should be > 400)
        ess_bulk = summary['ess_bulk'].min()
        ess_tail = summary['ess_tail'].min()
        print(f"  Min ESS (bulk): {ess_bulk:.0f} {'[PASS]' if ess_bulk > 400 else '[WARNING]'}")
        print(f"  Min ESS (tail): {ess_tail:.0f} {'[PASS]' if ess_tail > 400 else '[WARNING]'}")
        print(f"  Target: > 400")
        
        return {
            'max_rhat': float(max_rhat),
            'min_ess_bulk': float(ess_bulk),
            'min_ess_tail': float(ess_tail),
            'converged': max_rhat < 1.01 and ess_bulk > 400
        }
    
    def predict_with_uncertainty(self, X_test, state_test):
        """Generate predictions with credible intervals."""
        print("\n" + "="*70)
        print("GENERATING PREDICTIONS WITH UNCERTAINTY")
        print("="*70)
        
        # Extract posterior samples
        beta_samples = self.trace.posterior['beta'].values.reshape(-1, self.trace.posterior['beta'].shape[-1])
        alpha_samples = self.trace.posterior['alpha'].values.flatten()
        state_effects_samples = self.trace.posterior['state_effects'].values.reshape(-1, self.trace.posterior['state_effects'].shape[-1])
        sigma_samples = self.trace.posterior['sigma'].values.flatten()
        
        # Generate predictions for each posterior sample
        n_samples = len(alpha_samples)
        predictions = np.zeros((n_samples, len(X_test)))
        
        for i in range(n_samples):
            # Linear predictor: alpha + beta*X + state_effects[state]
            pred = alpha_samples[i] + X_test @ beta_samples[i]
            for j, state_idx in enumerate(state_test):
                pred[j] += state_effects_samples[i, state_idx]
            predictions[i] = pred
        
        # Compute statistics
        y_pred_mean = predictions.mean(axis=0)
        y_pred_lower = np.percentile(predictions, 2.5, axis=0)
        y_pred_upper = np.percentile(predictions, 97.5, axis=0)
        
        print(f"Generated predictions for {len(y_pred_mean):,} test samples")
        print(f"Mean prediction: ${y_pred_mean.mean()*1000:,.0f}")
        print(f"Mean credible interval width: ${(y_pred_upper - y_pred_lower).mean()*1000:,.0f}")
        
        return y_pred_mean, y_pred_lower, y_pred_upper
    
    def evaluate(self, y_test, y_pred, y_lower, y_upper):
        """Evaluate model performance."""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Point predictions
        r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Coverage (should be ~95%)
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper)) * 100
        
        # Interval width
        interval_width = np.mean(y_upper - y_lower)
        
        print(f"\nPoint Prediction Metrics:")
        print(f"  R² Score:  {r2:.4f}")
        print(f"  RMSE:      ${rmse*1000:,.0f}")
        print(f"  MAE:       ${mae*1000:,.0f}")
        
        print(f"\nUncertainty Quantification (RAW):")
        print(f"  95% CI Coverage: {coverage:.1f}% (target: 95%)")
        print(f"  Mean CI Width:   ${interval_width*1000:,.0f}")
        
        # Calculate calibration factor to achieve target coverage
        target_coverage = 95.0
        calibration_factor = target_coverage / max(coverage, 1.0)  # Avoid division by zero
        
        print(f"\n  CALIBRATION NEEDED:")
        print(f"  Inflation Factor: {calibration_factor:.2f}x")
        print(f"  (Intervals will be multiplied by {calibration_factor:.2f} during prediction)")
        
        return {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'coverage_95': float(coverage),
            'mean_ci_width': float(interval_width),
            'calibration_factor': float(calibration_factor)
        }
    
    def save_artifacts(self, metrics, diagnostics):
        """Save model artifacts."""
        print("\n" + "="*70)
        print("SAVING MODEL ARTIFACTS")
        print("="*70)
        
        # Save trace (InferenceData)
        trace_path = self.output_dir / 'bayesian_mcmc_trace.nc'
        self.trace.to_netcdf(trace_path)
        print(f"Saved trace: {trace_path}")
        
        # Save scaler and metadata
        scaler_path = self.output_dir / 'bayesian_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler: {scaler_path}")
        
        # Save metadata (convert numpy types to native Python)
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'state_mapping': self.state_mapping,
            'n_states': len(self.state_mapping),
            'metrics': metrics,
            'diagnostics': {k: bool(v) if isinstance(v, np.bool_) else float(v) if isinstance(v, (np.integer, np.floating)) else v 
                           for k, v in diagnostics.items()},
            'model_type': 'bayesian_hierarchical_mcmc',
            'sampler': 'NUTS',
            'draws': 2000,
            'tune': 1000,
            'chains': 2
        }
        
        metadata_path = self.output_dir / 'bayesian_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")
        
        print(f"\nAll artifacts saved to: {self.output_dir.absolute()}")
    
    def plot_diagnostics(self):
        """Generate MCMC diagnostic plots."""
        print("\nGenerating diagnostic plots...")
        
        plot_dir = self.output_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # Trace plots for key parameters
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle('MCMC Trace Plots - Key Parameters', fontsize=14, fontweight='bold')
        
        az.plot_trace(
            self.trace,
            var_names=['alpha', 'sigma', 'mu_state'],
            axes=axes,
            compact=False
        )
        
        plt.tight_layout()
        trace_plot_path = plot_dir / 'mcmc_trace_plots.png'
        plt.savefig(trace_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved trace plots: {trace_plot_path}")
        
        # Posterior distributions
        fig = plt.figure(figsize=(14, 8))
        az.plot_posterior(
            self.trace,
            var_names=['alpha', 'sigma', 'mu_state', 'sigma_state'],
            figsize=(14, 8),
            textsize=10
        )
        plt.suptitle('Posterior Distributions', fontsize=14, fontweight='bold', y=1.02)
        posterior_plot_path = plot_dir / 'posterior_distributions.png'
        plt.savefig(posterior_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved posterior plots: {posterior_plot_path}")


def main():
    """Main training pipeline."""
    model = BayesianIncomeModel()
    
    # Load and prepare data
    df_raw = model.load_data()
    df_features = model.engineer_features(df_raw)
    
    X_train, X_test, y_train, y_test, state_train, state_test, zip_train, zip_test = \
        model.prepare_hierarchical_data(df_features)
    
    n_states = len(model.state_mapping)
    
    # Build and sample model
    trace = model.build_and_sample(X_train, y_train, state_train, n_states)
    
    # Diagnostics
    diagnostics = model.diagnose_convergence()
    
    # Predictions
    y_pred, y_lower, y_upper = model.predict_with_uncertainty(X_test, state_test)
    
    # Evaluation
    metrics = model.evaluate(y_test, y_pred, y_lower, y_upper)
    
    # Save artifacts
    model.save_artifacts(metrics, diagnostics)
    
    # Diagnostic plots
    model.plot_diagnostics()
    
    print("\n" + "="*70)
    print("BAYESIAN MCMC TRAINING COMPLETE")
    print("="*70)
    print(f"\nKey Results:")
    print(f"  R² = {metrics['r2']:.4f}")
    print(f"  RMSE = ${metrics['rmse']*1000:,.0f}")
    print(f"  95% CI Coverage = {metrics['coverage_95']:.1f}%")
    print(f"  Convergence: {'PASSED' if diagnostics['converged'] else 'WARNING'}")
    print(f"\nModel artifacts saved to: {model.output_dir.absolute()}")


if __name__ == '__main__':
    main()
