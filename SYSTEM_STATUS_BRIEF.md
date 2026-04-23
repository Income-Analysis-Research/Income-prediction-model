# System Status Brief

Date: 2026-04-23
Project: Regional Income Dynamics Modeling (US ZIP-level, IRS SOI 2011-2022)

## 1) What this project does

This repository builds and runs a Hierarchical Bayesian Panel Regression to explain regional income structure at ZIP level across time.

Core model:
Income(z,t) = beta * X(z,t) + state_effect + year_effect + error

Response variable:
log(1 + AVG_INCOME)

Primary goal:
Explain regional income dynamics with uncertainty (not individual-income prediction).

## 2) Current implementation status

- Model script exists and runs: hierarchical_bayesian_panel.py
- Data file is present in CSV and Parquet forms
- Results artifacts are present in results folder (CSV summaries, NetCDF posterior trace, plots)
- Writer-facing docs exist (README, WRITER_GUIDE, CHANGES)
- Research references are stored in references folder
- Working tree note: CHANGES.md and Full-Paper-template.docx are currently untracked in git

## 3) Directory map (what each item is)

- .git/
  - Git metadata

- .gitignore
  - Ignore rules for local/runtime files

- .venv/
  - Main Python virtual environment used to run project

- .venv-1/
  - Secondary environment folder (legacy/alternate)

- hierarchical_bayesian_panel.py
  - Main end-to-end pipeline:
    - load data
    - engineer features
    - build Bayesian model (PyMC)
    - sample with NUTS
    - evaluate and save outputs

- zip_income_panel_structural_2011_2022.csv
  - Main dataset used by script

- zip_income_panel_structural_2011_2022.parquet
  - Columnar version of same dataset

- requirements.txt
  - Python dependency list

- README.md
  - Setup, run instructions, model summary, result interpretation

- WRITER_GUIDE.md
  - Detailed writing guide for paper authors

- CHANGES.md
  - Change log describing recently added engineered features and decisions

- Full-Paper-template.docx
  - Conference/journal formatting template for manuscript writing

- references/
  - Input research papers used for methodological grounding:
    - Estimation of Median Household Income for Small Areas iima.pdf
    - estimation-methods-of-inequality-of-the-population-incomes.pdf
    - kalivoshko2020.pdf
    - sherri2021.pdf
    - zheng2009.pdf

- results/
  - Generated outputs from model runs:
    - trace.nc (full posterior trace)
    - summary_global.csv (alpha and sigma summaries)
    - beta_coefficients.csv (feature effects)
    - state_effects.csv, year_effects.csv
    - Plot files: trace_plot, beta_forest, state_effects, year_effects, posterior_predictive, prior_vs_posterior

## 4) Feature set currently in model

Base composition-share features:
- share_wages
- share_business
- share_capital_gains
- share_interest
- share_dividends
- share_unemployment
- share_social_security

Added structural features:
- income_stability_index (year-over-year composition volatility)
- income_diversity_index (entropy-based source diversification)
- shock_response_index (ZIP COVID shock proxy from 2019 to 2020 income change)

## 5) How to run quickly

1. Activate environment
2. Install requirements if needed
3. Run:
   python hierarchical_bayesian_panel.py

Outputs are regenerated into results folder.

## 6) Known assumptions/config choices

- Script currently points to local dataset filename in project root
- MCMC configuration values are set in script constants (draws, tune, chains, seed)
- Balanced-panel sampling logic currently assumes 12-year panel for full ZIP selection
- Shock feature is tied to years 2019 and 2020 by design

These are configurable by code edits if needed for broader reuse.

## 7) Scope boundaries (important)

In-scope:
- Regional ZIP-level income dynamics
- Bayesian uncertainty and interpretable coefficients

Out-of-scope:
- Individual income prediction
- Deep learning / clustering / spatial graph pipelines
- Reproducing or copying methods from the IIM paper

## 8) Recommended handoff starting points

For engineers:
- Start with hierarchical_bayesian_panel.py and results folder

For paper writers:
- Start with WRITER_GUIDE.md, then README.md, then CHANGES.md

For reviewers:
- Validate model outputs from results folder, then inspect assumptions in script constants
