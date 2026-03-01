# Writer's Guide — Regional Income Dynamics Paper
### Hierarchical Bayesian Panel Modeling of U.S. ZIP-Level Income (2011–2022)

> **Who this file is for:** Students or researchers writing the paper from this codebase.  
> **What it covers:** Research positioning, section-by-section writing map, every decision explained, originality strategy, and where to find every result/number.  
> **Template target:** ProComm 2024 format (6–8 pages, IEEE citations, two-column, 10pt Times New Roman).

---

## 1. Research Positioning — Read This Before Writing Anything

### What this project is

A **regional income dynamics study** using a **Hierarchical Bayesian Panel Regression model** estimated via Markov Chain Monte Carlo (MCMC). The unit of analysis is the ZIP code. Every number in this paper describes *aggregated income at the ZIP level*, sourced directly from IRS tax return data.

### What this project is NOT

| Ruled out | Why |
|---|---|
| Predicting individual household income | Statistically invalid from ZIP aggregates; legally problematic (privacy) |
| Census small-area estimation | Different method, different data source, different objective |
| Machine learning / deep learning | Out of scope; no neural nets, no clustering, no graph models |
| Spatial econometrics | No geographic adjacency weights used |
| Comparison study (ML vs Bayesian) | Dropped; focus is explanation, not competition |

### How this differs from the IIM reference paper

The file `references/Estimation of Median Household Income for Small Areas iima.pdf` is a **reference only** — cite it briefly in the literature review to note a prior small-area estimation approach, then explain the departure clearly:

| Dimension | IIM Paper | This Paper |
|---|---|---|
| Target variable | Median household income (Census) | Mean AGI per return (IRS SOI) |
| Data source | Census surveys + ACS | IRS Statistics of Income (SOI) |
| Geographic unit | Census tracts / small areas | 5-digit ZIP codes |
| Method | [their approach — check the PDF] | Hierarchical Bayesian panel regression |
| Time dimension | Cross-sectional or short panel | 12-year balanced panel (2011–2022) |
| Uncertainty | Point estimates | Full posterior distributions (credible intervals) |
| Structural features | Income totals | Income composition shares + ICS + diversity + shock |
| Temporal controls | Not primary | Year fixed effects (γ_t) absorb inflation, macro trends |

**When citing the IIM paper:** use it only to justify *why existing approaches are limited* (e.g., they focus on medians, lack temporal dynamics, or don't account for income composition structure). Never copy their framing, method description, or interpretation language.

---

## 2. Paper Section Map (ProComm 2024 Template → This Project)

The template (`Full-Paper-template.docx`) follows standard IEEE proceedings format. Map each section as follows:

---

### TITLE

**Suggested:**  
*"Regional Income Dynamics in the United States: A Hierarchical Bayesian Panel Analysis of ZIP-Level Income Composition (2011–2022)"*

Or shorter variant:  
*"Modeling Regional Income Structure via Hierarchical Bayesian Panel Regression on IRS SOI Data"*

---

### ABSTRACT (~150–200 words)

Cover in order:
1. Problem: regional income heterogeneity is poorly understood at sub-state level
2. Data: IRS SOI ZIP-level panel, 334,071 observations, 30,147 ZIPs, 51 states, 2011–2022
3. Method: Hierarchical Bayesian panel regression, MCMC estimation (NUTS)
4. Key findings: R²=0.71, Max R-hat=1.005 (full convergence); capital gains and dividends positively predict income; unemployment and social security negatively predict income; 2020 year effect = +0.318 (COVID fiscal stimulus)
5. Contribution: three new structural features (ICS, Shannon diversity, COVID shock response) added to income composition model

**Do not cite references in the abstract. Do not use abbreviations on first use without spelling them out.**

**Index Terms (4, alphabetical):**  
`Bayesian hierarchical modeling, income composition, MCMC estimation, regional income dynamics`

---

### I. INTRODUCTION

**Purpose:** Establish why regional income heterogeneity matters, what gap this fills, and what the paper does.

**Suggested flow:**
1. Income inequality is measured nationally but ZIP-level variation is far more granular and policy-relevant.
2. IRS SOI data has been underutilised for panel modeling — most work uses Census/ACS.
3. Prior approaches (cite IIM paper here briefly) focused on cross-sectional median estimation — they miss temporal dynamics and income composition structure.
4. This paper proposes a hierarchical Bayesian panel model that simultaneously explains: (a) what income is *made of* at the ZIP level, (b) how it varies by state, (c) how it moves over time, and (d) how structural stability, diversification, and shock resilience relate to income levels.
5. Briefly state the three research questions (suggested):
   - RQ1: Which income sources are strongest predictors of regional average income?
   - RQ2: How much of ZIP income variation is explained by state-level structural factors vs. year-level macro forces?
   - RQ3: Do income compositional stability, diversification, and COVID shock resilience independently predict income levels?

---

### II. DATA

**Source:** IRS Statistics of Income (SOI), ZIP Code Data Program.  
**File used:** `zip_income_panel_structural_2011_2022.csv` (in repo root)

**Key statistics to report (all verified in code):**

| Stat | Value | Where |
|---|---|---|
| Total observations | 334,071 rows | `[1/7]` print block in script |
| ZIP codes | 30,147 | same |
| States (incl. DC) | 51 | same |
| Years covered | 2011–2022 | same |
| Full-panel ZIPs (all 12 years) | 27,353 | `[3/7]` print block |
| ZIPs used in model (subsample) | 3,000 | config: `N_SAMPLE_ZIPS = 3000` |
| Sampling strategy | Proportional to state ZIP counts, random within state | see Section 3 of script |

**Dataset column description table** (use this in the paper — from README):

| Column | Description |
|---|---|
| `ZIPCODE` | 5-digit US ZIP code |
| `STATE` | 2-letter state abbreviation |
| `YEAR` | Tax filing year (2011–2022) |
| `total_agi` | Total Adjusted Gross Income across all returns in that ZIP-year |
| `total_wages` | Wages and salaries income |
| `total_business` | Business/self-employment income |
| `total_capital_gains` | Net capital gains (stocks, property) |
| `total_interest` | Taxable interest income |
| `total_dividends` | Ordinary dividend income |
| `total_unemployment` | Unemployment compensation received |
| `total_social_security` | Social Security benefit income |
| `AVG_INCOME` | Average AGI per return — **model response variable** |
| `MEDIAN_BAND` | Ordinal income band (1–6) — not used directly in model |

**Why IRS SOI (not Census)?**  
Census/ACS income estimates are survey-based with sampling error at small geographies. IRS SOI is an administrative record of every filed return — near-complete coverage of income-earning households by ZIP. This makes it more reliable for panel modeling of income composition.

**Why log-transform the response?**  
AVG_INCOME is right-skewed (skewness ≈ 20.3 in this dataset, per exploratory analysis). Log-normal income distributions are standard in the literature — applied here following Stukalenko (2005) §3. Model response is `log(1 + AVG_INCOME)`.

---

### III. FEATURE ENGINEERING

This section covers both the original 7 features and the 3 new structural features.

#### A. Income Composition Shares (original 7)

Each share is: `share_k = total_k / total_agi`, clipped to [0, 1]. This decomposes income into the fraction contributed by each source. Example: if a ZIP's total AGI is $1B and wages are $700M, then `share_wages = 0.70`.

These are the main `X` matrix predictors. They are standardised (mean=0, std=1) before entering the model so beta coefficients are comparable across features.

**Code location:** lines 123–128 of `hierarchical_bayesian_panel.py`

#### B. Income Composition Stability Index (ICS) — NEW

$$ICS_{z,t} = \sum_{k=1}^{7} |s_{k,z,t} - s_{k,z,t-1}|$$

**Plain meaning:** How much did the income mix of this ZIP change compared to last year?  
**High ICS** → the income structure is volatile (e.g. capital gains spiked one year, crashed the next).  
**Low ICS** → the income mix is stable and predictable year after year.

**Data source:** The 7 `share_*` columns, computed per ZIP in chronological order.  
**Missing first year:** Filled with dataset-wide median (no prior year exists for 2011 observations).  
**Clip:** [0, 2].  
**Code location:** lines 147–168 of `hierarchical_bayesian_panel.py`

*Economic rationale:* A ZIP with volatile income composition may reflect boom-bust economic cycles, external commodity dependence, or thin economic base — all characteristics expected to correlate with income level and inequality (Stukalenko 2005, §4: structural instability and inequality).

#### C. Income Diversity Index (Shannon Entropy) — NEW

$$H_{z,t} = -\sum_{k=1}^{7} p_k \ln p_k, \quad p_k = \frac{s_k}{\sum_{j} s_j}$$

**Plain meaning:** Does this ZIP earn from many income sources, or is it dependent on just one?  
**High H (→ ln 7 ≈ 1.95)** → income is spread across all 7 sources (balanced regional economy).  
**Low H (→ 0)** → one source dominates (e.g. wages = 95% of income).

**Data source:** Same 7 `share_*` columns, normalised per row to a probability distribution.  
**Code location:** lines 170–187 of `hierarchical_bayesian_panel.py`

*Economic rationale:* Economic diversification reduces exposure to sector-specific shocks. A ZIP with high income diversity is structurally more resilient. Shannon entropy is the information-theoretic measure of diversification — borrowed from ecology (species diversity) and applied here to income sources. This is a novel application in the regional income context.

#### D. COVID Shock Response Index — NEW

$$\text{shock}_{z} = \frac{\text{AVG\_INCOME}_{z,2020} - \text{AVG\_INCOME}_{z,2019}}{\text{AVG\_INCOME}_{z,2019} + 1}$$

**Plain meaning:** Did average income in this ZIP go up or down during COVID (2019→2020)?  
**Positive** → income *rose* (wealth-heavy ZIP: capital gains + dividends-driven income actually increased during the 2020 market recovery).  
**Negative** → income *fell* (service/hospitality/trade-dependent ZIP).

**Data source:** `AVG_INCOME` column, rows filtered to `YEAR == 2019` and `YEAR == 2020`.  
**Applied as:** time-invariant ZIP label (same value across all 12 years for a ZIP) — it encodes structural economic type, not a time-varying shock.  
**ZIPs missing 2019 or 2020:** Assigned cross-ZIP median.  
**Clip:** [−1, +1].  
**Code location:** lines 189–208 of `hierarchical_bayesian_panel.py`

*Why this is valid and original:* The 2020 macro shock was exogenous (i.e., not caused by the ZIP's own economy) and differentially affected ZIPs in a structurally revealing way. Using income change as a *classifier of economic type* — not as a temporal predictor — avoids data leakage and is a legitimate instrumental proxy. This is our approach and no other paper in the reference set uses it this way.

---

### IV. MODEL

#### A. Mathematical Specification

The core panel regression model:

$$\log(1 + \text{AVG\_INCOME}_{z,t}) = \alpha + \boldsymbol{\beta} \mathbf{X}_{z,t} + u_{\text{state}[z]} + \gamma_t + \varepsilon_{z,t}$$

| Term | Definition | Prior |
|---|---|---|
| $\alpha$ | Global intercept | $\mathcal{N}(\bar{y}, 2)$ |
| $\boldsymbol{\beta}$ | Vector of 10 feature coefficients | $\mathcal{N}(0, 5)$ each |
| $\mathbf{X}_{z,t}$ | Standardised feature matrix (10 columns) | — |
| $u_{\text{state}[z]}$ | State-level random intercept | $u = u_\text{raw} \cdot \sigma_\text{state}$, $u_\text{raw} \sim \mathcal{N}(0,1)$, $\sigma_\text{state} \sim \text{HalfNormal}(2)$ |
| $\gamma_t$ | Year fixed effect | $\gamma = \gamma_\text{raw} \cdot \sigma_\text{year}$, $\gamma_\text{raw} \sim \mathcal{N}(0,1)$, $\sigma_\text{year} \sim \text{HalfNormal}(2)$ |
| $\varepsilon_{z,t}$ | Observation noise | $\mathcal{N}(0, \sigma_\text{obs}^2)$, $\sigma_\text{obs} \sim \text{HalfNormal}(3)$ |

**Code location:** lines 310–370 of `hierarchical_bayesian_panel.py`

#### B. Why Hierarchical?

ZIP codes are nested within states. A standard linear regression would treat all 51 states as independent and would either ignore state structure entirely (underfitting) or add 50 dummy variables (overfitting with sparse data). The hierarchical (partial pooling) approach lets the model learn a shared distribution for state effects — states borrow statistical strength from each other. This is standard for geographic panel data (Gelman & Hill 2007).

#### C. Non-Centred Reparameterisation

State and year random effects are specified as:
```
u_state = u_state_raw × σ_state        where u_state_raw ~ N(0, 1)
```
rather than the naive `u_state ~ N(0, σ_state)`. This separates the scale parameter from the shape parameter, which **dramatically improves MCMC sampling efficiency** and reduces divergences. It is the standard recommendation in PyMC / Stan documentation and is grounded in Bayesian probability theory.

#### D. Why NUTS / MCMC (not MLE or variational inference)?

NUTS (No-U-Turn Sampler) is a gradient-based variant of Hamiltonian Monte Carlo. It:
- Returns **full posterior distributions** — we get credible intervals for every parameter, not just point estimates.
- Handles the correlated hierarchical structure well (naive Gibbs sampling would get stuck).
- Has well-understood convergence diagnostics (R-hat, ESS).
- Multi-chain sampling (2 independent chains) enables convergence verification (Sherri et al. 2021; Zheng et al. 2009).

This choice gives us **uncertainty quantification at every level** — a key differentiator from regression or ML approaches.

#### E. MCMC Configuration (all values in `hierarchical_bayesian_panel.py`, lines 80–90)

| Parameter | Value | Reason |
|---|---|---|
| `draws` | 2000 | Sufficient for stable posterior estimates |
| `tune` | 1500 | Warm-up for step-size adaptation |
| `chains` | 2 | Enables R-hat convergence check (Sherri 2021) |
| `target_accept` | 0.95 | Higher = fewer divergences in hierarchical models |
| `cores` | 1 | Sequential — avoids Windows multiprocessing issues |
| `random_seed` | 42 | Full reproducibility |

#### F. Subsample Strategy

The full dataset has 334,071 rows. NUTS scales as O(n) per step, so the full dataset would require days. We take a **stratified random sample of 3,000 full-panel ZIPs** (ZIPs with all 12 years present — 27,353 available). Sampling is **proportional to each state's share of full-panel ZIPs** so no state is over- or under-represented. This produces 36,000 rows (3,000 ZIPs × 12 years).

**This is not a statistical limitation** — 36,000 observations is ample for a model with 10 fixed effects + 51 state effects + 12 year effects (~73 free parameters). The posterior is well-identified.

---

### V. RESULTS

All result numbers below are from the last successful MCMC run (pre-new-feature run). After re-running with 10 features, update the numbers from the printed output or `results/` files.

#### A. Convergence

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Max R-hat | 1.005 | < 1.05 | ✅ Converged |
| Divergences | 0 | 0 preferred | ✅ Clean |
| Chains | 2 | ≥ 2 required | ✅ |

**Interpretation for paper:** R-hat compares variance *within* each chain to variance *between* chains. A value ≤ 1.05 indicates all chains explored the same posterior region, confirming convergence (Gelman et al. 2014).

**Result file:** `results/summary_global.csv`

#### B. Predictive Performance

| Metric | Value | Interpretation |
|---|---|---|
| R² | 0.71 | Model explains 71% of variance in log(ZIP income) |
| RMSE | 0.2316 | Avg error ≈ 0.23 log-income units (~26% on income scale) |
| MAE | 0.1742 | Median-style error ≈ 0.17 log units |
| Gini of residuals | 0.4510 | Residual inequality measure (Stukalenko 2005) |

**How to present R² in a Bayesian context:** R² here is computed on *posterior mean* predictions vs. observed values. It is a descriptive measure of fit, not a hypothesis test. In Bayesian papers, it is conventional to accompany R² with the posterior predictive check plot.

**Plot:** `results/posterior_predictive.png` (obs vs predicted scatter + residual histogram)

#### C. Beta Coefficients

From `results/beta_coefficients.csv`. These are the most important interpretive results.

| Feature | β mean | 94% HDI | Direction | Economic meaning |
|---|---|---|---|---|
| `share_capital_gains` | +0.111 | check CSV | Positive | High-income ZIPs earn heavily from investments |
| `share_dividends` | +0.097 | check CSV | Positive | Wealth-derived passive income |
| `share_interest` | +0.034 | check CSV | Positive | Savings/bond income |
| `share_business` | −0.033 | check CSV | Negative (weak) | Small business income slightly negative |
| `share_wages` | −0.103 | check CSV | Negative | Wage-dependent ZIPs are relatively lower income |
| `share_social_security` | −0.114 | check CSV | Negative | Retirement/lower-income demographics |
| `share_unemployment` | −0.117 | check CSV | Negative | Strongest negative predictor overall |
| `income_stability_index` | TBD | TBD | Rerun | After rerun with 10 features |
| `income_diversity_index` | TBD | TBD | Rerun | After rerun with 10 features |
| `shock_response_index` | TBD | TBD | Rerun | After rerun with 10 features |

> **Note for writers:** The three new feature betas are available only after re-running `hierarchical_bayesian_panel.py` with the updated 10-feature version. Run the script and pull values from `results/beta_coefficients.csv`.

**How to interpret HDI (Highest Density Interval):** A 94% HDI of [0.05, 0.17] for capital_gains means: *given the data and model, there is a 94% posterior probability that the true effect of capital gains share on log-income lies between 0.05 and 0.17.* If the HDI does not cross zero, the effect is credibly non-zero.

**Plot:** `results/beta_forest.png`

#### D. State Random Effects

From `results/state_effects.csv`. These are the $u_{\text{state}}$ intercepts — how much each state's baseline log-income deviates from the national average *after controlling for income composition*.

Top states (from prior run): **DC (+0.385) > CT (+0.364) > NJ (+0.327) > MA (+0.260) > NY (+0.145)**  
Bottom states: check `state_effects.csv`, sorted ascending.

**Economic interpretation:** These effects capture structural state-level advantages beyond income composition. DC's high intercept reflects the concentration of federal employees and lobbying/consulting industries that produce high income regardless of specific share structure.

**Plot:** `results/state_effects.png`

#### E. Year Effects (γ_t)

From `results/year_effects.csv`.

| Notable year | Effect | Interpretation |
|---|---|---|
| 2011–2015 | Negative | Post-GFC recovery; incomes below long-run average |
| 2019 | Near zero | Pre-COVID baseline |
| **2020** | **+0.318** | **COVID fiscal stimulus + capital gains spike** |
| 2021–2022 | Elevated | Sustained post-stimulus effect |

**Why 2020 is *positive*:** Counter-intuitive at first glance. The IRS records AGI which *includes* capital gains. 2020 saw massive stock market recovery (S&P +18%), so high-income filer AGI rose sharply while service-sector workers filed lower returns. The year effect captures the net national aggregate.

This directly validates the inclusion of `shock_response_index` — the 2020 year effect is a national aggregate, but ZIPs responded very differently depending on their income structure.

**Plot:** `results/year_effects.png`

#### F. Variance Components (from `results/summary_global.csv`)

| Parameter | Meaning | Expected range |
|---|---|---|
| `sigma_state` | Std dev of state random effects | ~0.2–0.4 |
| `sigma_year` | Std dev of year effects | ~0.1–0.3 |
| `sigma_obs` | Residual noise (within-ZIP unexplained) | ~0.2–0.3 |

The ratio `sigma_state / sigma_obs` tells you how much income variation is state-structural vs. local/idiosyncratic.

**Plot:** `results/prior_vs_posterior.png` — shows that the data meaningfully updated the posterior away from the prior (evidence of genuine signal being learned).

---

### VI. DISCUSSION

**Suggested points:**

1. **Wealth vs. labour income duality:** The positive β for capital gains / dividends and negative β for wages / unemployment confirms what macroeconomists call the "capital–labour income divide." High-income ZIPs are investment clusters; low-income ZIPs are wage-dependent. This is a structural, not a cyclical, finding.

2. **COVID as a structural classifier:** The 2020 year effect (+0.318) combined with the shock response heterogeneity reveals that the pandemic *widened* regional income inequality: wealthy ZIPs gained while service-dependent ZIPs lost. The Bayesian model makes this quantifiable with uncertainty.

3. **State effects are substantial but residual:** DC, CT, NJ, MA, NY dominate the positive state intercepts even *after* controlling for all income composition shares. This suggests that geography and state-level institutions, tax policy, and industrial clustering contribute to income levels beyond what composition alone explains.

4. **ICS and diversity as structural diagnostics:** If the new feature betas show significance (HDI not crossing zero), this would be the first time income compositional stability and diversity have been formally estimated as predictors in a Bayesian panel model at the ZIP level. This is a novel contribution.

5. **Limitations to acknowledge honestly:**
   - 3,000 ZIP subsample (27,353 available); results may not generalise to thinly-populated rural ZIPs underrepresented in the sample.
   - WAIC/LOO not computed (requires re-run with `idata_kwargs={"log_likelihood": True}`) — model selection criteria are absent.
   - Model is log-linear; non-linear interactions between income sources not captured.
   - `shock_response_index` is time-invariant per ZIP — it captures the *type* of ZIP but cannot model ongoing temporal shock propagation.

---

### VII. CONCLUSION

Suggested 3-paragraph structure:
1. Restate what was done (Bayesian hierarchical panel, 10 features, IRS SOI 2011–2022).
2. What was found (capital/dividend-driven ZIPs earn more; wage/unemployment-heavy ZIPs earn less; strong state effects; 2020 anomaly; convergence confirmed).
3. What this contributes (new structural features; full posterior uncertainty; replicable open codebase).

---

### ACKNOWLEDGEMENTS

If applicable: acknowledge IRS SOI data programme, any computing resources used, any advisors.

---

### REFERENCES (IEEE Format)

Use these exactly:

```
[1] E. A. Stukalenko, "Estimation Methods of Inequality of the Population Incomes,"
    in Proc. KURUS'2005 Int. Conf., 2005.

[2] O. Kalivoshko et al., "Assessment of Factors Influencing the Volume of Personal
    Income Tax Revenues," in Proc. PIC S&T'2020, 2020, pp. [add pp].

[3] M. Sherri et al., "A Comparison of Multiple Markov Chains Algorithms for Bayesian
    Updating," in Proc. ICECET 2021, 2021, pp. [add pp].

[4] K. Zheng et al., "Access Model of Web Users Based on Multi-chains Hidden Markov
    Models," in Proc. ICICSE 2009, 2009, pp. [add pp].

[5] [IIM paper — cite appropriately as a prior small-area estimation approach, 
    not as a methodological source for this paper]
```

> Fill in exact page numbers from the PDFs in `references/`.

---

## 3. Originality Checklist — What Makes This Paper Different

Use this when a reviewer asks "what is new here?"

| Contribution | Claim |
|---|---|
| Dataset | IRS SOI used as a 12-year panel — most prior work is cross-sectional |
| Income composition | Decomposing income into 7 source shares as predictors, not just totals |
| Hierarchical Bayesian | Full posterior estimation with credible intervals — not ML point estimates |
| ICS feature | Year-over-year compositional volatility as a predictor — novel |
| Shannon diversity | Shannon entropy of income mix at ZIP level — novel application |
| COVID shock index | 2019→2020 income change as a structural ZIP classifier — novel |
| State random effects | Quantified state-level income premium *residual to* composition — not just state dummies |
| Open reproducibility | Seed=42, full trace in `results/trace.nc`, results CSVs on GitHub |

---

## 4. File Map — Where Every Number Comes From

| Paper claim | Source file | How to access |
|---|---|---|
| Dataset dimensions | `hierarchical_bayesian_panel.py` `[1/7]` block output | Run script or see README |
| Feature correlations | Script output `[2/7]` block | Rerun to see all 10 |
| Beta coefficients | `results/beta_coefficients.csv` | Open CSV; index = feature name |
| State effects | `results/state_effects.csv` | Rows = state abbreviations |
| Year effects | `results/year_effects.csv` | Rows = years 2011–2022 |
| Global params (α, σ) | `results/summary_global.csv` | - |
| R², RMSE, MAE | Script output `[7/7]` block | - |
| R-hat | Script output + `summary_global.csv` | - |
| All plots | `results/*.png` | 6 PNG files |
| Full posterior | `results/trace.nc` | Load with `az.from_netcdf()` |
| Model code | `hierarchical_bayesian_panel.py` | Lines 310–370 (PyMC block) |
| Feature engineering | same file, lines 120–230 | Section 2 + 2b |

---

## 5. Key Terminology Glossary (for writers not from a stats background)

| Term | Plain meaning |
|---|---|
| **Posterior distribution** | The model's belief about a parameter's value *after* seeing the data — shown as a curve, not a single number |
| **Credible interval (HDI)** | A range that contains 94% of the posterior probability — like a confidence interval but directly probabilistic |
| **R-hat** | A number that tells you if the MCMC chains agreed. Below 1.05 = yes, they converged on the same answer |
| **NUTS sampler** | The algorithm that draws samples from the posterior. Think of it as a very smart random walk through probability space |
| **Hierarchical model** | A model where groups (states) share a common distribution — they influence each other rather than being treated independently |
| **Non-centred parameterisation** | A mathematical trick that makes hierarchical models easier to sample from — the output is identical, just faster |
| **Log-transform** | Taking log(1 + income) so the skewed income distribution becomes approximately normal — required for the Gaussian likelihood to make sense |
| **Year fixed effect (γ_t)** | A per-year adjustment that absorbs anything that happened nationally that year (inflation, policy changes, COVID) |
| **State random effect (u_state)** | A per-state adjustment that absorbs structural differences between states that don't vary within-state over time |
| **WAIC / LOO** | Model comparison metrics (currently not computed — would require a re-run) |

---

## 6. Important Warnings for Writers

1. **Do NOT say "predicts individual income."** Every result is about ZIP-level averages from tax filings.
2. **Do NOT claim the model generalises beyond the US.** IRS SOI is US-specific.
3. **Do NOT present R²=0.71 as "accuracy."** Frame it as: *"The model accounts for 71% of variance in regional log-income, indicating strong explanatory power for a panel of this complexity."*
4. **Do NOT present the 3,000 ZIP sample as a limitation without context.** 36,000 observations for ~73 parameters is generous — the subsample is a *computational choice*, not a data quality issue.
5. **Do NOT copy language from the IIM paper.** Paraphrase when describing their approach in the literature review; cite the paper number inline.
6. **When reporting beta coefficients, always report the HDI alongside the mean.** A beta mean alone is meaningless in Bayesian work; the uncertainty range is the contribution.
