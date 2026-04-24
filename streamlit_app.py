"""
Streamlit UI for ZIP-level Bayesian income forecasting.

Run:
  streamlit run streamlit_app.py
"""

import pandas as pd
import streamlit as st

from zip_income_predictor import ZipIncomePredictor


st.set_page_config(page_title="ZIP Income Predictor", page_icon="📈", layout="centered")

st.title("ZIP Income Predictor")
st.caption("Predict next-year ZIP income using saved Bayesian posterior artifacts.")

with st.sidebar:
    st.header("Settings")
    results_dir = st.text_input("Results directory", value="results")
    year_effect_mode = st.selectbox(
        "Future year effect mode",
        options=["sample_sigma", "last_year", "zero"],
        index=0,
        help="How to handle year effect for years outside training data.",
    )
    strict_training_zip = st.toggle(
        "Strict training ZIP scope",
        value=False,
        help="When enabled, only ZIPs from training_sampled_zips.csv are allowed.",
    )
    draws = st.slider("Posterior draws", min_value=100, max_value=2000, value=500, step=100)

try:
    predictor = ZipIncomePredictor(
        results_dir=results_dir,
        year_effect_mode=year_effect_mode,
        strict_training_zip=strict_training_zip,
    )
except Exception as exc:
    st.error(
        "Could not load model artifacts. Run hierarchical_bayesian_panel_next_year.py first."
    )
    st.exception(exc)
    st.stop()

if predictor.has_trained_zip_scope:
    st.sidebar.caption(f"Training ZIP scope loaded: {len(predictor.trained_zip_keys):,} ZIPs")
else:
    st.sidebar.caption("Training ZIP scope file not found; ZIP scope status may be unknown.")

zip_input = st.text_input("ZIP code", value="10001", help="Enter a 5-digit ZIP or numeric ZIP key.")
forecast_year = st.number_input("Forecast year", min_value=2011, max_value=2100, value=2023, step=1)

col_a, col_b = st.columns(2)
with col_a:
    run_btn = st.button("Predict", type="primary", use_container_width=True)
with col_b:
    show_sample_btn = st.button("Show Available ZIP Samples", use_container_width=True)

if show_sample_btn:
    sample_df = predictor.latest_zip_features[["ZIPCODE", "STATE", "YEAR"]].head(20).copy()
    st.subheader("Sample ZIPs from artifacts")
    st.dataframe(sample_df, use_container_width=True)

if run_btn:
    try:
        pred = predictor.predict_zip(zipcode=zip_input, forecast_year=int(forecast_year), draws=int(draws))

        st.success("Prediction complete")

        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Pred Income Mean",
                    "Pred Income P05",
                    "Pred Income P50",
                    "Pred Income P95",
                ],
                "Value": [
                    pred["pred_income_mean"],
                    pred["pred_income_p05"],
                    pred["pred_income_p50"],
                    pred["pred_income_p95"],
                ],
            }
        )

        info_cols = st.columns(4)
        info_cols[0].metric("ZIP", pred["zip"])
        info_cols[1].metric("State", pred["state"])
        info_cols[2].metric("Source Year", pred["source_year"])
        info_cols[3].metric("Forecast Year", pred["forecast_year"])

        st.subheader("Income Forecast")
        st.dataframe(summary_df.style.format({"Value": "{:,.2f}"}), use_container_width=True)

        st.caption(
            f"Year effect mode: {pred['year_effect_mode']} | Draws used: {int(draws)}"
        )

        if pred["in_training_sample"] is None:
            st.info("Training subsample scope unknown (training_sampled_zips.csv not found).")
        elif pred["in_training_sample"]:
            st.success("ZIP is in the 3,000 ZIP training subsample.")
        else:
            st.warning("ZIP is outside the 3,000 ZIP training subsample (out-of-sample prediction).")

    except Exception as exc:
        st.error("Prediction failed. Check ZIP format and artifacts.")
        st.exception(exc)
