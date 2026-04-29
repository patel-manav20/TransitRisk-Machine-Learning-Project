"""Tab 2 — What-If Simulator."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import PALETTE, begin_panel, inject_callout, inject_metric_tile, inject_section_title, style_figure


def render_what_if(model, X_test: pd.DataFrame, feature_names: list[str],
                   explainer=None) -> None:
    inject_section_title(
        "Scenario planning",
        "Model the impact of weather and operations changes",
        "Use a representative baseline and simulate worsening or improving conditions to estimate whether risk is likely to cross the operational alert boundary.",
    )

    col_left, col_right = st.columns(2)

    with col_left:
        begin_panel("Scenario controls", "Adjust the highest-leverage environmental and service variables.")
        precip     = st.slider("Precipitation (mm/h)", 0.0, 30.0, 2.0, step=0.5)
        wind       = st.slider("Wind speed (kph)", 0.0, 60.0, 15.0, step=1.0)
        demand_mul = st.slider("Demand multiplier", 0.5, 2.0, 1.0, step=0.1)
        headway    = st.slider("Headway (minutes)", 1.0, 30.0, 10.0, step=0.5)

    # Take a representative baseline row (median of test set)
    X_base = X_test.median().to_frame().T.copy()
    X_sim  = X_base.copy()

    if "mean_precip_mm" in feature_names:
        X_sim["mean_precip_mm"] = precip
        X_sim["precip_lag_1h"] = precip * 0.8
        X_sim["precip_rolling_3h_sum"] = precip * 2.5
    if "mean_wind_kph" in feature_names:
        X_sim["mean_wind_kph"] = wind
        X_sim["wind_lag_1h"] = wind * 0.9
    if "mean_demand_current" in feature_names:
        X_sim["mean_demand_current"] = X_base["mean_demand_current"].values[0] * demand_mul
    if "mean_headway_current" in feature_names:
        X_sim["mean_headway_current"] = headway

    # Interaction features
    if "peak_x_precip" in feature_names:
        is_peak = X_sim.get("is_peak_morning", pd.Series([0])).values[0] or X_sim.get("is_peak_evening", pd.Series([0])).values[0]
        X_sim["peak_x_precip"] = int(is_peak) * precip
    if "weekend_x_precip" in feature_names:
        is_we = X_sim.get("is_weekend", pd.Series([0])).values[0]
        X_sim["weekend_x_precip"] = int(is_we) * precip

    p_base = float(model.predict_proba(X_base[feature_names])[:, 1][0])
    p_sim  = float(model.predict_proba(X_sim[feature_names])[:, 1][0])
    delta = p_sim - p_base

    with col_right:
        begin_panel("Scenario outcome", "Compare the simulated state against the representative baseline from the test distribution.")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            inject_metric_tile("Baseline risk", f"{p_base:.1%}", "Median test-set operating state")
        with cc2:
            inject_metric_tile("Simulated risk", f"{p_sim:.1%}", "Predicted risk under current slider settings")
        with cc3:
            inject_metric_tile("Delta", f"{delta:+.1%}", "Change from baseline probability")

        # Conformal badge
        if p_sim > 0.7:
            badge = "Scenario likely triggers a high-risk state."
            tone = "danger"
        elif p_sim < 0.3:
            badge = "Scenario remains comfortably below alert territory."
            tone = "success"
        else:
            badge = "Scenario moves into a watchlist zone and deserves operator review."
            tone = "warning"
        inject_callout("Operational interpretation", badge, tone=tone)

    # SHAP contributions (top-3) if explainer available
    if explainer is not None:
        begin_panel("Top contributing drivers", "Explainability is used when available; otherwise the app falls back to feature deltas.")
        try:
            sv = explainer.shap_values(X_sim[feature_names])
            sv_arr = sv if not isinstance(sv, list) else sv[1]
            importance = pd.Series(sv_arr[0], index=feature_names)
            top3 = importance.abs().nlargest(3)

            fig = go.Figure(go.Bar(
                x=importance[top3.index].values,
                y=top3.index.tolist(),
                orientation="h",
                marker_color=[PALETTE.danger if v > 0 else PALETTE.accent for v in importance[top3.index].values],
            ))
            style_figure(fig, height=200)
            fig.update_layout(xaxis_title="SHAP contribution")
            st.plotly_chart(fig, use_container_width=True, key="wi_shap")
        except Exception:
            st.info("SHAP unavailable — run notebook 09 first.")
    else:
        # Fallback: show feature deltas
        begin_panel("Feature movement", "This fallback view shows which inputs changed most from the representative baseline.")
        changes = (X_sim[feature_names] - X_base[feature_names]).T
        changes.columns = ["delta"]
        changes = changes[changes["delta"].abs() > 0].sort_values("delta", ascending=False)
        if len(changes) > 0:
            fig = go.Figure(go.Bar(
                x=changes["delta"].values,
                y=changes.index.tolist(),
                orientation="h",
                marker_color=[PALETTE.danger if v > 0 else PALETTE.accent for v in changes["delta"].values],
            ))
            style_figure(fig, height=225)
            fig.update_layout(xaxis_title="Delta from baseline")
            st.plotly_chart(fig, use_container_width=True, key="wi_delta")
