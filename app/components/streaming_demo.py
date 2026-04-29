"""Tab 6 — Live Inference Feed.

Simulates what a transit operations centre sees every hour:
  raw transit data → 38 features → XGBoost → P(elevated risk next hour)
  → cost-optimal threshold → DISPATCH ALERT / HOLD decision
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import PALETTE, begin_panel, inject_callout, inject_status_badge, inject_section_title, style_figure

_HIGHLIGHT_FEATURES = [
    "mean_delay_current",
    "lag_1h_mean_delay",
    "lag_3h_mean_delay",
    "share_delayed_5_current",
    "mean_precip_mm",
    "mean_demand_current",
    "mean_headway_current",
    "rolling_6h_mean_delay",
    "is_peak_morning",
    "is_peak_evening",
]


def _badge(prob: float, t_cost: float) -> tuple[str, str]:
    if prob >= 0.7:
        return "High risk", PALETTE.danger
    if prob >= t_cost:
        return "Elevated", PALETTE.warning
    if prob >= 0.3:
        return "Watchlist", PALETTE.warning
    return "Low risk", PALETTE.success


def _decision_badge(prob: float, t_cost: float) -> tuple[str, str]:
    if prob >= t_cost:
        return "Dispatch alert", PALETTE.danger
    return "Hold", PALETTE.success


def _tone_for_prob(prob: float, t_cost: float) -> str:
    if prob >= 0.7:
        return "danger"
    if prob >= t_cost:
        return "warning"
    return "success"


def _gauge(prob: float, t_cost: float, key_suffix: str = "") -> go.Figure:
    label, color = _badge(prob, t_cost)
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=round(prob * 100, 1),
        domain={"x": [0.06, 0.94], "y": [0.18, 0.98]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],   "color": "rgba(47,191,113,0.12)"},
                {"range": [30, 50],  "color": "rgba(244,184,96,0.10)"},
                {"range": [50, 70],  "color": "rgba(244,184,96,0.18)"},
                {"range": [70, 100], "color": "rgba(255,107,107,0.16)"},
            ],
            "threshold": {
                "line": {"color": PALETTE.purple, "width": 3},
                "thickness": 0.8,
                "value": t_cost * 100,
            },
        },
        title={"text": "P(elevated risk — next hour)", "font": {"size": 13}},
    ))
    style_figure(fig, height=305)
    fig.update_layout(
        margin=dict(t=52, b=24, l=34, r=34),
        annotations=[
            dict(
                x=0.5,
                y=0.15,
                xref="paper",
                yref="paper",
                text=f"<span style='font-size:36px;font-weight:800;color:{color}'>{prob:.0%}</span>",
                showarrow=False,
                xanchor="center",
                yanchor="middle",
            )
        ],
    )
    return fig


def _feature_bar(row: pd.Series, highlight_cols: list[str]) -> go.Figure:
    cols = [c for c in highlight_cols if c in row.index]
    vals = row[cols]
    normed = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

    fig = go.Figure(go.Bar(
        x=normed.values,
        y=cols,
        orientation="h",
        marker_color=PALETTE.accent,
        text=[f"{v:.2f}" for v in vals.values],
        textposition="outside",
    ))
    style_figure(fig, height=225)
    fig.update_layout(xaxis=dict(range=[0, 1.4], showticklabels=False, showgrid=False), yaxis=dict(tickfont=dict(size=10)))
    return fig


def _running_metrics_chart(log_df: pd.DataFrame, t_cost: float) -> go.Figure:
    preds  = (log_df["prob"] >= t_cost).astype(int)
    truths = log_df["y_true"].astype(int)
    rows = []
    for i in range(1, len(log_df) + 1):
        p = preds.iloc[:i]; t = truths.iloc[:i]
        tp = ((p == 1) & (t == 1)).sum()
        fp = ((p == 1) & (t == 0)).sum()
        fn = ((p == 0) & (t == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        rows.append({"n": i, "Precision": prec, "Recall": rec, "F1": f1})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for col, color in [("Precision", PALETTE.accent), ("Recall", PALETTE.success), ("F1", PALETTE.purple)]:
        fig.add_trace(go.Scatter(
            x=df["n"], y=df[col], mode="lines",
            name=col, line=dict(color=color, width=2),
        ))
    fig.add_hline(y=0.5, line_dash="dot", line_color=PALETTE.muted, line_width=1)
    style_figure(fig, height=215)
    fig.update_layout(xaxis_title="Predictions so far", yaxis_title="Score", yaxis_range=[0, 1], legend=dict(orientation="h", y=1.12, font=dict(size=10)))
    return fig


def _format_log(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["station_id", "route_id", "hour_floor", "prob", "decision",
              "y_true", "correct"]].copy()
    out = out.rename(columns={
        "station_id": "Station", "route_id": "Route",
        "hour_floor": "Hour", "prob": "P(risk)",
        "decision": "Decision", "y_true": "Actual", "correct": "✓",
    })
    out["P(risk)"] = out["P(risk)"].map(lambda x: f"{x:.3f}")
    out["Actual"]  = out["Actual"].map({1: "Elevated", 0: "Low"})
    out["✓"]       = out["✓"].map({1: "✓", 0: "✗"})
    return out.iloc[::-1].reset_index(drop=True)


def render_streaming_demo(
    model,
    modeling_test: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    t_cost: float,
    feature_names: list[str],
) -> None:
    inject_section_title(
        "Live replay",
        "Experience the model like an operations control room",
        "Replay held-out predictions as if hourly data were arriving live, and watch the system issue operational guidance with continuously updating metrics.",
    )

    with st.expander("ℹ️  What is this? How does the model help? (click to read)", expanded=False):
        st.markdown("""
**The prediction problem:**  
Every hour, for every station-route pair, we ask *"Will the next hour have elevated delays?"*
Elevated = mean delay ≥ 5 min. This is predicted **one hour in advance** so action can be taken.

**The pipeline each hour:**
```
Current-hour transit data (trips, delays, demand, weather)
       ↓
38 engineered features (lag delays, rolling averages, weather interactions, time signals)
       ↓
XGBoost Calibrated  →  P(elevated delay risk)  ← raw probability output
       ↓
Cost threshold  t = 0.163   (FN costs 5× more than FP)
       ↓
⚡ DISPATCH ALERT  /  ✅ HOLD
```

**Why this threshold?**  
Missing a real delay (FN) → passengers stranded, no warning → costs 5 units.  
Unnecessary alert (FP) → one bus repositioned → costs 1 unit.  
So we accept more FPs to ensure we catch almost all real delays.

**What the simulation shows:**  
Replays the held-out test set row-by-row as if data is arriving live.  
Each row = one station-route-hour observation. The model scores it in real time.
""")
    inject_callout(
        "Demo mode",
        "This is the closest thing to an operator-facing product moment in the app: a live feed, model score, recommendation, and running performance telemetry.",
    )
    begin_panel("Streaming simulator", "Configure replay speed, sample size, and whether to expose feature inputs during the demo.")

    ctrl1, ctrl2, ctrl3 = st.columns([1.35, 1.35, 0.9], gap="large")
    with ctrl1:
        speed = st.select_slider(
            "Replay speed",
            options=["Slow (1/s)", "Normal (3/s)", "Fast (10/s)", "Instant"],
            value="Normal (3/s)",
        )
    with ctrl2:
        n_show = st.slider("Rows to replay", min_value=20, max_value=300, value=60, step=10)
    with ctrl3:
        st.markdown("")
        show_features = st.toggle("Show feature inputs", value=True)

    speed_map = {"Slow (1/s)": 1.0, "Normal (3/s)": 0.33, "Fast (10/s)": 0.10, "Instant": 0.0}
    delay = speed_map[speed]

    action_col, _ = st.columns([0.24, 0.76], gap="large")
    with action_col:
        run = st.button("▶  Run simulation", type="primary", use_container_width=True)
    st.markdown("---")

    if "stream_log" not in st.session_state:
        st.session_state["stream_log"] = pd.DataFrame()

    # ── All placeholders declared once, named uniquely ─────────────────────────
    col_gauge, col_feat = st.columns([1.08, 1], gap="large")
    with col_gauge:
        ph_gauge = st.empty()
    with col_feat:
        ph_features = st.empty()

    row1_left, row1_right = st.columns([1, 1], gap="large")
    with row1_left:
        ph_decision = st.empty()
        ph_meta     = st.empty()
    with row1_right:
        ph_conf = st.empty()

    st.markdown("#### Running Prediction Log")
    ph_log = st.empty()
    st.markdown("#### Live Metrics (Precision / Recall / F1)")
    ph_metrics = st.empty()

    # ── Show last run if not running ───────────────────────────────────────────
    if not run:
        log = st.session_state["stream_log"]
        if not log.empty:
            last = log.iloc[-1]
            ph_gauge.plotly_chart(_gauge(last["prob"], t_cost),
                                  use_container_width=True, key="sd_gauge_static")
            dec_label, _ = _decision_badge(last["prob"], t_cost)
            risk_label, _ = _badge(last["prob"], t_cost)
            ph_decision.markdown("")
            with ph_decision.container():
                inject_status_badge(dec_label, tone=_tone_for_prob(last["prob"], t_cost))
            ph_meta.caption(
                f"Last: Station {last['station_id']} · Route {last['route_id']} · "
                f"{str(last['hour_floor'])[:16]}"
            )
            ph_conf.markdown("")
            with ph_conf.container():
                inject_status_badge(risk_label, tone=_tone_for_prob(last["prob"], t_cost))
            if show_features:
                feat_row = X_test.iloc[int(last["test_row_idx"])]
                ph_features.plotly_chart(
                    _feature_bar(feat_row, _HIGHLIGHT_FEATURES),
                    use_container_width=True, key="sd_feat_static",
                )
            ph_log.dataframe(_format_log(log.tail(50)), use_container_width=True, height=260)
            ph_metrics.plotly_chart(
                _running_metrics_chart(log, t_cost),
                use_container_width=True, key="sd_metrics_static",
            )
        else:
            ph_gauge.plotly_chart(_gauge(0.0, t_cost),
                                  use_container_width=True, key="sd_gauge_empty")
            ph_features.markdown(
                "_Feature inputs appear here when the toggle is enabled and the replay begins._"
                if show_features else
                "_Enable `Show feature inputs` to inspect the model inputs for each replayed row._"
            )
            ph_decision.markdown("*Press ▶ Run simulation to start the live feed.*")
        return

    # ── LIVE simulation ────────────────────────────────────────────────────────
    st.session_state["stream_log"] = pd.DataFrame()
    n_total = len(X_test)
    indices = np.linspace(0, n_total - 1, min(n_show, n_total), dtype=int)
    log_rows: list[dict] = []

    for i, idx in enumerate(indices):
        x_row  = X_test.iloc[[idx]]
        prob   = float(model.predict_proba(x_row[feature_names])[:, 1][0])
        y_true = int(y_test[idx])
        meta   = modeling_test.iloc[idx]
        station = meta.get("station_id", "—")
        route   = meta.get("route_id", "—")
        hour_fl = meta.get("hour_floor", "—")

        risk_label, _ = _badge(prob, t_cost)
        dec_label, _ = _decision_badge(prob, t_cost)
        correct = int((prob >= t_cost) == bool(y_true))

        log_rows.append({
            "test_row_idx": idx,
            "station_id": station, "route_id": route, "hour_floor": hour_fl,
            "prob": prob, "y_true": y_true,
            "decision": "ALERT" if prob >= t_cost else "HOLD",
            "correct": correct,
        })
        log_df = pd.DataFrame(log_rows)

        # Update all placeholders with unique keys per iteration
        ph_gauge.plotly_chart(
            _gauge(prob, t_cost),
            use_container_width=True, key=f"sd_gauge_{i}",
        )
        if show_features:
            ph_features.plotly_chart(
                _feature_bar(X_test.iloc[idx], _HIGHLIGHT_FEATURES),
                use_container_width=True, key=f"sd_feat_{i}",
            )
        truth_str   = "✓ correct" if correct else "✗ wrong"
        truth_color = PALETTE.success if correct else PALETTE.danger
        ph_decision.markdown("")
        with ph_decision.container():
            inject_status_badge(dec_label, tone=_tone_for_prob(prob, t_cost))
        ph_meta.markdown(
            f"<span style='color:{PALETTE.text_secondary};font-size:0.86rem;'>Station <b>{station}</b> · Route <b>{route}</b> · {str(hour_fl)[:16]} <span style='color:{truth_color};font-weight:700;'> {truth_str}</span></span>",
            unsafe_allow_html=True,
        )
        ph_conf.markdown("")
        with ph_conf.container():
            inject_status_badge(risk_label, tone=_tone_for_prob(prob, t_cost), meta=f"Ground truth: {'Elevated' if y_true else 'Low'}")
        ph_log.dataframe(_format_log(log_df.tail(50)), use_container_width=True, height=260)
        if len(log_df) >= 3:
            ph_metrics.plotly_chart(
                _running_metrics_chart(log_df, t_cost),
                use_container_width=True, key=f"sd_metrics_{i}",
            )
        if delay > 0:
            time.sleep(delay)

    st.session_state["stream_log"] = log_df

    # Final summary
    preds_arr = (log_df["prob"].values >= t_cost).astype(int)
    truth_arr = log_df["y_true"].values
    acc  = (preds_arr == truth_arr).mean()
    tp   = ((preds_arr == 1) & (truth_arr == 1)).sum()
    fp   = ((preds_arr == 1) & (truth_arr == 0)).sum()
    fn   = ((preds_arr == 0) & (truth_arr == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    st.markdown("---")
    st.markdown("#### Simulation Complete")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc:.1%}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall",    f"{rec:.3f}")
    c4.metric("F1",        f"{f1:.3f}")
    st.caption(
        f"Replayed {len(log_df)} predictions · cost threshold t={t_cost:.3f} · "
        f"XGBoost Calibrated (ROC-AUC 0.809 on full test set)."
    )
