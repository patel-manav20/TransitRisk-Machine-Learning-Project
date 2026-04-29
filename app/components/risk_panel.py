"""Tab 1 — Risk Panel.

Shows per-station risk across all routes it actually serves.
Cascading dropdowns: pick a station → only valid routes appear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import (
    PALETTE,
    begin_panel,
    inject_callout,
    inject_metric_grid,
    inject_section_title,
    inject_status_badge,
    style_figure,
)


def _badge_html(prob: float, t_cost: float) -> tuple[str, str]:
    if prob >= 0.7:
        return "High risk", PALETTE.danger
    if prob >= t_cost:
        return "Elevated", PALETTE.warning
    if prob >= 0.3:
        return "Watchlist", PALETTE.warning
    return "Low risk", PALETTE.success


def render_risk_panel(
    modeling_df: pd.DataFrame,
    X_features: pd.DataFrame,
    model,
    prediction_sets_df: pd.DataFrame,
    test_idx: list[int],
) -> None:
    inject_section_title(
        "Overview",
        "Current network risk by station and route",
        "Inspect the latest predicted disruption probability, validate route-level confidence, and compare current conditions before diving deeper.",
    )

    # Build station → routes mapping from ALL data (not just test)
    station_routes = (
        modeling_df.groupby("station_id")["route_id"]
        .unique()
        .apply(sorted)
        .to_dict()
    )
    all_stations = sorted(station_routes.keys())

    filters_left, filters_right, filters_note = st.columns([1.2, 1.2, 1.6])
    with filters_left:
        station = st.selectbox(
            "Station",
            all_stations,
            format_func=lambda s: f"Station {s}",
            key="rp_station",
        )

    valid_routes = station_routes.get(station, [])

    with filters_right:
        route = st.selectbox(
            "Route",
            valid_routes,
            format_func=lambda r: f"Route {r}",
            key="rp_route",
        )

    with filters_note:
        inject_callout(
            "Route availability",
            "Routes are filtered to only the combinations that actually serve the selected station, keeping the current-state view operationally valid.",
        )

    # ── Filter test set to this station-route pair ─────────────────────────────
    test_modeling = modeling_df.iloc[test_idx].reset_index(drop=True)
    test_X        = X_features.iloc[test_idx].reset_index(drop=True)

    mask_sr = (test_modeling["station_id"] == station) & (test_modeling["route_id"] == route)
    df_sr   = test_modeling[mask_sr]
    X_sr    = test_X[mask_sr]

    if len(X_sr) == 0:
        st.warning(f"No test data for Station {station} · Route {route}.")
        return

    # Run model
    proba      = model.predict_proba(X_sr)[:, 1]
    latest_p   = float(proba[-1])
    last_row   = df_sr.iloc[-1]

    t_cost = 0.163   # fixed cost-optimal threshold

    # ── Conformal badge ────────────────────────────────────────────────────────
    # prediction_sets_df is indexed over the full test set in order
    test_indices_sr = np.where(mask_sr.values)[0]
    last_test_i     = int(test_indices_sr[-1])
    if prediction_sets_df is not None and last_test_i < len(prediction_sets_df):
        ps = prediction_sets_df.iloc[last_test_i]
        if ps["pred_1"] and not ps["pred_0"]:
            conf_label = "🔴  90% confident: **ELEVATED RISK**"
            conf_color = "#ef4444"
        elif ps["pred_0"] and not ps["pred_1"]:
            conf_label = "🟢  90% confident: **LOW RISK**"
            conf_color = "#22c55e"
        else:
            conf_label = "🟡  **Uncertain** — both classes possible"
            conf_color = "#eab308"
    else:
        conf_label = "⚪  Conformal set unavailable"
        conf_color = "#6b7280"

    inject_metric_grid([
        ("Current probability", f"{latest_p:.1%}", "Predicted next-hour elevated delay risk"),
        ("Avg delay now", f"{last_row.get('mean_delay_current', 0):.1f} min", "Observed current-state delay"),
        ("Demand", f"{last_row.get('mean_demand_current', 0):.0f} pax/hr", "Latest passenger demand signal"),
        ("Headway", f"{last_row.get('mean_headway_current', 0):.1f} min", "Service spacing at the selected route"),
    ], columns=4)

    # ── Layout ─────────────────────────────────────────────────────────────────
    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        begin_panel("Latest risk score", "The model output is calibrated, then compared against the cost-sensitive dispatch threshold.")
        label, color = _badge_html(latest_p, t_cost)
        tone = "danger" if latest_p >= 0.7 else "warning" if latest_p >= t_cost else "success"
        inject_status_badge(label, tone=tone, meta=f"Dispatch threshold: {t_cost:.3f}")
        st.markdown(f"<div style='font-size:0.9rem;font-weight:600;color:{conf_color};margin-bottom:12px'>{conf_label}</div>", unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge",
            value=round(latest_p * 100, 1),
            domain={"x": [0.06, 0.94], "y": [0.18, 0.98]},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar":  {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],   "color": "rgba(47,191,113,0.16)"},
                    {"range": [30, 50],  "color": "rgba(244,184,96,0.13)"},
                    {"range": [50, 70],  "color": "rgba(244,184,96,0.20)"},
                    {"range": [70, 100], "color": "rgba(255,107,107,0.18)"},
                ],
                "threshold": {
                    "line": {"color": PALETTE.purple, "width": 3},
                    "thickness": 0.75,
                    "value": t_cost * 100,
                },
            },
            title={"text": "P(elevated delay risk — next hour)", "font": {"size": 13}},
        ))
        style_figure(fig_gauge, height=320)
        fig_gauge.update_layout(
            margin=dict(t=52, b=26, l=34, r=34),
            annotations=[
                dict(
                    x=0.5,
                    y=0.15,
                    xref="paper",
                    yref="paper",
                    text=f"<span style='font-size:38px;font-weight:800;color:{color}'>{latest_p:.1%}</span>",
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                )
            ],
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key="rp_gauge")

    with col_info:
        begin_panel("Current operating context", "These conditions are the primary drivers behind the selected route’s risk score.")
        metrics = [
            ("Current avg delay", f"{last_row.get('mean_delay_current', 0):.1f} min"),
            ("Share delayed >5 min", f"{last_row.get('share_delayed_5_current', 0):.1%}"),
            ("Passenger demand", f"{last_row.get('mean_demand_current', 0):.0f} pax/hr"),
            ("Headway", f"{last_row.get('mean_headway_current', 0):.1f} min"),
            ("Precipitation", f"{last_row.get('mean_precip_mm', 0):.1f} mm/hr"),
            ("Wind", f"{last_row.get('mean_wind_kph', 0):.0f} kph"),
            ("Incident flag", "Yes" if last_row.get("incident_flag") else "No"),
            ("Vehicle type", str(last_row.get("vehicle_type", "—"))),
        ]
        context_cards = "".join(
            f'<div class="tr-context-item"><div class="tr-context-label">{label}</div><div class="tr-context-value">{value}</div></div>'
            for label, value in metrics
        )
        st.markdown(f'<div class="tr-context-grid">{context_cards}</div>', unsafe_allow_html=True)

    inject_callout(
        "Operational readout",
        "Use the latest score to decide whether this station-route pair needs dispatch attention now, then compare nearby route exposure to understand local network stress.",
    )

    # ── All-routes comparison for this station ─────────────────────────────────
    begin_panel(f"All routes at Station {station}", "Compare the latest risk snapshot across every route serving the selected station.")

    route_probs = {}
    for r in valid_routes:
        m = (test_modeling["station_id"] == station) & (test_modeling["route_id"] == r)
        if m.sum() == 0:
            continue
        X_r = test_X[m]
        p_r = model.predict_proba(X_r)[:, 1]
        route_probs[r] = float(p_r[-1])

    colors_bar = [
        PALETTE.danger if v >= 0.7 else PALETTE.warning if v >= t_cost else "#d7a85d" if v >= 0.3 else PALETTE.success
        for v in route_probs.values()
    ]
    fig_bar = go.Figure(go.Bar(
        x=list(route_probs.keys()),
        y=list(route_probs.values()),
        marker_color=colors_bar,
        text=[f"{v:.0%}" for v in route_probs.values()],
        textposition="outside",
    ))
    fig_bar.add_hline(
        y=t_cost, line_dash="dash", line_color=PALETTE.purple, line_width=2,
        annotation_text=f"Alert threshold (t={t_cost})",
        annotation_position="right",
    )
    style_figure(fig_bar, height=235)
    fig_bar.update_layout(
        xaxis_title="Route",
        yaxis_title="Risk probability",
        yaxis_range=[0, 1.1],
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True, key="rp_routes_bar")

    # ── 24h history for selected route ─────────────────────────────────────────
    begin_panel(f"24-hour risk history · Station {station} / Route {route}", "Review trajectory and spot whether the current alert is part of a broader pattern.")
    hours_24 = df_sr["hour_floor"].values[-24:]
    proba_24  = proba[-24:]
    truth_24  = df_sr["y_primary"].values[-24:]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hours_24, y=proba_24,
        mode="lines+markers",
        name="P(risk)",
        line=dict(color=PALETTE.accent, width=3),
        marker=dict(
            size=8,
            color=[PALETTE.danger if p >= t_cost else PALETTE.success for p in proba_24],
            line=dict(width=1, color=PALETTE.bg),
        ),
        hovertemplate="Hour: %{x}<br>P(risk)=%{y:.2f}<extra></extra>",
    ))
    # Actual label markers
    elev_mask = truth_24 == 1
    fig_hist.add_trace(go.Scatter(
        x=hours_24[elev_mask], y=np.ones(elev_mask.sum()) * -0.04,
        mode="markers",
        name="Actual elevated",
        marker=dict(symbol="triangle-up", size=8, color=PALETTE.danger),
        hoverinfo="skip",
    ))
    fig_hist.add_hline(
        y=t_cost, line_dash="dash", line_color=PALETTE.purple, line_width=1.5,
        annotation_text=f"t={t_cost}", annotation_position="right",
    )
    style_figure(fig_hist, height=250)
    fig_hist.update_layout(
        xaxis_title="Hour",
        yaxis_title="Risk probability",
        yaxis_range=[-0.1, 1.05],
        legend=dict(orientation="h", y=1.05, font=dict(size=10)),
    )
    st.plotly_chart(fig_hist, use_container_width=True, key="rp_history")

    # Test-set accuracy for this combo
    correct = ((proba >= t_cost).astype(int) == df_sr["y_primary"].values).mean()
    pos_rate = df_sr["y_primary"].mean()
    n_alerts  = (proba >= t_cost).sum()
    st.caption(
        f"Test set — {len(df_sr)} hours · {pos_rate:.1%} elevated · "
        f"{n_alerts} alerts issued · {correct:.1%} correct decisions at t={t_cost}"
    )
