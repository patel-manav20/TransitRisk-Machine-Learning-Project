"""Tab 3 — Cost-Threshold Tuner."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from components.ui import PALETTE, begin_panel, inject_callout, inject_section_title, style_figure


def render_cost_tuner(model, X_test: pd.DataFrame, y_test: np.ndarray,
                      t_f1: float = 0.42, t_cost_default: float = 0.31) -> None:
    inject_section_title(
        "Decision strategy",
        "Tune the alert threshold against operational cost",
        "Explore how the cost of missing a real delay versus issuing a false alarm changes the optimal policy threshold and confusion-matrix trade-offs.",
    )

    begin_panel("Cost assumptions", "Use the slider to model how punitive missed events are relative to unnecessary dispatches.")
    cost_ratio = st.slider(
        "C(FN) / C(FP) — cost of missing a delay vs false alarm",
        min_value=1.0, max_value=10.0, value=5.0, step=0.5
    )

    proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold for selected ratio
    thresholds = np.linspace(0.01, 0.99, 300)
    costs = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        costs.append((cost_ratio * fn + 1 * fp) / len(y_test))

    t_star = float(thresholds[np.argmin(costs)])
    min_cost = float(np.min(costs))

    col1, col2, col3 = st.columns(3)
    col1.metric("Cost-optimal threshold", f"{t_star:.3f}")
    col2.metric("F1-optimal threshold",   f"{t_f1:.3f}")
    col3.metric("Expected cost/pred",      f"{min_cost:.4f}")

    inject_callout(
        "Threshold recommendation",
        f"At a miss-to-false-alarm ratio of {cost_ratio:.1f}:1, the optimal threshold shifts to "
        f"{t_star:.3f}. This can differ materially from both the F1-focused threshold ({t_f1:.3f}) "
        f"and the standard 0.500 default.",
    )

    # Cost curve plot
    begin_panel("Expected cost curve", "The minimum point is the best operating policy under the selected cost ratio.")
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=thresholds, y=costs, mode="lines",
                                   line=dict(color=PALETTE.accent, width=3),
                                   name="Expected cost"))
    fig_curve.add_vline(x=t_star, line_dash="dash", line_color=PALETTE.warning,
                        annotation_text=f"t*={t_star:.2f}", annotation_position="top right")
    fig_curve.add_vline(x=t_f1,   line_dash="dot",  line_color=PALETTE.success,
                        annotation_text=f"t_F1={t_f1:.2f}", annotation_position="top left")
    fig_curve.add_vline(x=0.5,    line_dash="dot",  line_color=PALETTE.muted,
                        annotation_text="0.50", annotation_position="bottom right")
    style_figure(fig_curve, height=255)
    fig_curve.update_layout(xaxis_title="Threshold", yaxis_title="Expected cost per prediction", showlegend=False)
    st.plotly_chart(fig_curve, use_container_width=True, key="ct_curve")

    # Confusion matrix at cost-optimal threshold
    begin_panel("Resulting confusion matrix", "This view shows what the recommended threshold means for alerts, misses, and false positives.")
    y_pred_cost = (proba >= t_star).astype(int)
    cm = confusion_matrix(y_test, y_pred_cost)

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Pred: Low", "Pred: High"],
        y=["True: Low", "True: High"],
        text=cm, texttemplate="%{text}",
        colorscale=[[0, "rgba(110,168,254,0.08)"], [1, PALETTE.accent]],
        showscale=False,
    ))
    style_figure(fig_cm, height=240)
    fig_cm.update_layout(margin=dict(t=10, b=40, l=80, r=20))
    st.plotly_chart(fig_cm, use_container_width=True, key="ct_cm")

    tn, fp_n, fn_n, tp_n = cm.ravel()
    cols = st.columns(4)
    cols[0].metric("True Pos",  tp_n)
    cols[1].metric("True Neg",  tn)
    cols[2].metric("False Pos", fp_n)
    cols[3].metric("False Neg", fn_n, delta=f"Cost×{cost_ratio:.0f}",
                   delta_color="inverse")
