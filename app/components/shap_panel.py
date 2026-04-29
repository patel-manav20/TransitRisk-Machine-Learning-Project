"""Tab 5 — SHAP Explanation Panel."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.ui import PALETTE, begin_panel, inject_callout, inject_section_title, style_figure


def render_shap_panel(model, X_test: pd.DataFrame, y_test: np.ndarray,
                      feature_names: list[str], explainer=None) -> None:
    inject_section_title(
        "Explainability",
        "Open up an individual prediction",
        "Inspect a single held-out observation to understand why the model leaned toward elevated risk, and whether the prediction ultimately matched reality.",
    )

    n_test = len(X_test)
    row_id = st.number_input("Test row ID", min_value=0, max_value=n_test - 1,
                              value=0, step=1)

    X_row = X_test.iloc[[row_id]]
    proba = float(model.predict_proba(X_row)[:, 1][0])
    true_label = int(y_test[row_id])

    col1, col2 = st.columns(2)
    col1.metric("P(elevated risk)", f"{proba:.3f}")
    col2.metric("True label", "Elevated" if true_label == 1 else "Low risk")

    outcome_label = ""
    if true_label == 1 and proba > 0.5:
        outcome_label = "✅ True Positive"
    elif true_label == 0 and proba <= 0.5:
        outcome_label = "✅ True Negative"
    elif true_label == 1 and proba <= 0.5:
        outcome_label = "❌ False Negative"
    else:
        outcome_label = "⚠️ False Positive"
    inject_callout("Classification result", outcome_label)

    if explainer is not None:
        begin_panel("Top SHAP contributors", "Positive contributions push the prediction toward elevated risk; negative values pull it down.")
        try:
            sv = explainer.shap_values(X_row[feature_names])
            sv_arr = sv if not isinstance(sv, list) else sv[1]
            importance = pd.Series(sv_arr[0], index=feature_names)
            top5 = importance.abs().nlargest(5)
            top5_vals = importance[top5.index]

            colors = ["#dc2626" if v > 0 else "#2563eb" for v in top5_vals.values]
            fig = go.Figure(go.Bar(
                x=top5_vals.values,
                y=top5_vals.index.tolist(),
                orientation="h",
                marker_color=[PALETTE.danger if v > 0 else PALETTE.accent for v in top5_vals.values],
                text=[f"{v:+.4f}" for v in top5_vals.values],
                textposition="outside",
            ))
            fig.add_vline(x=0, line_color=PALETTE.muted, line_width=1)
            style_figure(fig, height=260)
            fig.update_layout(xaxis_title="SHAP contribution (positive increases risk)")
            st.plotly_chart(fig, use_container_width=True, key="shap_bar")

            # Feature values vs population
            begin_panel("Feature values vs population", "Compare the selected observation against the broader held-out distribution.")
            pop_means = X_test[top5.index].mean()
            compare_df = pd.DataFrame({
                "Feature": top5.index,
                "This row": X_row[top5.index].values[0],
                "Population mean": pop_means.values,
                "SHAP": top5_vals.values,
            })
            st.dataframe(compare_df.style.format({
                "This row": "{:.3f}",
                "Population mean": "{:.3f}",
                "SHAP": "{:+.4f}",
            }), use_container_width=True)

        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
            _show_feature_values_fallback(X_row, X_test, feature_names)
    else:
        _show_feature_values_fallback(X_row, X_test, feature_names)


def _show_feature_values_fallback(X_row, X_test, feature_names):
    begin_panel("Feature deviation fallback", "Notebook 09 is optional; this fallback ranks features by how unusual they are relative to the population.")
    row_vals = X_row[feature_names].iloc[0]
    pop_means = X_test[feature_names].mean()
    deviations = ((row_vals - pop_means) / (X_test[feature_names].std() + 1e-9)).abs()
    top_devs = deviations.nlargest(10)

    display_df = pd.DataFrame({
        "Feature": top_devs.index,
        "Value": row_vals[top_devs.index].values,
        "Pop mean": pop_means[top_devs.index].values,
        "|Z-score|": top_devs.values,
    })
    st.dataframe(display_df.style.format({
        "Value": "{:.3f}", "Pop mean": "{:.3f}", "|Z-score|": "{:.2f}"
    }), use_container_width=True)
