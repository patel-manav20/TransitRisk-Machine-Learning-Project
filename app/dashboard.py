"""TransitRisk — Streamlit Dashboard.

Launch:
    cd transitrisk/
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
SIDEBAR_LOGO_PATH = Path("/Users/manavnayanbhaipatel/.cursor/projects/Users-manavnayanbhaipatel-Desktop-SJSU-DATA-245-ML-Project/assets/TransitRisk-9d218363-04fb-41a1-8e7b-d6a05b50669b.png")

from components.cost_tuner      import render_cost_tuner
from components.risk_panel      import render_risk_panel
from components.shap_panel      import render_shap_panel
from components.stress_explorer import render_stress_explorer
from components.streaming_demo  import render_streaming_demo
from components.ui              import inject_callout, inject_metric_grid, inject_shell_marker
from components.what_if         import render_what_if

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TransitRisk",
    page_icon="TR",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --tr-bg: #0a0f1a;
        --tr-panel: #111827;
        --tr-panel-alt: #0f172a;
        --tr-border: rgba(148, 163, 184, 0.18);
        --tr-text: #f8fafc;
        --tr-muted: #cbd5e1;
        --tr-subtle: #e2e8f0;
        --tr-accent: #7c9cff;
        --tr-accent-strong: #adc0ff;
        --tr-green: #34c77b;
        --tr-amber: #f6b85f;
        --tr-red: #f27d7d;
        --tr-purple: #8b8cf8;
    }
    *, *::before, *::after {
        box-sizing: border-box;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(124, 156, 255, 0.10), transparent 24%),
            radial-gradient(circle at top right, rgba(139, 140, 248, 0.08), transparent 20%),
            linear-gradient(180deg, #0a0f1a 0%, #0b1120 100%);
        color: var(--tr-text);
    }
    .block-container {
        max-width: 1480px;
        padding-top: 0.35rem;
        padding-bottom: 3.2rem;
        padding-left: 2.35rem;
        padding-right: 2.35rem;
    }
    header[data-testid="stHeader"] {
        height: 0;
        background: transparent;
    }
    [data-testid="stToolbar"] {
        right: 0.65rem;
        top: 0.45rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 15, 26, 0.98), rgba(13, 18, 31, 0.96));
        border-right: 1px solid var(--tr-border);
        width: 355px !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.35rem;
    }
    section[data-testid="stSidebar"] * { color: var(--tr-text) !important; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: var(--tr-subtle) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] strong {
        color: var(--tr-text) !important;
    }
    section[data-testid="stSidebar"] code {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.18rem 0.52rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.83rem;
        white-space: nowrap;
        color: var(--tr-text) !important;
        background: rgba(240, 246, 255, 0.12) !important;
        border: 1px solid rgba(194, 212, 240, 0.28) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }
    section[data-testid="stSidebar"] ul {
        padding-left: 1.15rem;
    }
    section[data-testid="stSidebar"] li {
        margin-bottom: 0.55rem;
    }
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        border-radius: 14px;
        padding: 0.9rem 0.9rem 0.8rem 0.9rem;
        min-height: 98px;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        line-height: 1.2;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        line-height: 1.02;
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        line-height: 1.35;
    }
    .tr-sidebar-section-title {
        margin: 0 0 0.7rem 0;
        color: var(--tr-text);
        font-size: 0.98rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .tr-sidebar-brand {
        display: grid;
        justify-items: center;
        text-align: center;
        gap: 0.7rem;
        margin-bottom: 0.35rem;
    }
    .tr-sidebar-brand-logo {
        width: 112px;
        max-width: 100%;
        height: auto;
        display: block;
        border-radius: 16px;
    }
    .tr-sidebar-brand-title {
        color: var(--tr-text);
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        line-height: 1.05;
    }
    .tr-sidebar-brand-subtitle {
        color: var(--tr-subtle);
        font-size: 0.9rem;
        line-height: 1.45;
        max-width: 220px;
    }
    .tr-sidebar-stack {
        display: grid;
        gap: 0.7rem;
    }
    .tr-sidebar-metric-grid {
        display: grid;
        gap: 0.7rem;
    }
    .tr-sidebar-card {
        padding: 0.92rem 0.95rem;
        border-radius: 16px;
        border: 1px solid rgba(194, 212, 240, 0.16);
        background: linear-gradient(180deg, rgba(18, 28, 45, 0.94), rgba(12, 21, 36, 0.92));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .tr-sidebar-metric-card {
        min-height: 88px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 0.45rem;
        padding: 0.9rem 0.95rem;
        border-radius: 16px;
        border: 1px solid rgba(194, 212, 240, 0.16);
        background: linear-gradient(180deg, rgba(18, 28, 45, 0.94), rgba(12, 21, 36, 0.92));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .tr-sidebar-metric-card.compact {
        min-height: 82px;
    }
    .tr-sidebar-metric-card.inline {
        min-height: 72px;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
    }
    .tr-sidebar-metric-label {
        color: var(--tr-text-secondary);
        font-size: 0.84rem;
        line-height: 1.3;
        font-weight: 600;
    }
    .tr-sidebar-metric-value {
        color: var(--tr-text);
        font-size: 1.15rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .tr-sidebar-list {
        display: grid;
        gap: 0.62rem;
    }
    .tr-sidebar-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 0.75rem;
        align-items: center;
    }
    .tr-sidebar-label {
        color: var(--tr-text-secondary);
        font-size: 0.9rem;
        line-height: 1.35;
        font-weight: 600;
    }
    .tr-sidebar-value {
        color: var(--tr-text);
        font-size: 0.9rem;
        line-height: 1.2;
        font-weight: 700;
        text-align: right;
        white-space: nowrap;
    }
    .tr-sidebar-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 34px;
        width: fit-content;
        padding: 0.4rem 0.78rem;
        border-radius: 12px;
        border: 1px solid rgba(194, 212, 240, 0.24);
        background: rgba(240, 246, 255, 0.12);
        color: var(--tr-text);
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    [data-testid="collapsedControl"] {
        background: rgba(15, 23, 37, 0.92);
        border: 1px solid var(--tr-border);
        border-radius: 12px;
        margin-top: 0.75rem;
        margin-left: 0.75rem;
    }
    .tr-shell-marker { display:none; }
    div[data-testid="stVerticalBlock"]:has(.tr-shell-topbar) {
        margin-bottom: 1.35rem;
    }
    div[data-testid="stVerticalBlock"]:has(.tr-shell-topbar) > div[data-testid="element-container"]:first-child + div,
    div[data-testid="stVerticalBlock"]:has(.tr-shell-topbar) {
        gap: 0.75rem;
    }
    div[data-testid="stVerticalBlock"]:has(.tr-shell-hero),
    div[data-testid="stVerticalBlock"]:has(.tr-shell-kpis),
    div[data-testid="stVerticalBlock"]:has(.tr-shell-tabs),
    div[data-testid="stVerticalBlock"]:has(.tr-shell-tabpage) {
        position: relative;
        padding: 1.1rem;
        border-radius: 24px;
        border: 1px solid var(--tr-border);
        background: linear-gradient(180deg, rgba(15, 23, 37, 0.96), rgba(11, 18, 30, 0.94));
        box-shadow: 0 14px 34px rgba(2, 6, 23, 0.20);
    }
    div[data-testid="stVerticalBlock"]:has(.tr-shell-kpis) {
        padding: 1.15rem 1.2rem 1.2rem 1.2rem;
        margin-top: 1.05rem;
        margin-bottom: 1.15rem;
    }
    div[data-testid="stVerticalBlock"]:has(.tr-shell-tabs) {
        margin-top: 1.5rem;
        padding: 1.15rem 1.25rem 1.2rem 1.25rem;
    }
    div[data-testid="stVerticalBlock"]:has(.tr-shell-tabpage) {
        padding: 1.55rem 1.5rem 1.7rem 1.5rem;
        margin-top: 0.7rem;
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.98), rgba(11, 18, 30, 0.96));
    }
    div[data-testid="stVerticalBlock"]:has(.tr-shell-tabpage) > div[data-testid="element-container"]:not(:last-child) {
        margin-bottom: 1.2rem;
    }
    div[data-testid="stVerticalBlock"]:has(.tr-shell-tabpage) div[data-testid="stHorizontalBlock"] {
        gap: 1.35rem;
    }
    div[data-testid="stVerticalBlock"]:has(.tr-metric-grid) {
        gap: 0;
    }
    div[data-testid="stVerticalBlock"]:has(.tr-shell-hero) {
        padding: 0.4rem;
        background: transparent;
        border: 0;
        box-shadow: none;
    }
    .tr-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: 0.2rem 0.1rem 0.6rem 0.1rem;
    }
    .tr-topbar-brand {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .tr-brand-mark {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(124, 156, 255, 0.92), rgba(139, 140, 248, 0.86));
        display:flex;
        align-items:center;
        justify-content:center;
        color:white;
        font-size:1.2rem;
        font-weight:700;
        box-shadow: 0 8px 20px rgba(59, 91, 219, 0.24);
    }
    .tr-topbar-copy h4 {
        margin: 0;
        font-size: 0.98rem;
        color: var(--tr-text);
        letter-spacing: -0.02em;
    }
    .tr-topbar-copy p {
        margin: 0.15rem 0 0 0;
        color: var(--tr-subtle);
        font-size: 0.86rem;
    }
    .tr-topbar-meta {
        display:flex;
        gap:0.6rem;
        flex-wrap:wrap;
        justify-content:flex-end;
    }
    .tr-meta-pill {
        padding:0.45rem 0.72rem;
        border-radius:999px;
        border:1px solid var(--tr-border);
        background: rgba(255, 255, 255, 0.06);
        color: var(--tr-text);
        font-size:0.79rem;
        font-weight:700;
        white-space:nowrap;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.55rem;
        background: rgba(9, 14, 24, 0.92);
        border: 1px solid var(--tr-border);
        border-radius: 16px;
        padding: 0.55rem;
        margin-top: 0.45rem;
        margin-bottom: 0.9rem;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        min-height: 46px;
        padding: 0.85rem 1.05rem;
        border-radius: 14px;
        color: var(--tr-subtle);
        background: transparent;
        border: 1px solid transparent;
        font-weight: 700;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(124, 156, 255, 0.14), rgba(139, 140, 248, 0.10));
        border-color: rgba(124, 156, 255, 0.22) !important;
        color: var(--tr-text) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(15, 27, 45, 0.95), rgba(12, 22, 38, 0.95));
        border: 1px solid var(--tr-border);
        border-radius: 16px;
        padding: 0.95rem 0.95rem 0.85rem 0.95rem;
    }
    [data-testid="stMetricLabel"] {
        color: var(--tr-subtle) !important;
        font-size: 0.84rem !important;
        letter-spacing: 0.02em;
        font-weight: 700 !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--tr-text) !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricDelta"] {
        color: var(--tr-subtle) !important;
    }
    div[data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
        background: rgba(15, 23, 37, 0.98) !important;
        border: 1px solid rgba(191, 208, 234, 0.24) !important;
        border-radius: 12px !important;
        color: var(--tr-text) !important;
    }
    div[data-baseweb="select"] span,
    .stNumberInput input,
    .stTextInput input,
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stToggle label,
    .stRadio label {
        color: var(--tr-text) !important;
        font-weight: 600;
    }
    div[data-baseweb="select"] > div:focus-within,
    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: rgba(110, 168, 254, 0.66) !important;
        box-shadow: 0 0 0 1px rgba(110, 168, 254, 0.35);
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background: rgba(191, 208, 234, 0.34) !important;
    }
    .stSlider [role="slider"] {
        background: var(--tr-accent) !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 0 4px rgba(110, 168, 254, 0.22);
    }
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(124, 156, 255, 0.34);
        background: linear-gradient(180deg, rgba(124, 156, 255, 0.26), rgba(124, 156, 255, 0.16));
        color: var(--tr-text);
        font-weight: 700;
        padding: 0.6rem 1rem;
    }
    .stButton > button:hover {
        border-color: rgba(124, 156, 255, 0.52);
        background: linear-gradient(180deg, rgba(124, 156, 255, 0.34), rgba(124, 156, 255, 0.22));
    }
    .stDataFrame, div[data-testid="stTable"] {
        border: 1px solid var(--tr-border);
        border-radius: 16px;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stMarkdownContainer"],
    .stDataFrame div,
    div[data-testid="stTable"] * {
        color: var(--tr-text) !important;
    }
    .stAlert {
        border-radius: 16px;
        border: 1px solid var(--tr-border);
    }
    .stCaption {
        color: var(--tr-subtle) !important;
        font-size: 0.86rem !important;
        opacity: 1 !important;
    }
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown label {
        color: var(--tr-subtle);
    }
    .stMarkdown strong,
    .stMarkdown b {
        color: var(--tr-text);
    }
    .tr-hero {
        position: relative;
        overflow: hidden;
        padding: 1.4rem 1.5rem;
        border-radius: 24px;
        border: 1px solid var(--tr-border);
        background:
            radial-gradient(circle at 0% 0%, rgba(124, 156, 255, 0.14), transparent 24%),
            radial-gradient(circle at 100% 0%, rgba(139, 140, 248, 0.10), transparent 20%),
            linear-gradient(180deg, rgba(17, 24, 39, 0.98), rgba(11, 18, 30, 0.98));
        margin-bottom: 1rem;
        box-shadow: 0 16px 36px rgba(2, 6, 23, 0.24);
    }
    .tr-hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.95fr) minmax(280px, 0.85fr);
        gap: 1rem;
        align-items: end;
    }
    .tr-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(194, 212, 240, 0.28);
        color: var(--tr-text);
        font-size: 0.76rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-weight: 700;
    }
    .tr-hero h1 {
        margin: 0.8rem 0 0.5rem 0;
        font-size: 2.02rem !important;
        line-height: 1.04;
        letter-spacing: -0.05em;
        color: var(--tr-text);
        white-space: nowrap;
    }
    .tr-hero p {
        margin: 0;
        max-width: 760px;
        color: var(--tr-subtle);
        line-height: 1.6;
        font-size: 1rem;
    }
    .tr-hero-panel {
        display: grid;
        gap: 0.7rem;
        padding: 1rem;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: rgba(12, 18, 30, 0.72);
        backdrop-filter: blur(8px);
    }
    .tr-hero-panel-label {
        color: var(--tr-subtle);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }
    .tr-hero-panel-value {
        color: var(--tr-text);
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    .tr-hero-panel-note {
        color: #c1d0e8;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .tr-metric-tile {
        padding: 1rem 1.02rem 0.95rem 1.02rem;
        border-radius: 16px;
        border: 1px solid var(--tr-border);
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.92), rgba(15, 23, 37, 0.94));
        min-height: 148px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 0.55rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .tr-metric-label {
        color: var(--tr-subtle);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        line-height: 1.25;
        min-height: 1.35rem;
    }
    .tr-metric-value {
        color: var(--tr-text);
        font-size: 1.6rem;
        letter-spacing: -0.04em;
        font-weight: 700;
        line-height: 1.02;
        margin-top: 0.15rem;
        min-height: 1.7rem;
    }
    .tr-metric-help {
        color: var(--tr-subtle);
        font-size: 0.86rem;
        line-height: 1.45;
        margin-top: auto;
        min-height: 2.6rem;
    }
    .tr-metric-grid {
        display: grid;
        gap: 18px;
        align-items: stretch;
        width: 100%;
        grid-auto-rows: 1fr;
        margin-bottom: 1.2rem;
        margin-left: 0;
        margin-right: 0;
        padding: 0;
    }
    .tr-metric-grid-4 {
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }
    .tr-metric-grid-5 {
        grid-template-columns: repeat(5, minmax(0, 1fr));
    }
    .tr-shell-kpis + div .tr-metric-grid-5 .tr-metric-tile,
    .tr-metric-grid-5 .tr-metric-tile {
        height: 160px;
        min-height: 160px;
    }
    .tr-metric-grid-item {
        min-width: 0;
        height: 100%;
        display: flex;
        align-self: stretch;
    }
    .tr-callout {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid var(--tr-border);
        margin: 0.15rem 0 1.35rem 0;
        background: rgba(17, 24, 39, 0.92);
    }
    .tr-callout-title {
        color: var(--tr-text);
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .tr-callout-body {
        color: var(--tr-subtle);
        line-height: 1.55;
    }
    .tr-callout-default { background: rgba(15, 27, 45, 0.92); }
    .tr-callout-success { background: rgba(56, 210, 122, 0.12); border-color: rgba(56, 210, 122, 0.28); }
    .tr-callout-warning { background: rgba(255, 191, 102, 0.12); border-color: rgba(255, 191, 102, 0.3); }
    .tr-callout-danger { background: rgba(255, 125, 125, 0.12); border-color: rgba(255, 125, 125, 0.3); }
    .tr-inline-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 30px;
        padding: 0.36rem 0.72rem;
        border-radius: 999px;
        font-size: 0.79rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .tr-badge-default {
        color: var(--tr-text);
        background: rgba(240, 246, 255, 0.12);
        border: 1px solid rgba(194, 212, 240, 0.28);
    }
    .tr-badge-success {
        color: #dffceb;
        background: rgba(56, 210, 122, 0.18);
        border: 1px solid rgba(56, 210, 122, 0.34);
    }
    .tr-badge-warning {
        color: #fff1d8;
        background: rgba(255, 191, 102, 0.18);
        border: 1px solid rgba(255, 191, 102, 0.34);
    }
    .tr-badge-danger {
        color: #ffe1e1;
        background: rgba(255, 125, 125, 0.18);
        border: 1px solid rgba(255, 125, 125, 0.34);
    }
    .tr-kv-list {
        display: grid;
        gap: 0.62rem;
    }
    .tr-kv-item {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 1rem;
        align-items: center;
        padding: 0.12rem 0;
        min-height: 32px;
    }
    .tr-kv-label {
        color: var(--tr-subtle);
        font-size: 0.86rem;
        line-height: 1.35;
    }
    .tr-kv-value {
        color: var(--tr-text);
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.15;
        text-align: right;
        white-space: nowrap;
    }
    .tr-context-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.9rem 1rem;
        margin-top: 0.45rem;
        align-items: stretch;
    }
    .tr-context-item {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 0.42rem;
        min-height: 84px;
        padding: 0.92rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: linear-gradient(180deg, rgba(13, 22, 36, 0.82), rgba(10, 18, 31, 0.8));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    }
    .tr-context-label {
        color: var(--tr-subtle);
        font-size: 0.84rem;
        line-height: 1.35;
    }
    .tr-context-value {
        color: var(--tr-text);
        font-size: 1.08rem;
        font-weight: 700;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .tr-nav-caption {
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:1rem;
        margin-bottom:0.55rem;
    }
    .tr-nav-caption h3 {
        margin:0;
        color:var(--tr-text);
        font-size:1rem !important;
    }
    .tr-nav-caption p {
        margin:0.25rem 0 0 0;
        color:var(--tr-subtle);
        font-size:0.88rem;
        line-height:1.5;
        max-width:720px;
    }
    .tr-nav-sidehint {
        color:var(--tr-subtle);
        font-size:0.8rem;
        white-space:nowrap;
        padding-top: 0.22rem;
    }
    div[data-testid="stPlotlyChart"] {
        padding: 0.8rem 0.8rem 0.35rem 0.8rem;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: linear-gradient(180deg, rgba(11, 18, 30, 0.76), rgba(9, 16, 27, 0.72));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
    }
    div[data-testid="stPlotlyChart"] > div {
        border-radius: 14px;
    }
    .js-plotly-plot .modebar {
        display: none !important;
    }
    @media (max-width: 1100px) {
        .tr-hero-grid { grid-template-columns: 1fr; }
        .tr-topbar { flex-direction: column; align-items: flex-start; }
        .tr-topbar-meta { justify-content: flex-start; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .tr-nav-caption { flex-direction: column; }
        .tr-nav-sidehint { white-space: normal; padding-top: 0; }
        .tr-metric-grid-4,
        .tr-metric-grid-5 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 720px) {
        .tr-metric-grid-4,
        .tr-metric-grid-5 { grid-template-columns: 1fr; }
        .tr-context-grid { grid-template-columns: 1fr; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(240, 246, 255, 0.08);
        color: var(--tr-text) !important;
    }
    .stTabs [data-baseweb="tab"]:focus-visible {
        outline: 2px solid rgba(157, 194, 255, 0.55);
        outline-offset: 2px;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(18, 31, 51, 0.98), rgba(11, 21, 36, 0.98));
        box-shadow: none;
    }
    .stExpander {
        border: 1px solid var(--tr-border) !important;
        border-radius: 16px !important;
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.9), rgba(15, 23, 37, 0.94)) !important;
        overflow: hidden;
    }
    .stExpander summary {
        color: var(--tr-text) !important;
        font-weight: 700 !important;
    }
    .stExpander details > div {
        border-top: 1px solid rgba(194, 212, 240, 0.12);
    }
    .stDataFrame thead tr th,
    div[data-testid="stTable"] thead tr th {
        background: rgba(240, 246, 255, 0.06) !important;
        color: var(--tr-text) !important;
        font-weight: 700 !important;
        border-bottom: 1px solid rgba(194, 212, 240, 0.14) !important;
    }
    .stDataFrame tbody tr:hover td,
    div[data-testid="stTable"] tbody tr:hover td {
        background: rgba(240, 246, 255, 0.04) !important;
    }
    .stToggle label[data-testid="stWidgetLabel"],
    .stSlider label[data-testid="stWidgetLabel"],
    .stSelectbox label[data-testid="stWidgetLabel"],
    .stNumberInput label[data-testid="stWidgetLabel"] {
        color: var(--tr-text) !important;
    }
    .st-emotion-cache-16idsys p,
    .st-emotion-cache-16idsys span {
        color: var(--tr-text-secondary);
    }
    hr {
        border: none;
        border-top: 1px solid rgba(194, 212, 240, 0.12);
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
def _artifact_roots() -> list[Path]:
    roots = [ROOT]

    # Allow an explicit artifact bundle location for local runs.
    env_root = os.environ.get("TRANSITRISK_ARTIFACTS_DIR")
    if env_root:
        roots.append(Path(env_root).expanduser())

    sibling_bundle = ROOT.parent / "transitrisk_data_models"
    if sibling_bundle != ROOT:
        roots.append(sibling_bundle)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for base in roots:
        resolved = base.resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def _pick_dir(relative_path: str, required_files: list[str]) -> Path:
    for base in _artifact_roots():
        candidate = base / relative_path
        if all((candidate / name).exists() for name in required_files):
            return candidate
    return (_artifact_roots()[0] / relative_path)


def _image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


DATA_DIR = _pick_dir(
    "data/processed",
    ["modeling_table.parquet", "X_features.parquet", "train_val_test_indices.json"],
)
MODELS_DIR = _pick_dir(
    "models",
    ["xgb_calibrated.joblib"],
)
FIGS_DIR = _pick_dir(
    "figures",
    ["metrics.json"],
)


def _missing_artifacts_message() -> str:
    searched_roots = "\n".join(f"- `{p}`" for p in _artifact_roots())
    return (
        "📂 **Saved artifacts not found.** This dashboard is configured to use the "
        "existing trained outputs only and will not retrain anything.\n\n"
        "Place the saved files in either the repo folders (`data/`, `models/`), the "
        "sibling `transitrisk_data_models/` bundle, or point "
        "`TRANSITRISK_ARTIFACTS_DIR` at the extracted artifact directory.\n\n"
        "**Required files:**\n"
        "- `data/processed/modeling_table.parquet`\n"
        "- `data/processed/X_features.parquet`\n"
        "- `data/processed/train_val_test_indices.json`\n"
        "- `models/xgb_calibrated.joblib` (or `models/xgb.joblib`)\n\n"
        f"**Checked locations:**\n{searched_roots}"
    )


@st.cache_data(show_spinner="Loading data…")
def load_data():
    req = [DATA_DIR / "modeling_table.parquet",
           DATA_DIR / "X_features.parquet",
           DATA_DIR / "train_val_test_indices.json"]
    if any(not p.exists() for p in req):
        return None, None, None
    modeling = pd.read_parquet(DATA_DIR / "modeling_table.parquet")
    X_all    = pd.read_parquet(DATA_DIR / "X_features.parquet")
    with open(DATA_DIR / "train_val_test_indices.json") as f:
        idx = json.load(f)
    return modeling, X_all, (idx["train"], idx["val"], idx["test"])


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import joblib
    for name in ["xgb_calibrated.joblib", "xgb.joblib"]:
        p = MODELS_DIR / name
        if p.exists():
            return joblib.load(p)
    return None


@st.cache_data(show_spinner=False)
def load_prediction_sets():
    p = DATA_DIR / "prediction_sets.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_thresholds():
    p = DATA_DIR / "thresholds.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"t_default": 0.5, "t_f1": 0.345, "t_cost": 0.163}


@st.cache_data(show_spinner=False)
def load_metrics():
    p = FIGS_DIR / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


# ── Load everything ────────────────────────────────────────────────────────────
modeling, X_all, indices_tuple = load_data()
model = load_model()

if modeling is None or model is None:
    st.warning(_missing_artifacts_message())
    st.stop()

train_idx, val_idx, test_idx = indices_tuple
X_test        = X_all.iloc[test_idx].reset_index(drop=True)
y_test        = modeling["y_primary"].values[test_idx]
modeling_test = modeling.iloc[test_idx].reset_index(drop=True)
feature_names = X_all.columns.tolist()

prediction_sets = load_prediction_sets()
thresholds      = load_thresholds()
metrics         = load_metrics()

# ── Sidebar — Help Panel ───────────────────────────────────────────────────────
_HELP = {
    "📡 Risk Panel": """
### What this tab shows
Real-time next-hour delay risk for any station-route pair.

**How to use:**
1. Pick a **Station** (1–60).
2. The **Route** dropdown auto-updates to only show routes that serve that station — not all routes pass every station.
3. Read the **gauge** — the needle shows the model's probability that delays will be elevated next hour.
4. The **purple tick** on the gauge = alert threshold (t=0.163). Anything past that triggers a dispatch alert.
5. The **bar chart** compares risk across all routes at this station simultaneously.
6. The **history chart** shows the last 24 hourly predictions. Red dots on the x-axis = hours that *actually* had elevated delays.

**What "elevated risk" means:**  
The next hour's mean delay on this route exceeds 5 minutes.

**Conformal badge:**  
The "90% confident" tag is from conformal prediction — a mathematical guarantee that the set covers the true label at least 90% of the time.
""",
    "🔧 What-If": """
### What this tab shows
Sensitivity analysis — how does risk change if conditions worsen?

**How to use:**
1. Move the sliders to simulate different weather or operational conditions.
2. Watch the **risk comparison** update — it shows baseline (typical) vs your simulated scenario.
3. The **feature changes bar** shows what changed and by how much.

**Example questions you can answer:**
- "If it rains 15 mm/hr during morning peak, what happens to risk?"
- "If headways double (buses become infrequent), how much does risk increase?"
- "What if demand drops to half normal?"

**Key insight:**  
This is how operators would use the model interactively — before a storm or event — to understand how the network will behave.
""",
    "💰 Cost Tuner": """
### What this tab shows
How the alert threshold should change depending on the cost of missing a delay vs a false alarm.

**The core trade-off:**  
- **False Negative (miss):** Model says low risk, but delays actually happen → passengers stranded, no warning.  
- **False Positive (false alarm):** Model says high risk, but delays don't happen → unnecessary bus repositioning.

**How to use:**
1. Drag the **C(FN)/C(FP) slider** to set your relative cost ratio.
2. The cost curve updates → the minimum (orange line) shows the new optimal threshold.
3. The confusion matrix shows what predictions look like at that threshold.

**Default setting:**  
C(FN)=5, C(FP)=1 → we accept 5 false alarms to avoid 1 missed delay. This gives t=0.163.

**Practical meaning:**  
A city with severe delay consequences (e.g. airport shuttle service) might use C(FN)=10. A low-frequency rural bus might use C(FN)=2.
""",
    "🌡 Stress Explorer": """
### What this tab shows
Does the model hold up under different operating conditions?

**How to use:**
1. Pick a **slice axis** — Weather, Time of Day, Demand Level, or Headway.
2. The table shows ROC-AUC, PR-AUC, and F1 broken down by each sub-group.
3. Pick a stratum to see **example predictions** — True Positives, False Positives, False Negatives.

**What to look for:**
- Does AUC drop significantly during heavy rain? → model may need weather-specific retraining.
- Does performance drop during late night? → fewer trips = less signal in lag features.
- Does it vary by demand quartile? → high-demand routes may be easier to predict.

**Why this matters:**  
A model that averages 0.81 AUC but scores 0.65 during heavy rain is unreliable for its most important use case.
""",
    "🔍 SHAP": """
### What this tab shows
Which features drove the model's decision for a **specific** prediction.

**How to use:**
1. Enter a **Test row ID** (0 to 30,623) — each ID is one station-route-hour in the test set.
2. See the model's probability, the true label, and whether it was correct.
3. The **feature values table** shows the top-10 features by deviation from the population mean — i.e. what was unusual about this hour.

**Z-score column:**  
How many standard deviations from average. Z-score = 3 means that feature was extreme this hour.

**Note on SHAP:**  
Full SHAP values require running notebook 09. The fallback (current mode) shows statistical deviation — nearly as informative for understanding individual predictions.
""",
    "🔴 Live Feed": """
### What this tab shows
The full end-to-end inference pipeline running live on the test set.

**How to use:**
1. Set **replay speed** — Slow lets you read each prediction, Instant replays all at once.
2. Set **rows to replay** — more rows = more stable running metrics.
3. Toggle **Show feature inputs** to see which signals drove each decision.
4. Press **▶ Run simulation**.

**What you're watching:**
- **Gauge** — probability output from XGBoost for the current row.
- **Feature bar** — the 10 most informative inputs for this prediction (normalised to 0-1).
- **Decision badge** — DISPATCH ALERT (p ≥ 0.163) or HOLD.
- **Ground truth** — was the model right? ✓ or ✗.
- **Prediction log** — scrolling table of all decisions so far.
- **Running metrics** — Precision/Recall/F1 computed cumulatively.

**The key insight:**  
This is exactly what a transit operations centre would see — a live risk score arriving every hour for every route, with a binary action recommendation.
""",
}

with st.sidebar:
    sidebar_logo_uri = _image_data_uri(SIDEBAR_LOGO_PATH)
    st.markdown(
        f"""
        <div class="tr-sidebar-brand">
            {"<img class='tr-sidebar-brand-logo' src='" + sidebar_logo_uri + "' alt='Transit Risk logo' />" if sidebar_logo_uri else ""}
            <div class="tr-sidebar-brand-title">TRANSIT RISK</div>
            <div class="tr-sidebar-brand-subtitle">Operational intelligence for transit control rooms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    xgb_m = metrics.get("xgb_calibrated", metrics.get("xgb", {}))
    if xgb_m:
        st.markdown("**Model Performance**")
        col_a, col_b = st.columns(2)
        col_a.metric("ROC-AUC", f"{xgb_m.get('roc_auc', 0):.3f}")
        col_b.metric("PR-AUC",  f"{xgb_m.get('pr_auc', 0):.3f}")
        col_a.metric("F1",      f"{xgb_m.get('f1', 0):.3f}")
        col_b.metric("Brier",   f"{xgb_m.get('brier', 0):.3f}")

    st.divider()
    t = thresholds
    st.markdown('<div class="tr-sidebar-section-title">Decision Thresholds</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="tr-sidebar-metric-grid">
            <div class="tr-sidebar-metric-card compact inline">
                <div class="tr-sidebar-metric-label">Cost-optimal</div>
                <div class="tr-sidebar-chip">t = {t['t_cost']:.3f}</div>
            </div>
            <div class="tr-sidebar-metric-card compact inline">
                <div class="tr-sidebar-metric-label">F1-optimal</div>
                <div class="tr-sidebar-chip">t = {t['t_f1']:.3f}</div>
            </div>
            <div class="tr-sidebar-metric-card compact inline">
                <div class="tr-sidebar-metric-label">Default</div>
                <div class="tr-sidebar-chip">t = {t['t_default']:.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        f"""
        <div class="tr-sidebar-metric-grid">
            <div class="tr-sidebar-metric-card">
                <div class="tr-sidebar-metric-label">Test set</div>
                <div class="tr-sidebar-metric-value">{len(test_idx):,} rows</div>
            </div>
            <div class="tr-sidebar-metric-card">
                <div class="tr-sidebar-metric-label">Positive rate</div>
                <div class="tr-sidebar-metric-value">{y_test.mean():.1%}</div>
            </div>
            <div class="tr-sidebar-metric-card">
                <div class="tr-sidebar-metric-label">Features</div>
                <div class="tr-sidebar-metric-value">{len(feature_names)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Workflow Guide")
    help_tab = st.selectbox(
        "Select tab to read its guide",
        list(_HELP.keys()),
        label_visibility="collapsed",
    )
    st.markdown(_HELP[help_tab])

# ── Header ─────────────────────────────────────────────────────────────────────
positive_rate = y_test.mean()
xgb_m = metrics.get("xgb_calibrated", metrics.get("xgb", {}))

with st.container():
    inject_shell_marker("topbar")
    st.empty()

with st.container():
    inject_shell_marker("hero")
    st.markdown(
        f"""
        <section class="tr-hero">
            <div class="tr-hero-grid">
                <div>
                    <div class="tr-pill">Transit operations intelligence</div>
                    <h1>Forecast network disruption before passengers feel it.</h1>
                    <p>
                        A polished command center for next-hour delay risk, scenario planning,
                        threshold strategy, explainability, and live inference replay. The workflow
                        is arranged for executive demos while preserving the existing ML behavior.
                    </p>
                </div>
                <div class="tr-hero-panel">
                    <div class="tr-hero-panel-label">Primary model</div>
                    <div class="tr-hero-panel-value">XGBoost Calibrated</div>
                    <div class="tr-hero-panel-note">
                        Cost-sensitive alerting at <b>t = {thresholds['t_cost']:.3f}</b> across
                        <b>{len(test_idx):,}</b> held-out rows with a <b>{positive_rate:.1%}</b> elevated-delay rate.
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

with st.container():
    inject_shell_marker("kpis")
    inject_metric_grid([
        ("ROC-AUC", f"{xgb_m.get('roc_auc', 0):.3f}", "Held-out ranking performance"),
        ("PR-AUC", f"{xgb_m.get('pr_auc', 0):.3f}", "Alert precision under class imbalance"),
        ("Alert threshold", f"{thresholds['t_cost']:.3f}", "Operational dispatch cutoff"),
        ("Positive rate", f"{positive_rate:.1%}", "Observed elevated-delay prevalence"),
        ("Features", f"{len(feature_names)}", "Signals feeding the prediction layer"),
    ], columns=5)
    inject_callout(
        "Executive flow",
        "Start with the Risk Panel for the network view, then move into What-If analysis, "
        "threshold strategy, resilience testing, explainability, and the live replay simulator.",
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────
with st.container():
    inject_shell_marker("tabs")
    st.markdown(
        """
        <div class="tr-nav-caption">
            <div>
                <h3>Product workflow</h3>
                <p>Move from overview and inspection into scenario planning, policy tuning, diagnostics, and live demonstration without leaving the same operational shell.</p>
            </div>
            <div class="tr-nav-sidehint">6 modules · shared model context</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📡 Risk Panel", "🔧 What-If", "💰 Cost Tuner",
        "🌡 Stress Explorer", "🔍 SHAP", "🔴 Live Feed",
    ])

with tab1:
    with st.container():
        inject_shell_marker("tabpage")
        render_risk_panel(
            modeling_df=modeling,
            X_features=X_all,
            model=model,
            prediction_sets_df=prediction_sets,
            test_idx=test_idx,
        )

with tab2:
    with st.container():
        inject_shell_marker("tabpage")
        render_what_if(
            model=model,
            X_test=X_test,
            feature_names=feature_names,
            explainer=None,
        )

with tab3:
    with st.container():
        inject_shell_marker("tabpage")
        render_cost_tuner(
            model=model,
            X_test=X_test,
            y_test=y_test,
            t_f1=thresholds["t_f1"],
            t_cost_default=thresholds["t_cost"],
        )

with tab4:
    with st.container():
        inject_shell_marker("tabpage")
        render_stress_explorer(
            model=model,
            X_test=X_test,
            y_test=y_test,
            modeling_test=modeling_test,
            t_cost=thresholds["t_cost"],
        )

with tab5:
    with st.container():
        inject_shell_marker("tabpage")
        render_shap_panel(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            explainer=None,
        )

with tab6:
    with st.container():
        inject_shell_marker("tabpage")
        render_streaming_demo(
            model=model,
            modeling_test=modeling_test,
            X_test=X_test,
            y_test=y_test,
            t_cost=thresholds["t_cost"],
            feature_names=feature_names,
        )
