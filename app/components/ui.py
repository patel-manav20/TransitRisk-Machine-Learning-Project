"""Reusable presentation helpers and theme tokens for the dashboard."""
from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import streamlit as st


@dataclass(frozen=True)
class Palette:
    bg: str = "#08111f"
    panel: str = "#0f1b2d"
    panel_alt: str = "#121f33"
    elevated: str = "#15243b"
    border: str = "rgba(148, 163, 184, 0.24)"
    text: str = "#f3f7ff"
    text_secondary: str = "#d8e4f7"
    muted: str = "#bfd0ea"
    accent: str = "#6ea8fe"
    accent_strong: str = "#9dc2ff"
    accent_soft: str = "rgba(110, 168, 254, 0.22)"
    success: str = "#38d27a"
    warning: str = "#ffbf66"
    danger: str = "#ff7d7d"
    info: str = "#8db9ff"
    purple: str = "#a78bfa"
    grid: str = "rgba(148, 163, 184, 0.24)"


@dataclass(frozen=True)
class Typography:
    page_title: str = "2.45rem"
    section_title: str = "1.35rem"
    subsection_title: str = "1rem"
    body: str = "1rem"
    label: str = "0.82rem"
    caption: str = "0.86rem"
    metric: str = "1.6rem"
    chip: str = "0.79rem"


@dataclass(frozen=True)
class Spacing:
    page_x: str = "2rem"
    page_y_top: str = "0.35rem"
    page_y_bottom: str = "2.8rem"
    section_gap: str = "1rem"
    card_padding: str = "1rem"
    panel_padding: str = "1.1rem"
    inline_gap: str = "0.65rem"


@dataclass(frozen=True)
class Radii:
    small: str = "8px"
    medium: str = "14px"
    large: str = "18px"
    xl: str = "24px"
    shell: str = "28px"
    pill: str = "999px"


@dataclass(frozen=True)
class Shadows:
    card: str = "0 12px 28px rgba(3, 8, 20, 0.18)"
    elevated: str = "0 20px 40px rgba(3, 8, 20, 0.28)"
    shell: str = "0 18px 36px rgba(3, 8, 20, 0.22)"


PALETTE = Palette()
TYPOGRAPHY = Typography()
SPACING = Spacing()
RADII = Radii()
SHADOWS = Shadows()


def build_global_styles() -> str:
    p = PALETTE
    t = TYPOGRAPHY
    s = SPACING
    r = RADII
    sh = SHADOWS
    return f"""
<style>
    :root {{
        --tr-bg: {p.bg};
        --tr-panel: {p.panel};
        --tr-panel-alt: {p.panel_alt};
        --tr-elevated: {p.elevated};
        --tr-border: {p.border};
        --tr-text: {p.text};
        --tr-text-secondary: {p.text_secondary};
        --tr-muted: {p.muted};
        --tr-accent: {p.accent};
        --tr-accent-strong: {p.accent_strong};
        --tr-accent-soft: {p.accent_soft};
        --tr-success: {p.success};
        --tr-warning: {p.warning};
        --tr-danger: {p.danger};
        --tr-info: {p.info};
        --tr-purple: {p.purple};
    }}
    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(110, 168, 254, 0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(167, 139, 250, 0.15), transparent 24%),
            linear-gradient(180deg, {p.bg} 0%, #0b1424 100%);
        color: var(--tr-text);
    }}
    .block-container {{
        max-width: 1380px;
        padding-top: {s.page_y_top};
        padding-bottom: {s.page_y_bottom};
        padding-left: {s.page_x};
        padding-right: {s.page_x};
    }}
    header[data-testid="stHeader"] {{
        height: 0;
        background: transparent;
    }}
    [data-testid="stToolbar"] {{
        right: 0.65rem;
        top: 0.45rem;
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(9, 17, 31, 0.98), rgba(10, 20, 35, 0.94));
        border-right: 1px solid var(--tr-border);
        width: 355px !important;
    }}
    section[data-testid="stSidebar"] * {{ color: var(--tr-text) !important; }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label {{
        color: var(--tr-text-secondary) !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] strong {{
        color: var(--tr-text) !important;
    }}
    section[data-testid="stSidebar"] code,
    .tr-meta-pill,
    .tr-pill,
    .tr-inline-badge {{
        color: var(--tr-text) !important;
        background: rgba(240, 246, 255, 0.12) !important;
        border: 1px solid rgba(194, 212, 240, 0.28) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }}
    section[data-testid="stSidebar"] code {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.18rem 0.52rem;
        border-radius: {r.small};
        font-weight: 700;
        font-size: {t.chip};
        white-space: nowrap;
    }}
    section[data-testid="stSidebar"] ul {{ padding-left: 1.15rem; }}
    section[data-testid="stSidebar"] li {{ margin-bottom: 0.55rem; }}
    [data-testid="collapsedControl"] {{
        background: rgba(15, 27, 45, 0.86);
        border: 1px solid var(--tr-border);
        border-radius: 12px;
        margin-top: 0.75rem;
        margin-left: 0.75rem;
    }}
    .tr-shell-marker {{ display:none; }}
    div[data-testid="stVerticalBlock"]:has(.tr-shell-topbar) {{ margin-bottom: {s.section_gap}; }}
    div[data-testid="stVerticalBlock"]:has(.tr-shell-topbar) > div[data-testid="element-container"]:first-child + div,
    div[data-testid="stVerticalBlock"]:has(.tr-shell-topbar) {{ gap: 0.75rem; }}
    div[data-testid="stVerticalBlock"]:has(.tr-shell-hero),
    div[data-testid="stVerticalBlock"]:has(.tr-shell-kpis),
    div[data-testid="stVerticalBlock"]:has(.tr-shell-tabs),
    div[data-testid="stVerticalBlock"]:has(.tr-shell-tabpage) {{
        position: relative;
        padding: {s.panel_padding};
        border-radius: {r.shell};
        border: 1px solid var(--tr-border);
        background: linear-gradient(180deg, rgba(11, 20, 36, 0.96), rgba(8, 16, 28, 0.92));
        box-shadow: {sh.shell};
    }}
    div[data-testid="stVerticalBlock"]:has(.tr-shell-kpis) {{
        padding-top: 1rem;
        padding-bottom: 1rem;
        margin-top: 0.8rem;
    }}
    div[data-testid="stVerticalBlock"]:has(.tr-shell-tabs) {{
        margin-top: 1rem;
        padding-top: 0.9rem;
    }}
    div[data-testid="stVerticalBlock"]:has(.tr-shell-tabpage) {{
        padding: 1.2rem 1.2rem 1.35rem 1.2rem;
        margin-top: 0.2rem;
        background: linear-gradient(180deg, rgba(13, 23, 40, 0.98), rgba(8, 16, 28, 0.95));
    }}
    div[data-testid="stVerticalBlock"]:has(.tr-shell-hero) {{
        padding: 0.4rem;
        background: transparent;
        border: 0;
        box-shadow: none;
    }}
    .tr-topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: 0.2rem 0.1rem 0.6rem 0.1rem;
    }}
    .tr-topbar-brand {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }}
    .tr-brand-mark {{
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(110, 168, 254, 0.95), rgba(167, 139, 250, 0.9));
        display:flex;
        align-items:center;
        justify-content:center;
        color:white;
        font-size:1.2rem;
        font-weight:700;
        box-shadow: 0 10px 22px rgba(110, 168, 254, 0.28);
    }}
    .tr-topbar-copy h4 {{
        margin: 0;
        font-size: 0.98rem;
        color: var(--tr-text);
        letter-spacing: -0.02em;
    }}
    .tr-topbar-copy p {{
        margin: 0.15rem 0 0 0;
        color: var(--tr-text-secondary);
        font-size: {t.caption};
    }}
    .tr-topbar-meta {{
        display:flex;
        gap:0.6rem;
        flex-wrap:wrap;
        justify-content:flex-end;
    }}
    .tr-meta-pill {{
        padding:0.45rem 0.72rem;
        border-radius:{r.pill};
        background: rgba(240, 246, 255, 0.12);
        color: var(--tr-text);
        font-size:{t.chip};
        font-weight:700;
        white-space:nowrap;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.55rem;
        background: rgba(8, 16, 28, 0.9);
        border: 1px solid var(--tr-border);
        border-radius: 20px;
        padding: 0.45rem;
        margin-top: 0.35rem;
        margin-bottom: 0.6rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: auto;
        padding: 0.8rem 1rem;
        border-radius: {r.medium};
        color: var(--tr-text-secondary);
        background: transparent;
        border: 1px solid transparent;
        font-weight: 700;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(180deg, rgba(110, 168, 254, 0.18), rgba(167, 139, 250, 0.14));
        border-color: rgba(110, 168, 254, 0.22) !important;
        color: var(--tr-text) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, rgba(15, 27, 45, 0.95), rgba(12, 22, 38, 0.95));
        border: 1px solid var(--tr-border);
        border-radius: {r.large};
        padding: {s.card_padding} {s.card_padding} 0.9rem {s.card_padding};
    }}
    [data-testid="stMetricLabel"] {{
        color: var(--tr-text-secondary) !important;
        font-size: 0.84rem !important;
        letter-spacing: 0.02em;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricValue"] {{
        color: var(--tr-text) !important;
        font-weight: 800 !important;
    }}
    [data-testid="stMetricDelta"] {{ color: var(--tr-text-secondary) !important; }}
    .stCaption {{
        color: var(--tr-text-secondary) !important;
        font-size: {t.caption} !important;
        opacity: 1 !important;
    }}
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown label {{ color: var(--tr-text-secondary); }}
    .stMarkdown strong,
    .stMarkdown b {{ color: var(--tr-text); }}
    div[data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {{
        background: rgba(12, 22, 38, 0.96) !important;
        border: 1px solid rgba(191, 208, 234, 0.24) !important;
        border-radius: {r.medium} !important;
        color: var(--tr-text) !important;
    }}
    div[data-baseweb="select"] span,
    .stNumberInput input,
    .stTextInput input,
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label,
    .stToggle label,
    .stRadio label {{
        color: var(--tr-text) !important;
        font-weight: 600;
    }}
    div[data-baseweb="select"] > div:focus-within,
    .stNumberInput input:focus,
    .stTextInput input:focus {{
        border-color: rgba(110, 168, 254, 0.66) !important;
        box-shadow: 0 0 0 1px rgba(110, 168, 254, 0.35);
    }}
    .stSlider [data-baseweb="slider"] > div > div {{ background: rgba(191, 208, 234, 0.34) !important; }}
    .stSlider [role="slider"] {{
        background: var(--tr-accent) !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 0 4px rgba(110, 168, 254, 0.22);
    }}
    .stButton > button {{
        border-radius: {r.medium};
        border: 1px solid rgba(157, 194, 255, 0.38);
        background: linear-gradient(180deg, rgba(110, 168, 254, 0.34), rgba(110, 168, 254, 0.2));
        color: var(--tr-text);
        font-weight: 700;
        padding: 0.6rem 1rem;
    }}
    .stButton > button:hover {{
        border-color: rgba(157, 194, 255, 0.64);
        background: linear-gradient(180deg, rgba(110, 168, 254, 0.44), rgba(110, 168, 254, 0.28));
    }}
    .stDataFrame, div[data-testid="stTable"] {{
        border: 1px solid var(--tr-border);
        border-radius: {r.large};
        overflow: hidden;
    }}
    .stDataFrame [data-testid="stMarkdownContainer"],
    .stDataFrame div,
    div[data-testid="stTable"] * {{
        color: var(--tr-text) !important;
    }}
    .stAlert {{
        border-radius: 16px;
        border: 1px solid var(--tr-border);
    }}
    .tr-hero {{
        position: relative;
        overflow: hidden;
        padding: 1.55rem 1.65rem;
        border-radius: {r.xl};
        border: 1px solid var(--tr-border);
        background:
            radial-gradient(circle at 0% 0%, rgba(110, 168, 254, 0.22), transparent 28%),
            radial-gradient(circle at 100% 0%, rgba(167, 139, 250, 0.18), transparent 24%),
            linear-gradient(180deg, rgba(16, 28, 47, 0.98), rgba(10, 19, 34, 0.98));
        margin-bottom: 1rem;
        box-shadow: {sh.elevated};
    }}
    .tr-hero-grid {{
        display: grid;
        grid-template-columns: 1.7fr 1fr;
        gap: 1rem;
        align-items: end;
    }}
    .tr-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.7rem;
        border-radius: {r.pill};
        color: var(--tr-text);
        font-size: 0.76rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-weight: 700;
    }}
    .tr-hero h1 {{
        margin: 0.8rem 0 0.5rem 0;
        font-size: {t.page_title} !important;
        line-height: 1.02;
        letter-spacing: -0.04em;
        color: var(--tr-text);
    }}
    .tr-hero p {{
        margin: 0;
        max-width: 760px;
        color: var(--tr-text-secondary);
        line-height: 1.6;
        font-size: {t.body};
    }}
    .tr-hero-panel {{
        display: grid;
        gap: 0.7rem;
        padding: {s.card_padding};
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: rgba(8, 17, 31, 0.58);
        backdrop-filter: blur(8px);
    }}
    .tr-hero-panel-label {{
        color: var(--tr-text-secondary);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }}
    .tr-hero-panel-value {{
        color: var(--tr-text);
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }}
    .tr-hero-panel-note {{
        color: var(--tr-text-secondary);
        font-size: 0.9rem;
        line-height: 1.45;
    }}
    .tr-metric-tile {{
        padding: {s.card_padding} {s.card_padding} 0.95rem {s.card_padding};
        border-radius: {r.large};
        border: 1px solid var(--tr-border);
        background: linear-gradient(180deg, rgba(15, 27, 45, 0.9), rgba(12, 22, 38, 0.92));
        min-height: 112px;
        box-shadow: {sh.card};
    }}
    .tr-metric-label {{
        color: var(--tr-text-secondary);
        font-size: {t.label};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }}
    .tr-metric-value {{
        color: var(--tr-text);
        font-size: {t.metric};
        letter-spacing: -0.04em;
        font-weight: 700;
        margin-top: 0.35rem;
    }}
    .tr-metric-help {{
        color: var(--tr-text-secondary);
        font-size: {t.caption};
        line-height: 1.45;
        margin-top: 0.45rem;
    }}
    .tr-section-title {{
        margin: 0.2rem 0 1rem 0;
    }}
    .tr-eyebrow {{
        color: var(--tr-accent-strong);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.74rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    .tr-section-title h2 {{
        margin: 0;
        color: var(--tr-text);
        font-size: {t.section_title} !important;
        letter-spacing: -0.03em;
    }}
    .tr-section-title p,
    .tr-panel-header p {{
        margin: 0.35rem 0 0 0;
        color: var(--tr-text-secondary);
        line-height: 1.55;
        max-width: 760px;
    }}
    .tr-panel-header {{
        margin-bottom: 0.7rem;
    }}
    .tr-panel-header h3 {{
        margin: 0;
        color: var(--tr-text);
        font-size: {t.subsection_title} !important;
    }}
    .tr-callout {{
        padding: {s.card_padding} 1.1rem;
        border-radius: {r.large};
        border: 1px solid var(--tr-border);
        margin: 0.5rem 0 1rem 0;
    }}
    .tr-callout-title {{
        color: var(--tr-text);
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    .tr-callout-body {{
        color: var(--tr-text-secondary);
        line-height: 1.55;
    }}
    .tr-callout-default {{ background: rgba(15, 27, 45, 0.92); }}
    .tr-callout-success {{ background: rgba(56, 210, 122, 0.12); border-color: rgba(56, 210, 122, 0.28); }}
    .tr-callout-warning {{ background: rgba(255, 191, 102, 0.12); border-color: rgba(255, 191, 102, 0.3); }}
    .tr-callout-danger {{ background: rgba(255, 125, 125, 0.12); border-color: rgba(255, 125, 125, 0.3); }}
    .tr-inline-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.34rem 0.68rem;
        border-radius: {r.pill};
        font-size: {t.chip};
        font-weight: 700;
        line-height: 1;
    }}
    .tr-badge-default {{
        color: var(--tr-text);
        background: rgba(240, 246, 255, 0.12);
        border: 1px solid rgba(194, 212, 240, 0.28);
    }}
    .tr-badge-success {{
        color: #dffceb;
        background: rgba(56, 210, 122, 0.18);
        border: 1px solid rgba(56, 210, 122, 0.34);
    }}
    .tr-badge-warning {{
        color: #fff1d8;
        background: rgba(255, 191, 102, 0.18);
        border: 1px solid rgba(255, 191, 102, 0.34);
    }}
    .tr-badge-danger {{
        color: #ffe1e1;
        background: rgba(255, 125, 125, 0.18);
        border: 1px solid rgba(255, 125, 125, 0.34);
    }}
    .tr-status-row {{
        display:flex;
        align-items:center;
        gap: {s.inline_gap};
        margin-bottom: 0.55rem;
        flex-wrap: wrap;
    }}
    .tr-status-meta {{
        color: var(--tr-text-secondary);
        font-size: {t.caption};
        font-weight: 600;
    }}
    .tr-kv-list {{
        display:grid;
        gap:0.45rem;
    }}
    .tr-kv-item {{
        display:grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 1rem;
        align-items:center;
        padding: 0.1rem 0;
    }}
    .tr-kv-label {{
        color: var(--tr-text-secondary);
        font-size: {t.caption};
    }}
    .tr-kv-value {{
        color: var(--tr-text);
        font-size: 0.9rem;
        font-weight: 700;
        text-align:right;
    }}
    .tr-nav-caption {{
        display:flex;
        justify-content:space-between;
        align-items:flex-end;
        gap:1rem;
        margin-bottom:0.2rem;
    }}
    .tr-nav-caption h3 {{
        margin:0;
        color:var(--tr-text);
        font-size:{t.subsection_title} !important;
    }}
    .tr-nav-caption p {{
        margin:0.25rem 0 0 0;
        color:var(--tr-text-secondary);
        font-size:0.88rem;
        line-height:1.5;
        max-width:720px;
    }}
    .tr-nav-sidehint {{
        color:var(--tr-text-secondary);
        font-size:0.8rem;
        white-space:nowrap;
    }}
    @media (max-width: 1100px) {{
        .tr-hero-grid {{ grid-template-columns: 1fr; }}
        .tr-topbar {{ flex-direction: column; align-items: flex-start; }}
        .tr-topbar-meta {{ justify-content: flex-start; }}
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    }}
</style>
"""


def inject_section_title(eyebrow: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="tr-section-title">
            <div class="tr-eyebrow">{eyebrow}</div>
            <h2>{title}</h2>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_callout(title: str, body: str, tone: str = "default") -> None:
    tone_class = {
        "default": "tr-callout-default",
        "success": "tr-callout-success",
        "warning": "tr-callout-warning",
        "danger": "tr-callout-danger",
    }.get(tone, "tr-callout-default")
    st.markdown(
        f"""
        <div class="tr-callout {tone_class}">
            <div class="tr-callout-title">{title}</div>
            <div class="tr-callout-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_metric_tile(label: str, value: str, help_text: str | None = None) -> None:
    help_html = f'<div class="tr-metric-help">{help_text}</div>' if help_text else ""
    st.markdown(
        f"""
        <div class="tr-metric-tile">
            <div class="tr-metric-label">{label}</div>
            <div class="tr-metric-value">{value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_metric_grid(items: list[tuple[str, str, str | None]], columns: int = 4) -> None:
    cards_parts: list[str] = []
    for label, value, help_text in items:
        help_html = f'<div class="tr-metric-help">{help_text}</div>' if help_text else ""
        cards_parts.append(
            f'<div class="tr-metric-grid-item">'
            f'<div class="tr-metric-tile">'
            f'<div class="tr-metric-label">{label}</div>'
            f'<div class="tr-metric-value">{value}</div>'
            f"{help_html}"
            f"</div>"
            f"</div>"
        )
    cards = "".join(cards_parts)
    st.markdown(
        f'<div class="tr-metric-grid tr-metric-grid-{columns}">{cards}</div>',
        unsafe_allow_html=True,
    )


def inject_inline_badge(text: str, tone: str = "default") -> None:
    tone_class = {
        "default": "tr-badge-default",
        "success": "tr-badge-success",
        "warning": "tr-badge-warning",
        "danger": "tr-badge-danger",
    }.get(tone, "tr-badge-default")
    st.markdown(f'<span class="tr-inline-badge {tone_class}">{text}</span>', unsafe_allow_html=True)


def inject_status_badge(text: str, tone: str = "default", meta: str | None = None) -> None:
    tone_class = {
        "default": "tr-badge-default",
        "success": "tr-badge-success",
        "warning": "tr-badge-warning",
        "danger": "tr-badge-danger",
    }.get(tone, "tr-badge-default")
    meta_html = f'<span class="tr-status-meta">{meta}</span>' if meta else ""
    st.markdown(
        f"""
        <div class="tr-status-row">
            <span class="tr-inline-badge {tone_class}">{text}</span>
            {meta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def begin_panel(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="tr-panel-header">
            <h3>{title}</h3>
            {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_shell_marker(name: str) -> None:
    st.markdown(f'<div class="tr-shell-marker tr-shell-{name}"></div>', unsafe_allow_html=True)


def inject_key_value_list(items: list[tuple[str, str]]) -> None:
    rows = "".join(
        f'<div class="tr-kv-item"><div class="tr-kv-label">{label}</div><div class="tr-kv-value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="tr-kv-list">{rows}</div>', unsafe_allow_html=True)


def style_figure(
    fig: go.Figure,
    *,
    height: int | None = None,
    legend_orientation: str = "h",
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(t=24, b=46, l=42, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE.text, family="Inter, ui-sans-serif, system-ui, sans-serif"),
        legend=dict(
            orientation=legend_orientation,
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=PALETTE.text_secondary, size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=PALETTE.grid,
        tickfont=dict(color=PALETTE.text_secondary, size=11),
        title_font=dict(color=PALETTE.text_secondary, size=11),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=PALETTE.grid,
        zeroline=False,
        tickfont=dict(color=PALETTE.text_secondary, size=11),
        title_font=dict(color=PALETTE.text_secondary, size=11),
        automargin=True,
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig
