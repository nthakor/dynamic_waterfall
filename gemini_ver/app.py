"""
Main Streamlit App — Rule Waterfall Analyzer
=========================================
Features:
  • Smart column-role auto-detection (date, bad flag, rule indicators, categoricals)
  • Column Configuration panel — inspect & override detected roles
  • Date filter: year and quarter multi-select (driven by detected date column)
  • Categorical filters: one multiselect per detected/chosen categorical column
  • Dynamic rule grouping GUI (add/rename/delete groups, assign rules, reorder groups)
  • Waterfall chart: applications declined per group
  • Bad Rate % per group (bad declined / total declined × 100)
  • Rule-level detail scatter plot
  • Summary KPI cards
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

from analytics import (
    ColumnConfig,
    detect_column_roles,
    load_data,
    enrich_date_cols,
    filter_data,
    compute_waterfall,
    rule_level_stats,
)
from generate_data import generate_dummy_dataset

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Rule Waterfall Analyzer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

.section-header {
    font-size: 1.05rem; font-weight: 600; color: #e2e8f0;
    margin-top: 12px; margin-bottom: 4px;
    display: flex; align-items: center; gap: 8px;
}

.col-role-badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
    margin: 2px 3px;
}
.badge-date  { background: #2b6cb0; color: #bee3f8; }
.badge-bad   { background: #742a2a; color: #fed7d7; }
.badge-rule  { background: #276749; color: #c6f6d5; }
.badge-cat   { background: #553c9a; color: #e9d8fd; }
.badge-ign   { background: #2d3748; color: #718096; }

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #e2e8f0 !important; }
div[data-testid="stMetricLabel"] { color: #a0aec0 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DATA_PATH = "data/applications.parquet"


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
def _init():
    defaults = {
        "data_loaded": False,
        "col_config": None,  # ColumnConfig (auto-detected, then user-overridden)
        "groups": {},
        "group_order": [],
        "ungrouped": [],
        "df_raw": None,  # raw loaded df (no date enrichment yet)
        "df_enriched": None,  # df with _year / _quarter added
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init()


# ─────────────────────────────────────────────
# Data loading / generation
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load(path: str) -> pd.DataFrame:
    return load_data(path)


def ensure_data():
    if not os.path.exists(DATA_PATH):
        with st.spinner("⚙️ Generating dummy dataset (100k records, 30 rules)…"):
            generate_dummy_dataset(n_records=100_000, n_rules=30, output_path=DATA_PATH)

    df_raw = _load(DATA_PATH)

    if not st.session_state.data_loaded:
        cfg = detect_column_roles(df_raw)
        st.session_state.col_config = cfg
        st.session_state.ungrouped = list(cfg.rule_cols)
        st.session_state.groups = {}
        st.session_state.group_order = []
        st.session_state.df_raw = df_raw

        if cfg.date_col:
            st.session_state.df_enriched = enrich_date_cols(df_raw, cfg.date_col)
        else:
            st.session_state.df_enriched = df_raw

        st.session_state.data_loaded = True

    return st.session_state.df_raw, st.session_state.df_enriched


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Rule Waterfall")
    st.markdown("---")

    # ── Dataset settings ─────────────────────────────────────────────────
    with st.expander("⚙️ Dataset Settings", expanded=False):
        n_rec = st.selectbox(
            "Records",
            [10_000, 100_000, 500_000, 1_000_000, 3_000_000],
            index=1,
            format_func=lambda x: f"{x:,}",
        )
        n_rules = st.slider("Number of Rules", 5, 200, 30)
        if st.button("🔄 Regenerate Dataset"):
            with st.spinner("Generating…"):
                generate_dummy_dataset(
                    n_records=n_rec, n_rules=n_rules, output_path=DATA_PATH
                )
            _load.clear()
            st.session_state.data_loaded = False
            st.rerun()

    df_raw, df_enriched = ensure_data()
    cfg: ColumnConfig = st.session_state.col_config
    all_cols = list(df_raw.columns)

    # ── Column Configuration ──────────────────────────────────────────────
    with st.expander("🔬 Column Configuration", expanded=False):
        st.markdown(
            "Auto-detected roles are shown below. Override if needed, then click **Apply**."
        )

        # Auto-detection summary badges
        badge_html = ""
        if cfg.date_col:
            badge_html += (
                f'<span class="col-role-badge badge-date">📅 {cfg.date_col}</span>'
            )
        if cfg.bad_col:
            badge_html += (
                f'<span class="col-role-badge badge-bad">🎯 {cfg.bad_col}</span>'
            )
        for r in cfg.rule_cols[:6]:
            badge_html += f'<span class="col-role-badge badge-rule">⚖️ {r}</span>'
        if len(cfg.rule_cols) > 6:
            badge_html += f'<span class="col-role-badge badge-rule">…+{len(cfg.rule_cols) - 6} rules</span>'
        for c in cfg.cat_cols:
            badge_html += f'<span class="col-role-badge badge-cat">🏷️ {c}</span>'
        st.markdown(badge_html, unsafe_allow_html=True)

        st.markdown("---")

        # Override widgets
        new_date_col = st.selectbox(
            "📅 Date column",
            options=["(none)"] + all_cols,
            index=(["(none)"] + all_cols).index(cfg.date_col) if cfg.date_col else 0,
        )
        new_bad_col = st.selectbox(
            "🎯 Bad / Target flag column",
            options=["(none)"] + all_cols,
            index=(["(none)"] + all_cols).index(cfg.bad_col) if cfg.bad_col else 0,
        )
        new_rule_cols = st.multiselect(
            "⚖️ Rule indicator columns",
            options=all_cols,
            default=cfg.rule_cols,
        )
        new_cat_cols = st.multiselect(
            "🏷️ Categorical filter columns",
            options=[c for c in all_cols if c not in new_rule_cols],
            default=[c for c in cfg.cat_cols if c not in new_rule_cols],
        )

        if st.button("✅ Apply Column Config"):
            new_cfg = ColumnConfig(
                date_col=None if new_date_col == "(none)" else new_date_col,
                bad_col=None if new_bad_col == "(none)" else new_bad_col,
                rule_cols=new_rule_cols,
                cat_cols=new_cat_cols,
            )
            st.session_state.col_config = new_cfg
            cfg = new_cfg

            # Re-enrich date columns
            if cfg.date_col:
                st.session_state.df_enriched = enrich_date_cols(df_raw, cfg.date_col)
            else:
                st.session_state.df_enriched = df_raw

            # Reset rule grouping if rule_cols changed
            st.session_state.ungrouped = list(cfg.rule_cols)
            st.session_state.groups = {}
            st.session_state.group_order = []
            st.rerun()

    df_enriched = st.session_state.df_enriched

    # ── Date filters ──────────────────────────────────────────────────────
    st.markdown("### 📅 Date Filters")

    sel_years = []
    sel_quarters = []

    if cfg.date_col and "_year" in df_enriched.columns:
        all_years = sorted(df_enriched["_year"].dropna().unique().tolist())
        all_quarters = [1, 2, 3, 4]
        sel_years = st.multiselect(
            "Year", all_years, default=all_years, format_func=str
        )
        sel_quarters = st.multiselect(
            "Quarter", all_quarters, default=all_quarters, format_func=lambda q: f"Q{q}"
        )
    else:
        st.caption("_No date column configured._")

    # ── Categorical filters ───────────────────────────────────────────────
    cat_filter_values: dict = {}

    if cfg.cat_cols:
        st.markdown("### 🏷️ Categorical Filters")
        for col in cfg.cat_cols:
            if col not in df_raw.columns:
                continue
            vals = sorted(df_raw[col].dropna().unique().tolist())
            sel = st.multiselect(
                col.replace("_", " ").title(),
                options=vals,
                default=vals,
                key=f"cat_{col}",
            )
            cat_filter_values[col] = sel
    else:
        st.markdown("### 🏷️ Categorical Filters")
        st.caption(
            "_No categorical columns detected. Use Column Configuration to assign some._"
        )

    # ── Apply all filters ─────────────────────────────────────────────────
    df = filter_data(
        df_enriched,
        date_col=cfg.date_col,
        years=sel_years or None,
        quarters=sel_quarters or None,
        cat_filters=cat_filter_values or None,
    )
    st.markdown("---")
    st.caption(
        f"**{len(df):,}** of {len(df_raw):,} records after filters  \n"
        f"Bad col: `{cfg.bad_col or 'none'}` · Rule cols: `{len(cfg.rule_cols)}`"
    )

# ─────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────
st.markdown(
    """
<div style='padding:10px 0 24px'>
  <h1 style='color:#e2e8f0;font-size:2rem;font-weight:700;margin:0'>
    🌊 Rule Waterfall Analyzer
  </h1>
  <p style='color:#a0aec0;margin:4px 0 0'>
    Dynamically group rules, analyze declined applications, and measure bad rate by group.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Guard: need bad_col
# ─────────────────────────────────────────────
if not cfg.bad_col or cfg.bad_col not in df.columns:
    st.error(
        "⚠️ No Bad / Target flag column is configured. "
        "Open **Column Configuration** in the sidebar and select one."
    )
    st.stop()

rule_cols = cfg.rule_cols

# ─────────────────────────────────────────────
# Group Builder
# ─────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🗂️ Rule Group Builder</div>', unsafe_allow_html=True
)
st.markdown(
    "_Create groups, assign rules to them, and reorder groups. "
    "The waterfall is applied in the order shown below._"
)

# Sync ungrouped
all_assigned = [r for rules in st.session_state.groups.values() for r in rules]
st.session_state.ungrouped = [r for r in rule_cols if r not in all_assigned]

# Add new group
col_new1, col_new2 = st.columns([3, 1])
with col_new1:
    new_group_name = st.text_input(
        "New group name", placeholder="e.g. Credit Policy", label_visibility="collapsed"
    )
with col_new2:
    if st.button("➕ Add Group"):
        name = new_group_name.strip()
        if name and name not in st.session_state.groups:
            st.session_state.groups[name] = []
            st.session_state.group_order.append(name)
            st.rerun()
        elif name in st.session_state.groups:
            st.warning(f"Group '{name}' already exists.")

st.markdown("---")

if not st.session_state.group_order:
    st.info("👆 Add your first group above, then assign rules to it.")
else:
    for idx, grp in enumerate(st.session_state.group_order):
        g_col1, g_col2, g_col3, g_col4 = st.columns([0.5, 3.5, 1.2, 1.2])
        with g_col1:
            if idx > 0 and st.button("⬆", key=f"up_{grp}"):
                order = st.session_state.group_order
                order[idx], order[idx - 1] = order[idx - 1], order[idx]
                st.rerun()
            if idx < len(st.session_state.group_order) - 1 and st.button(
                "⬇", key=f"dn_{grp}"
            ):
                order = st.session_state.group_order
                order[idx], order[idx + 1] = order[idx + 1], order[idx]
                st.rerun()

        with g_col2:
            available = (
                st.session_state.groups.get(grp, []) + st.session_state.ungrouped
            )
            current = st.session_state.groups.get(grp, [])
            selected = st.multiselect(
                f"**{idx + 1}. {grp}**",
                options=available,
                default=current,
                key=f"ms_{grp}",
                placeholder="Select rules…",
            )
            if selected != current:
                st.session_state.groups[grp] = selected
                all_assigned = [
                    r for g, rules in st.session_state.groups.items() for r in rules
                ]
                st.session_state.ungrouped = [
                    r for r in rule_cols if r not in all_assigned
                ]
                st.rerun()

        with g_col3:
            st.markdown(
                f"<br><span style='color:#a0aec0;font-size:0.82rem'>{len(selected)} rules</span>",
                unsafe_allow_html=True,
            )
        with g_col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Delete", key=f"del_{grp}"):
                st.session_state.groups.pop(grp, None)
                st.session_state.group_order.remove(grp)
                all_assigned = [
                    r for g, rules in st.session_state.groups.items() for r in rules
                ]
                st.session_state.ungrouped = [
                    r for r in rule_cols if r not in all_assigned
                ]
                st.rerun()

        st.markdown(
            "<hr style='margin:4px 0;border-color:rgba(255,255,255,0.06)'>",
            unsafe_allow_html=True,
        )

# Quick-assign ungrouped
if st.session_state.ungrouped and st.session_state.group_order:
    with st.expander(
        f"⚡ Quick-assign {len(st.session_state.ungrouped)} ungrouped rules",
        expanded=False,
    ):
        target_grp = st.selectbox(
            "Assign all ungrouped rules to:", st.session_state.group_order
        )
        if st.button("Assign All"):
            st.session_state.groups[target_grp] = (
                st.session_state.groups.get(target_grp, []) + st.session_state.ungrouped
            )
            st.session_state.ungrouped = []
            st.rerun()

# Auto-group
with st.expander("🪄 Auto-group all rules evenly", expanded=False):
    n_auto = st.slider("Number of auto-groups", 2, 10, 5)
    if st.button("Create Auto Groups"):
        st.session_state.groups = {}
        st.session_state.group_order = []
        for i, chunk in enumerate(np.array_split(rule_cols, n_auto)):
            gname = f"Group {i + 1}"
            st.session_state.groups[gname] = list(chunk)
            st.session_state.group_order.append(gname)
        st.session_state.ungrouped = []
        st.rerun()

st.markdown("---")

# ─────────────────────────────────────────────
# Compute waterfall
# ─────────────────────────────────────────────
active_groups = {
    g: st.session_state.groups[g]
    for g in st.session_state.group_order
    if st.session_state.groups.get(g)
}

if not active_groups:
    st.warning("⬆️ Create at least one group and assign rules to see the waterfall.")
    st.stop()

with st.spinner("Computing waterfall…"):
    wf_df, summary = compute_waterfall(
        df, active_groups, st.session_state.group_order, bad_col=cfg.bad_col
    )

# ── KPI cards ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Summary</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
total = summary["total_apps"]
metrics = [
    (k1, "Total Applications", f"{total:,}", ""),
    (
        k2,
        "Total Declined",
        f"{summary['total_declined']:,}",
        f"{summary['total_declined'] / total * 100:.1f}% of total",
    ),
    (
        k3,
        "Total Approved",
        f"{summary['n_approved']:,}",
        f"{summary['n_approved'] / total * 100:.1f}% of total",
    ),
    (k4, "Overall Bad Rate", f"{summary['bad_rate_overall']:.2f}%", "Across all apps"),
    (
        k5,
        "Approved Bad Rate",
        f"{summary['bad_rate_approved']:.2f}%",
        "Among approved apps",
    ),
]
for col, label, value, sub in metrics:
    col.metric(
        label=label,
        value=value,
        delta=sub or None,
        delta_color="inverse" if "Bad" in label else "normal",
    )

st.markdown("---")

# ── Waterfall Chart ────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🌊 Waterfall: Applications Declined by Group</div>',
    unsafe_allow_html=True,
)

if not wf_df.empty:
    groups_ordered = list(wf_df["group"])
    declined_vals = list(wf_df["declined"])
    remaining_vals = list(wf_df["remaining"])

    x_labels = groups_ordered + ["✅ Approved"]
    measures = ["relative"] * len(groups_ordered) + ["total"]
    y_values = [-v for v in declined_vals] + [
        remaining_vals[-1] if remaining_vals else 0
    ]

    fig_wf = go.Figure(
        go.Waterfall(
            name="Applications",
            orientation="v",
            measure=measures,
            x=x_labels,
            y=y_values,
            connector={"line": {"color": "rgba(255,255,255,0.2)"}},
            decreasing={
                "marker": {
                    "color": "#FC8181",
                    "line": {"color": "#E53E3E", "width": 1.5},
                }
            },
            totals={
                "marker": {
                    "color": "#68D391",
                    "line": {"color": "#38A169", "width": 1.5},
                }
            },
            text=[f"-{v:,}" for v in declined_vals] + [f"{remaining_vals[-1]:,}"],
            textposition="outside",
        )
    )
    bad_rates = list(wf_df["bad_rate_pct"])
    fig_wf.add_trace(
        go.Scatter(
            x=groups_ordered,
            y=bad_rates,
            mode="lines+markers+text",
            name="Bad Rate % (declined)",
            yaxis="y2",
            line=dict(color="#F6E05E", width=2.5, dash="dot"),
            marker=dict(size=8, color="#F6E05E"),
            text=[f"{v:.1f}%" for v in bad_rates],
            textposition="top center",
            textfont=dict(color="#F6E05E", size=11),
        )
    )
    fig_wf.update_layout(
        yaxis=dict(
            title="Applications", color="#a0aec0", gridcolor="rgba(255,255,255,0.08)"
        ),
        yaxis2=dict(
            title="Bad Rate %",
            overlaying="y",
            side="right",
            color="#F6E05E",
            gridcolor="rgba(255,255,255,0.04)",
            rangemode="tozero",
        ),
        xaxis=dict(color="#a0aec0"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.3)",
        ),
        height=520,
        margin=dict(l=60, r=80, t=40, b=60),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

# ── Group detail table ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Group Detail</div>', unsafe_allow_html=True)
display_df = wf_df[
    [
        "group",
        "rule_count",
        "pool_before",
        "declined",
        "bad_declined",
        "good_declined",
        "bad_rate_pct",
        "pct_of_total",
        "remaining",
    ]
].copy()
display_df.columns = [
    "Group",
    "# Rules",
    "Pool Before",
    "Declined",
    "Bad Declined",
    "Good Declined",
    "Bad Rate % (declined)",
    "% of Total Apps",
    "Remaining",
]
st.dataframe(
    display_df.style.background_gradient(
        subset=["Bad Rate % (declined)"], cmap="RdYlGn_r"
    )
    .background_gradient(subset=["Declined"], cmap="Reds")
    .format(
        {
            "Bad Rate % (declined)": "{:.2f}%",
            "% of Total Apps": "{:.2f}%",
            "Pool Before": "{:,}",
            "Declined": "{:,}",
            "Remaining": "{:,}",
            "Bad Declined": "{:,}",
            "Good Declined": "{:,}",
        }
    ),
    use_container_width=True,
    height=min(50 + 40 * len(display_df), 420),
)

# ── Bad rate + donut ───────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">📉 Bad Rate % by Group</div>', unsafe_allow_html=True
)
col_br1, col_br2 = st.columns([2, 1])
with col_br1:
    fig_br = px.bar(
        wf_df,
        x="group",
        y="bad_rate_pct",
        color="bad_rate_pct",
        color_continuous_scale="RdYlGn_r",
        text=wf_df["bad_rate_pct"].map(lambda v: f"{v:.2f}%"),
        labels={"group": "Group", "bad_rate_pct": "Bad Rate %"},
    )
    fig_br.update_traces(textposition="outside")
    fig_br.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        coloraxis_showscale=False,
        xaxis=dict(color="#a0aec0"),
        yaxis=dict(color="#a0aec0", title="Bad Rate %"),
        height=380,
        margin=dict(l=40, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_br, use_container_width=True)

with col_br2:
    fig_donut = px.pie(
        wf_df,
        values="declined",
        names="group",
        hole=0.55,
        title="Share of Declined Apps",
        color_discrete_sequence=px.colors.sequential.Plasma_r,
    )
    fig_donut.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        title_font=dict(color="#e2e8f0"),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ── Rule-level detail ──────────────────────────────────────────────────────
with st.expander("🔍 Rule-level Statistics", expanded=False):
    with st.spinner("Computing rule stats…"):
        rls_df = rule_level_stats(df, rule_cols, bad_col=cfg.bad_col)
    fig_rules = px.scatter(
        rls_df,
        x="hit_rate_pct",
        y="bad_rate_pct",
        text="rule",
        size="hit_count",
        color="bad_rate_pct",
        color_continuous_scale="RdYlGn_r",
        labels={"hit_rate_pct": "Rule Hit Rate %", "bad_rate_pct": "Bad Rate %"},
        title="Rule Performance: Hit Rate vs Bad Rate",
    )
    fig_rules.update_traces(textposition="top center", textfont_size=9)
    fig_rules.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
        coloraxis_showscale=False,
        xaxis=dict(color="#a0aec0", gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(color="#a0aec0", gridcolor="rgba(255,255,255,0.07)"),
        height=500,
        margin=dict(l=40, r=40, t=50, b=60),
    )
    st.plotly_chart(fig_rules, use_container_width=True)
    st.dataframe(
        rls_df.style.background_gradient(
            subset=["bad_rate_pct"], cmap="RdYlGn_r"
        ).format(
            {"hit_rate_pct": "{:.2f}%", "bad_rate_pct": "{:.2f}%", "hit_count": "{:,}"}
        ),
        use_container_width=True,
        height=300,
    )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style='text-align:center;color:#4a5568;font-size:0.78rem;margin-top:32px;
padding-top:16px;border-top:1px solid rgba(255,255,255,0.06)'>
  Rule Waterfall Analyzer — Built with Streamlit & Plotly
</div>
""",
    unsafe_allow_html=True,
)
