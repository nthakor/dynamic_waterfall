"""
Rule Waterfall Analyzer
=======================
Supports two data modes — set by assigning column roles in the ⚙️ Column Setup tab:

RECORD-LEVEL mode  (each row = 1 application)
  • bad_flag    – binary 0/1 outcome column (1 = bad account)
  Total apps  = row count
  Bad rate %  = sum(bad_flag) / count × 100

AGGREGATED mode  (each row = many applications, pre-summed)
  • count       – volume column; sum(count) = total applications in the bucket
  • bad_num     – bad-rate numerator;  bad rate = sum(bad_num) / sum(bad_denom) × 100
  • bad_denom   – bad-rate denominator (optional — falls back to count if unset)

Common roles (both modes):
  • rule        – binary 0/1 decision-rule indicator
  • date        – raw datetime column (not filtered directly)
  • categorical – low-cardinality column shown as a sidebar filter
  • ignore      – excluded from all analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# Page config & CSS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rule Waterfall Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="stExpander"] summary { font-weight: 600; background: #f7f8fa; border-radius: 6px; }
    div[data-testid="stHorizontalBlock"] .stButton > button {
        font-size: 0.75rem; padding: 2px 8px; height: auto; border-radius: 12px;
    }
    hr { margin: 0.6rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Role catalogue ─────────────────────────────────────────────────────────────
ROLE_OPTIONS = ["rule", "bad_flag", "count", "bad_num", "bad_denom", "date", "categorical", "ignore"]

ROLE_HELP = (
    "**rule** — binary 0/1 decision-rule indicator  \n"
    "**bad_flag** — *(record mode)* binary 0/1 outcome variable  \n"
    "**count** — *(aggregated mode)* volume column; sum = total applications  \n"
    "**bad_num** — *(aggregated mode)* bad-rate numerator (e.g. bad_count)  \n"
    "**bad_denom** — *(aggregated mode)* bad-rate denominator — defaults to count if unset  \n"
    "**date** — raw datetime column (used to derive year/quarter, not filtered directly)  \n"
    "**categorical** — low-cardinality column shown as a sidebar filter  \n"
    "**ignore** — excluded from all analysis"
)

# ── Detection keywords ─────────────────────────────────────────────────────────
_BAD_NAMES     = {"bad_flag", "bad", "target", "label", "default", "is_bad",
                  "outcome", "delinquent", "charged_off", "event", "default_flag"}
_DATE_SUBS     = {"date", "_dt", "dt_", "timestamp", "time", "period",
                  "vintage", "booking", "origination"}
_DATE_PARTS    = {"year", "quarter", "month", "week", "day"}

# Substring keywords for aggregated-mode columns  (checked via `kw in col_name`)
_BAD_NUM_KWS   = {"bad_count", "bad_n", "n_bad", "num_bad", "bad_apps",
                  "bad_cnt", "bad_vol", "n_defaults", "defaults"}
_BAD_DENOM_KWS = {"denom", "denominator", "bad_denom", "eligible_count", "eligible"}
_COUNT_KWS     = {"app_count", "n_apps", "num_apps", "total_count", "total_apps",
                  "volume", "freq", "weight", "obs", "n_obs"}
# broad fallback — only used if nothing more specific matched
_COUNT_BROAD   = {"count", "apps", "applications", "cnt"}


# ──────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ──────────────────────────────────────────────────────────────────────────────
def ss():
    return st.session_state


def _init_groups(rule_cols: list[str]) -> None:
    ss().group_order = []
    ss().groups      = {}
    ss().ungrouped   = sorted(rule_cols)


def _col_groups(col_config: dict[str, str]) -> dict:
    """Derive typed column lists from a role-config dict."""
    return {
        "bad_flag":     next((c for c, r in col_config.items() if r == "bad_flag"),  None),
        "date":         next((c for c, r in col_config.items() if r == "date"),       None),
        "count":        next((c for c, r in col_config.items() if r == "count"),      None),
        "bad_num":      next((c for c, r in col_config.items() if r == "bad_num"),    None),
        "bad_denom":    next((c for c, r in col_config.items() if r == "bad_denom"),  None),
        "rules":        sorted(c for c, r in col_config.items() if r == "rule"),
        "categoricals": [c for c, r in col_config.items() if r == "categorical"],
    }


def _data_mode(cg: dict) -> str:
    """'aggregated' when a count column is assigned, else 'record'."""
    return "aggregated" if cg.get("count") else "record"


# ──────────────────────────────────────────────────────────────────────────────
# Auto-detect column roles
# ──────────────────────────────────────────────────────────────────────────────
def auto_detect_roles(df: pd.DataFrame) -> dict[str, str]:
    """
    Priority order
    ─────────────
    1  datetime dtype                                   → date
    2  string column with date-y name                  → date
    3  binary 0/1 + name in bad-flag set               → bad_flag
    4  name is a date-part (year/quarter/…) + int ≤20u → categorical
    5  binary 0/1                                       → rule
    6  non-binary numeric + name matches bad_num kws   → bad_num
    7  non-binary numeric + name matches bad_denom kws → bad_denom
    8  non-binary numeric + name matches count kws     → count
    9  object/string ≤ 50 unique values                → categorical
    10 integer 2 < unique ≤ 30                         → categorical
    11 everything else                                  → ignore
    """
    roles: dict[str, str] = {}

    for col in df.columns:
        series = df[col].dropna()
        n_uniq = int(df[col].nunique())
        dtype  = df[col].dtype
        name_l = col.lower()

        is_dt  = pd.api.types.is_datetime64_any_dtype(dtype)
        is_int = pd.api.types.is_integer_dtype(dtype)
        is_flt = pd.api.types.is_float_dtype(dtype)
        is_obj = pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
        is_bin = is_int and n_uniq <= 2 and (len(series) == 0 or series.isin([0, 1]).all())

        if is_dt:
            roles[col] = "date"

        elif any(sub in name_l for sub in _DATE_SUBS) and not is_int:
            roles[col] = "date"

        elif is_bin and name_l in _BAD_NAMES:
            roles[col] = "bad_flag"

        elif name_l in _DATE_PARTS and is_int and n_uniq <= 20:
            roles[col] = "categorical"

        elif is_bin:
            roles[col] = "rule"

        elif is_int or is_flt:
            # Non-binary numeric — check aggregated-mode keywords first
            if any(kw in name_l for kw in _BAD_NUM_KWS):
                roles[col] = "bad_num"
            elif any(kw in name_l for kw in _BAD_DENOM_KWS):
                roles[col] = "bad_denom"
            elif any(kw in name_l for kw in _COUNT_KWS):
                roles[col] = "count"
            elif any(kw in name_l for kw in _COUNT_BROAD):
                roles[col] = "count"
            elif is_int and 2 < n_uniq <= 30:
                roles[col] = "categorical"
            else:
                roles[col] = "ignore"

        elif is_obj and n_uniq <= 50:
            roles[col] = "categorical"

        else:
            roles[col] = "ignore"

    return roles


# ──────────────────────────────────────────────────────────────────────────────
# Dummy data — record-level
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_dummy_data(n_records: int, n_rules: int, seed: int = 42) -> pd.DataFrame:
    """
    Each row = 1 application.
    Outcome column : bad_flag (binary)
    """
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp("2022-01-01").value, pd.Timestamp("2024-12-31").value
    dates   = pd.to_datetime(rng.integers(t0, t1, n_records))
    bad     = rng.binomial(1, 0.25, n_records).astype(np.int8)

    rule_dict: dict[str, np.ndarray] = {}
    for i in range(1, n_rules + 1):
        p_bad  = float(rng.uniform(0.08, 0.55))
        p_good = p_bad * float(rng.uniform(0.10, 0.45))
        rule_dict[f"rule_{i:03d}"] = rng.binomial(1, np.where(bad == 1, p_bad, p_good)).astype(np.int8)

    df = pd.DataFrame({
        "app_date":     dates,
        "bad_flag":     bad,
        "product_type": rng.choice(["Personal Loan", "Auto Loan", "Credit Card", "Mortgage"],
                                    n_records, p=[0.35, 0.25, 0.30, 0.10]),
        "channel":      rng.choice(["Online", "Branch", "Mobile", "Partner"],
                                    n_records, p=[0.40, 0.20, 0.30, 0.10]),
        "region":       rng.choice(["Northeast", "Southeast", "Midwest", "Southwest", "West"], n_records),
        **rule_dict,
    })
    df["year"]    = df["app_date"].dt.year.astype(np.int16)
    df["quarter"] = df["app_date"].dt.quarter.astype(np.int8)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Dummy data — aggregated
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_dummy_agg_data(n_segments: int, n_rules: int, seed: int = 42) -> pd.DataFrame:
    """
    Each row = a pre-aggregated segment (combination of dimensions + rule outcomes).

    app_count  → count role   : total applications in the segment
    bad_count  → bad_num role : bad applications in the segment
    bad rate   = sum(bad_count) / sum(app_count) × 100

    Rules are binary at segment level:
      1 = every app in this segment fired this rule
      0 = no app in this segment fired it
    Bad count is correlated with the number of rules that fired.
    """
    rng = np.random.default_rng(seed)

    product = rng.choice(["Personal Loan", "Auto Loan", "Credit Card", "Mortgage"],
                          n_segments, p=[0.35, 0.25, 0.30, 0.10])
    channel = rng.choice(["Online", "Branch", "Mobile", "Partner"],
                          n_segments, p=[0.40, 0.20, 0.30, 0.10])
    region  = rng.choice(["Northeast", "Southeast", "Midwest", "Southwest", "West"], n_segments)
    years   = rng.choice([2022, 2023, 2024], n_segments).astype(np.int16)
    qtrs    = rng.choice([1, 2, 3, 4],       n_segments).astype(np.int8)

    rule_dict: dict[str, np.ndarray] = {}
    for i in range(1, n_rules + 1):
        rule_dict[f"rule_{i:03d}"] = rng.binomial(1, 0.30, n_segments).astype(np.int8)

    app_count    = rng.integers(200, 15_000, n_segments)
    rules_matrix = np.column_stack(list(rule_dict.values()))           # (n_seg, n_rules)
    n_fired      = rules_matrix.sum(axis=1)
    bad_rate_seg = np.clip(0.05 + n_fired * (0.50 / max(n_rules, 1)), 0.02, 0.92)
    bad_count    = rng.binomial(app_count, bad_rate_seg)

    df = pd.DataFrame({
        "year":         years,
        "quarter":      qtrs,
        "product_type": product,
        "channel":      channel,
        "region":       region,
        "app_count":    app_count,
        "bad_count":    bad_count,
        **rule_dict,
    })
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Waterfall logic — handles both modes
# ──────────────────────────────────────────────────────────────────────────────
def compute_waterfall(
    df: pd.DataFrame,
    group_order: list[str],
    groups: dict[str, list[str]],
    *,
    # Record-level mode
    bad_flag_col: str | None = None,
    # Aggregated mode
    count_col: str    | None = None,
    bad_num_col: str  | None = None,
    bad_denom_col: str | None = None,   # defaults to count_col when None
) -> pd.DataFrame:
    """
    Incremental waterfall: each group captures only apps not already declined.

    Record mode   : cnt = rows; bad = sum(bad_flag_col)
    Aggregated mode: cnt = sum(count_col); bad rate = sum(bad_num_col)/sum(eff_denom_col)
    """
    is_agg  = count_col is not None
    n_rows  = len(df)
    eff_den = bad_denom_col or count_col   # effective denominator column

    total = float(df[count_col].sum()) if is_agg else float(n_rows)
    if not is_agg:
        bad_arr = df[bad_flag_col].to_numpy(dtype=np.int8)

    already_declined = np.zeros(n_rows, dtype=bool)
    rows: list[dict] = []
    cumulative = 0.0

    for gname in group_order:
        rules = groups.get(gname, [])
        if not rules:
            continue

        fired       = df[rules].to_numpy(dtype=np.int8).any(axis=1)
        incremental = fired & ~already_declined
        already_declined |= fired

        if is_agg:
            inc_df   = df[incremental]
            cnt      = float(inc_df[count_col].sum())
            bad_num  = float(inc_df[bad_num_col].sum())
            bad_den  = float(inc_df[eff_den].sum())
            bad_rate = bad_num / bad_den * 100 if bad_den > 0 else 0.0
        else:
            cnt      = float(incremental.sum())
            bad_num  = float(bad_arr[incremental].sum())
            bad_den  = cnt
            bad_rate = bad_num / cnt * 100 if cnt > 0 else 0.0

        cumulative += cnt
        rows.append({
            "Group":                gname,
            "# Rules":              len(rules),
            "Incremental Declined": cnt,
            "Cumulative Declined":  cumulative,
            "Bad Count":            bad_num,
            "Bad Rate %":           round(bad_rate, 2),
            "Decline Rate %":       round(cnt / total * 100, 3) if total > 0 else 0.0,
        })

    # Survivors
    surv_mask = ~already_declined
    if is_agg:
        surv_df    = df[surv_mask]
        passed     = float(surv_df[count_col].sum())
        surv_bad   = float(surv_df[bad_num_col].sum())
        surv_den   = float(surv_df[eff_den].sum())
        surv_rate  = surv_bad / surv_den * 100 if surv_den > 0 else 0.0
    else:
        passed     = float(surv_mask.sum())
        surv_bad   = float(bad_arr[surv_mask].sum())
        surv_rate  = surv_bad / passed * 100 if passed > 0 else 0.0

    rows.append({
        "Group":                "✅ Passed All Rules",
        "# Rules":              "—",
        "Incremental Declined": passed,
        "Cumulative Declined":  total,
        "Bad Count":            surv_bad,
        "Bad Rate %":           round(surv_rate, 2),
        "Decline Rate %":       round(passed / total * 100, 3) if total > 0 else 0.0,
    })
    return pd.DataFrame(rows)


def _overall_bad_rate(df: pd.DataFrame, cg: dict) -> float:
    """Compute overall bad rate consistently for both modes."""
    mode = _data_mode(cg)
    if mode == "aggregated":
        num = df[cg["bad_num"]].sum()
        den = df[cg["bad_denom"] or cg["count"]].sum()
        return float(num / den * 100) if den > 0 else 0.0
    else:
        col = cg["bad_flag"]
        return float(df[col].mean() * 100) if col and col in df.columns else 0.0


def _total_apps(df: pd.DataFrame, cg: dict) -> float:
    """Total application count for both modes."""
    return float(df[cg["count"]].sum()) if _data_mode(cg) == "aggregated" else float(len(df))


# ──────────────────────────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────────────────────────
def _bar_colors(bad_rates: list[float]) -> list[str]:
    return ["#c62828" if r >= 50 else "#ef6c00" if r >= 30 else "#2e7d32" for r in bad_rates]


def build_waterfall_chart(
    result: pd.DataFrame,
    total: float,
    overall_bad_rate: float,
) -> go.Figure:
    """Two-panel: waterfall (top) + bad-rate bars (bottom)."""
    groups    = result["Group"].tolist()
    counts    = result["Incremental Declined"].tolist()
    bad_rates = result["Bad Rate %"].tolist()

    x_labels = ["Total"] + groups[:-1] + ["✅ Passed"]
    measures  = ["absolute"] + ["relative"] * (len(groups) - 1) + ["absolute"]
    wf_y      = [total] + [-c for c in counts[:-1]] + [counts[-1]]

    def _fmt(v: float) -> str:
        if v >= 1_000_000:
            return f"{v/1_000_000:.2f}M"
        if v >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{int(v):,}"

    wf_text = [_fmt(total)] + [f"−{_fmt(c)}" for c in counts[:-1]] + [_fmt(counts[-1])]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.60, 0.40],
        vertical_spacing=0.13,
        subplot_titles=(
            "Applications Declined per Group — incremental waterfall",
            "Bad Rate %  per Group  (bad_num ÷ bad_denom × 100)",
        ),
    )

    fig.add_trace(
        go.Waterfall(
            orientation="v", measure=measures,
            x=x_labels, y=wf_y, text=wf_text, textposition="outside",
            connector={"line": {"color": "#bdbdbd", "width": 1, "dash": "dot"}},
            increasing={"marker": {"color": "#1565c0"}},
            decreasing={"marker": {"color": "#c62828"}},
            totals={"marker":    {"color": "#1565c0"}},
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=groups, y=bad_rates,
            text=[f"{r:.1f}%" for r in bad_rates],
            textposition="outside",
            marker_color=_bar_colors(bad_rates),
        ),
        row=2, col=1,
    )

    fig.add_hline(
        y=overall_bad_rate, line_dash="dash", line_color="#616161",
        annotation_text=f" Overall {overall_bad_rate:.1f}%",
        annotation_position="top right",
        row=2, col=1,
    )

    fig.update_layout(
        height=820, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        margin={"t": 80, "b": 20, "l": 60, "r": 40},
    )
    fig.update_yaxes(gridcolor="#eeeeee", gridwidth=1)
    fig.update_xaxes(tickangle=-35, tickfont={"size": 11})
    return fig


def build_coverage_chart(
    df: pd.DataFrame,
    rule_cols: list[str],
    count_col: str | None = None,
) -> go.Figure:
    """
    Rule trigger rate.
    Record mode   : fraction of rows where rule = 1
    Aggregated mode: volume-weighted fraction  sum(rule × count) / sum(count)
    """
    if count_col and count_col in df.columns:
        total_vol = df[count_col].sum()
        cov = (
            df[rule_cols].multiply(df[count_col], axis=0).sum() / total_vol * 100
        ).sort_values(ascending=False).head(60)
        title = "Rule Trigger Rate % — Top 60 (volume-weighted, % of total app_count)"
    else:
        cov = (df[rule_cols].sum() / len(df) * 100).sort_values(ascending=False).head(60)
        title = "Rule Trigger Rate % — Top 60 (% of filtered records flagged)"

    fig = go.Figure(go.Bar(
        x=cov.index, y=cov.values,
        marker_color="#5c6bc0",
        text=[f"{v:.1f}%" for v in cov.values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, xaxis_title="Rule", yaxis_title="% Triggered",
        height=370, plot_bgcolor="white", paper_bgcolor="white",
        margin={"t": 50, "b": 20},
    )
    fig.update_yaxes(gridcolor="#eeeeee")
    fig.update_xaxes(tickangle=-45, tickfont={"size": 9})
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Group management
# ──────────────────────────────────────────────────────────────────────────────
def create_group(name: str, rules: list[str]) -> str | None:
    name = name.strip()
    if not name:         return "Group name cannot be empty."
    if name in ss().groups: return f'A group named "{name}" already exists.'
    ss().groups[name] = list(rules)
    ss().group_order.append(name)
    for r in rules:
        if r in ss().ungrouped: ss().ungrouped.remove(r)
    return None


def delete_group(name: str) -> None:
    freed = ss().groups.pop(name, [])
    ss().group_order.remove(name)
    ss().ungrouped.extend(freed)
    ss().ungrouped.sort()


def move_group(name: str, delta: int) -> None:
    order = ss().group_order
    idx = order.index(name)
    new_idx = max(0, min(len(order) - 1, idx + delta))
    order[idx], order[new_idx] = order[new_idx], order[idx]


def add_rules_to_group(name: str, rules: list[str]) -> None:
    for r in rules:
        if r not in ss().groups[name]: ss().groups[name].append(r)
        if r in ss().ungrouped:        ss().ungrouped.remove(r)


def remove_rule_from_group(group: str, rule: str) -> None:
    if rule in ss().groups[group]: ss().groups[group].remove(rule)
    if rule not in ss().ungrouped:
        ss().ungrouped.append(rule)
        ss().ungrouped.sort()


def rename_group(old_name: str, new_name: str) -> str | None:
    new_name = new_name.strip()
    if not new_name:           return "Name cannot be empty."
    if new_name == old_name:   return None
    if new_name in ss().groups: return f'"{new_name}" already exists.'
    idx = ss().group_order.index(old_name)
    ss().group_order[idx] = new_name
    ss().groups[new_name] = ss().groups.pop(old_name)
    return None


def auto_split(n_groups: int) -> None:
    rules = list(ss().ungrouped)
    if not rules: return
    chunk = max(1, -(-len(rules) // n_groups))
    for i in range(0, len(rules), chunk):
        batch = rules[i : i + chunk]
        gname = base = f"Auto Group {i // chunk + 1}"
        sfx = 1
        while gname in ss().groups:
            gname = f"{base} ({sfx})"; sfx += 1
        create_group(gname, batch)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: categorical filters
# ──────────────────────────────────────────────────────────────────────────────
def _render_sidebar_filters(df_all: pd.DataFrame, cat_cols: list[str]) -> None:
    if not cat_cols:
        st.caption("No categorical columns detected. Assign 'categorical' role in Column Setup.")
        return
    priority = [c for c in ["year", "quarter"] if c in cat_cols]
    ordered  = priority + sorted(c for c in cat_cols if c not in priority)
    for col in ordered:
        all_vals = sorted(df_all[col].dropna().unique().tolist(), key=str)
        key      = f"catf_{col}"
        if key not in ss(): ss()[key] = all_vals
        label    = col.replace("_", " ").title()
        fmt      = (lambda q: f"Q{q}") if col == "quarter" else None
        if fmt:  st.multiselect(label, all_vals, format_func=fmt, key=key)
        else:    st.multiselect(label, all_vals, key=key)


def _apply_filters(df_all: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    filt = pd.Series(True, index=df_all.index)
    for col in cat_cols:
        selected = ss().get(f"catf_{col}")
        if selected is not None and len(selected) < df_all[col].nunique():
            filt &= df_all[col].isin(selected)
    return df_all[filt].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Column Setup tab
# ──────────────────────────────────────────────────────────────────────────────
def _render_column_setup(df: pd.DataFrame) -> None:
    st.markdown("#### Column Role Configuration")
    st.markdown(
        "The app **auto-detected** a role for every column. "
        "Edit the **Role** dropdown, then click **Apply** to confirm.\n\n" + ROLE_HELP
    )

    cfg  = ss().col_config
    cg   = _col_groups(cfg)
    mode = _data_mode(cg)

    # ── Mode badge + summary metrics ─────────────────────────────────────────
    if mode == "aggregated":
        st.success("🗂  **Aggregated mode** — total apps = sum(count column)")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Rule columns",    len(cg["rules"]))
        m2.metric("Filter columns",  len(cg["categoricals"]))
        m3.metric("Count col",       cg["count"]     or "⚠️ None")
        m4.metric("Bad Num col",     cg["bad_num"]   or "⚠️ None")
        m5.metric("Bad Denom col",   cg["bad_denom"] or f"= {cg['count']}")
        m6.metric("Ignored",         sum(1 for r in cfg.values() if r == "ignore"))
        if not cg["bad_num"]:
            st.error("No **bad_num** column assigned — bad-rate metrics unavailable.")
    else:
        st.info("📝  **Record-level mode** — each row = 1 application")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rule columns",   len(cg["rules"]))
        m2.metric("Filter columns", len(cg["categoricals"]))
        m3.metric("Bad-flag col",   cg["bad_flag"] or "⚠️ None")
        m4.metric("Date col",       cg["date"]     or "None")
        m5.metric("Ignored",        sum(1 for r in cfg.values() if r == "ignore"))
        if not cg["bad_flag"]:
            st.error("No **bad_flag** column assigned — bad-rate metrics unavailable.")

    st.divider()

    display = pd.DataFrame([
        {
            "Column":        col,
            "Dtype":         str(df[col].dtype),
            "Unique Values": int(df[col].nunique()),
            "Sample":        ", ".join(str(v) for v in df[col].dropna().unique()[:4]),
            "Role":          role,
        }
        for col, role in cfg.items()
    ])

    edited = st.data_editor(
        display,
        column_config={
            "Column":        st.column_config.TextColumn("Column",        disabled=True),
            "Dtype":         st.column_config.TextColumn("Dtype",         disabled=True),
            "Unique Values": st.column_config.NumberColumn("Unique Values", disabled=True),
            "Sample":        st.column_config.TextColumn("Sample Values", disabled=True),
            "Role":          st.column_config.SelectboxColumn(
                                 "Role", options=ROLE_OPTIONS, required=True, help=ROLE_HELP),
        },
        hide_index=True,
        use_container_width=True,
        height=min(600, 55 + 35 * len(cfg)),
        key="col_setup_editor",
    )

    ca, cb, _ = st.columns([1, 1, 5])
    with ca:
        if st.button("✅ Apply Roles", type="primary", use_container_width=True):
            new_cfg   = dict(zip(edited["Column"], edited["Role"]))
            old_cats  = {c for c, r in cfg.items() if r == "categorical"}
            new_cats  = {c for c, r in new_cfg.items() if r == "categorical"}
            old_rules = {c for c, r in cfg.items() if r == "rule"}
            new_rules = {c for c, r in new_cfg.items() if r == "rule"}
            for col in old_cats - new_cats:
                ss().pop(f"catf_{col}", None)
            ss().col_config = new_cfg
            if old_rules != new_rules:
                _init_groups(sorted(new_rules))
                st.toast("Rule columns changed — groups have been reset.", icon="⚠️")
            else:
                st.toast("Column roles applied.", icon="✅")
            st.rerun()

    with cb:
        if st.button("🔄 Re-run Auto-detect", use_container_width=True):
            new_cfg = auto_detect_roles(df)
            for k in [k for k in ss() if k.startswith("catf_")]:
                del ss()[k]
            ss().col_config = new_cfg
            _init_groups(sorted(c for c, r in new_cfg.items() if r == "rule"))
            st.toast("Auto-detect complete.", icon="🔍")
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("📊 Rule Waterfall Analyzer")
    st.caption("Incremental decline analysis · record-level & aggregated modes · 200 rules × 3 M records")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Data Source")
        source = st.radio(
            "source",
            ["🎲 Record-level Dummy Data", "🗂 Aggregated Dummy Data", "📂 Upload CSV"],
            label_visibility="collapsed",
        )
        st.divider()

        # ── Record-level dummy ────────────────────────────────────────────────
        if source == "🎲 Record-level Dummy Data":
            n_records = st.select_slider(
                "Records",
                options=[10_000, 50_000, 100_000, 500_000, 1_000_000, 3_000_000],
                value=100_000, format_func=lambda x: f"{x:,}",
            )
            n_rules = st.slider("Rules", 5, 200, 20, 5)
            if st.button("⚡ Generate", type="primary", use_container_width=True):
                with st.spinner(f"Generating {n_records:,} records…"):
                    df = generate_dummy_data(n_records, n_rules)
                _load_data(df)
                st.success(f"Ready · {n_records:,} rows · {n_rules} rules · record-level mode")

        # ── Aggregated dummy ──────────────────────────────────────────────────
        elif source == "🗂 Aggregated Dummy Data":
            n_segments = st.select_slider(
                "Segments (rows)",
                options=[500, 1_000, 5_000, 10_000, 50_000],
                value=1_000, format_func=lambda x: f"{x:,}",
            )
            n_rules = st.slider("Rules", 5, 200, 15, 5)
            st.caption(
                "Each row = one pre-aggregated segment.  \n"
                "`app_count` = volume · `bad_count` = bad apps"
            )
            if st.button("⚡ Generate", type="primary", use_container_width=True):
                with st.spinner("Generating aggregated data…"):
                    df = generate_dummy_agg_data(n_segments, n_rules)
                _load_data(df)
                cg = _col_groups(ss().col_config)
                total_apps = int(df[cg["count"]].sum())
                st.success(
                    f"Ready · {n_segments:,} segments · {total_apps:,} total apps · "
                    f"{n_rules} rules · aggregated mode"
                )

        # ── CSV upload ────────────────────────────────────────────────────────
        else:
            uploaded = st.file_uploader("CSV file", type=["csv"])
            if uploaded:
                with st.spinner("Reading CSV…"):
                    df = pd.read_csv(uploaded, low_memory=False)
                    for col in df.columns:
                        if "date" in col.lower() or "time" in col.lower():
                            parsed = pd.to_datetime(df[col], errors="coerce")
                            if parsed.notna().mean() > 0.8:
                                df[col] = parsed
                    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                    if dt_cols:
                        dc = dt_cols[0]
                        if "year"    not in df.columns: df["year"]    = df[dc].dt.year.astype("Int16")
                        if "quarter" not in df.columns: df["quarter"] = df[dc].dt.quarter.astype("Int8")
                _load_data(df)
                cg = _col_groups(ss().col_config)
                mode_label = "aggregated" if cg["count"] else "record-level"
                st.success(
                    f"Loaded · {len(df):,} rows · {len(cg['rules'])} rules · "
                    f"{len(cg['categoricals'])} filter cols · {mode_label} mode"
                )

        # ── Filters ───────────────────────────────────────────────────────────
        if ss().get("data_loaded"):
            st.divider()
            st.header("🔍 Filters")
            _render_sidebar_filters(ss().df, _col_groups(ss().col_config)["categoricals"])

    # ── Guard ──────────────────────────────────────────────────────────────────
    if not ss().get("data_loaded"):
        _render_landing()
        return

    # ── Resolve columns & filter ───────────────────────────────────────────────
    df_all: pd.DataFrame = ss().df
    cg                   = _col_groups(ss().col_config)
    mode                 = _data_mode(cg)
    fdf                  = _apply_filters(df_all, cg["categoricals"])
    all_rule_cols        = sorted(list(ss().ungrouped) + [r for g in ss().groups.values() for r in g])

    # Pre-compute scalars used across tabs
    total_apps      = _total_apps(fdf, cg)
    overall_br      = _overall_bad_rate(fdf, cg)
    has_bad_metric  = (mode == "record" and bool(cg["bad_flag"])) or \
                      (mode == "aggregated" and bool(cg["bad_num"]))

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_setup, tab_preview, tab_grouping, tab_analysis = st.tabs(
        ["⚙️  Column Setup", "📋  Data Preview", "🔧  Rule Grouping", "📈  Waterfall Analysis"]
    )

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1 — Column Setup
    # ════════════════════════════════════════════════════════════════════════════
    with tab_setup:
        _render_column_setup(df_all)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2 — Data Preview
    # ════════════════════════════════════════════════════════════════════════════
    with tab_preview:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Rows (filtered)", f"{len(fdf):,}")
        k2.metric("Rule Columns",    len(all_rule_cols))

        if mode == "aggregated":
            k3.metric("Total Apps (sum)",     f"{int(total_apps):,}")
            k4.metric("Bad Count (sum)",      f"{int(fdf[cg['bad_num']].sum()):,}" if cg["bad_num"] else "—")
            k5.metric("Bad Rate",             f"{overall_br:.2f}%" if has_bad_metric else "—")
        else:
            k3.metric("Total Apps",           f"{len(fdf):,}")
            k4.metric("Bad Accounts",         f"{int(fdf[cg['bad_flag']].sum()):,}" if cg["bad_flag"] else "—")
            k5.metric("Bad Rate",             f"{overall_br:.2f}%" if has_bad_metric else "—")

        # Active filter summary
        active = [
            f"**{c.replace('_',' ').title()}**: {', '.join(str(v) for v in ss()[f'catf_{c}'])}"
            for c in cg["categoricals"]
            if f"catf_{c}" in ss() and len(ss()[f"catf_{c}"]) < df_all[c].nunique()
        ]
        if active:
            st.info("Active filters — " + "  ·  ".join(active))

        st.dataframe(fdf.head(2_000), use_container_width=True, height=340)

        if all_rule_cols:
            st.plotly_chart(
                build_coverage_chart(fdf, all_rule_cols, count_col=cg["count"]),
                use_container_width=True,
            )

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 3 — Rule Grouping  (unchanged)
    # ════════════════════════════════════════════════════════════════════════════
    with tab_grouping:
        st.markdown(
            "#### Build your waterfall by grouping rules.  "
            "Groups fire **top → bottom**; an app is counted only in the first group that declines it."
        )
        with st.form("create_group", clear_on_submit=True):
            c1, c2, c3 = st.columns([2, 5, 1])
            new_name  = c1.text_input("Group Name", placeholder="e.g. Credit Score")
            new_rules = c2.multiselect("Rules to include", ss().ungrouped, placeholder="Select rules…")
            c3.write(""); c3.write("")
            if c3.form_submit_button("➕ Create", type="primary", use_container_width=True):
                err = create_group(new_name, new_rules)
                if err: st.error(err)
                else:   st.rerun()

        with st.expander("⚡ Bulk utilities", expanded=False):
            u1, u2, u3, u4 = st.columns(4)
            n_auto = u1.number_input("# groups", min_value=2, max_value=50, value=5)
            with u2:
                st.write("")
                if st.button("Split remaining equally", use_container_width=True):
                    auto_split(int(n_auto)); st.rerun()
            with u3:
                st.write("")
                if st.button("Group all remaining → 1", use_container_width=True):
                    err = create_group("Default Group", list(ss().ungrouped))
                    if err: st.error(err)
                    else:   st.rerun()
            with u4:
                st.write("")
                if st.button("🔄 Reset all groups", use_container_width=True, type="secondary"):
                    _init_groups(all_rule_cols); st.rerun()

        st.divider()
        left, right = st.columns([3, 1])

        with right:
            st.markdown(f"**Ungrouped ({len(ss().ungrouped)})**")
            st.text_area("", "\n".join(ss().ungrouped) or "(none)",
                         height=560, disabled=True, label_visibility="collapsed")

        with left:
            n_grps = len(ss().group_order)
            st.markdown(f"**Groups in waterfall order ({n_grps})** — ⬆ ⬇ reorder · 🗑 delete")
            if not ss().group_order:
                st.info("No groups yet — create one above or use a bulk utility.")

            for idx, gname in enumerate(list(ss().group_order)):
                rules_here = ss().groups.get(gname, [])
                n_r = len(rules_here)
                with st.expander(f"**{idx+1}.  {gname}**  ·  {n_r} rule{'s' if n_r!=1 else ''}", expanded=True):
                    ab1, ab2, ab3, ab4, _ = st.columns([1, 1, 1, 2, 4])
                    if ab1.button("⬆", key=f"up_{idx}", disabled=(idx==0)):
                        move_group(gname, -1); st.rerun()
                    if ab2.button("⬇", key=f"dn_{idx}", disabled=(idx==n_grps-1)):
                        move_group(gname, +1); st.rerun()
                    if ab3.button("🗑", key=f"del_{idx}"):
                        delete_group(gname); st.rerun()
                    with ab4:
                        new_gname = st.text_input("Rename", value=gname,
                                                   key=f"ren_{idx}_{gname}",
                                                   label_visibility="collapsed")
                    if new_gname != gname:
                        err = rename_group(gname, new_gname)
                        if err: st.error(err)
                        else:   st.rerun()

                    if rules_here:
                        for rs in range(0, len(rules_here), 5):
                            for col, rule in zip(st.columns(5), rules_here[rs:rs+5]):
                                if col.button(f"✕ {rule}", key=f"rm_{idx}_{gname}_{rule}"):
                                    remove_rule_from_group(gname, rule); st.rerun()
                    else:
                        st.caption("Empty group — add rules below or delete.")

                    if ss().ungrouped:
                        extra = st.multiselect("Add rules →", ss().ungrouped, key=f"add_{idx}_{gname}")
                        if st.button("Add selected", key=f"addbtn_{idx}_{gname}"):
                            add_rules_to_group(gname, extra); st.rerun()

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 4 — Waterfall Analysis
    # ════════════════════════════════════════════════════════════════════════════
    with tab_analysis:
        if not ss().group_order:
            st.info("👈  Create groups in the **Rule Grouping** tab first.")
            st.stop()

        if not has_bad_metric:
            col_needed = "bad_num (and count)" if mode == "aggregated" else "bad_flag"
            st.error(
                f"No **{col_needed}** column is configured for {mode} mode. "
                "Go to **Column Setup** and assign the correct role."
            )
            st.stop()

        if ss().ungrouped:
            st.warning(
                f"⚠️  {len(ss().ungrouped)} rules are ungrouped and excluded. "
                "Use 'Group all remaining' in the Rule Grouping tab to include them."
            )

        # ── Mode-specific label for the "unit" ───────────────────────────────
        unit = "apps" if mode == "aggregated" else "applications"

        with st.spinner("Computing waterfall…"):
            if mode == "aggregated":
                result = compute_waterfall(
                    fdf, ss().group_order, ss().groups,
                    count_col=cg["count"],
                    bad_num_col=cg["bad_num"],
                    bad_denom_col=cg["bad_denom"],
                )
            else:
                result = compute_waterfall(
                    fdf, ss().group_order, ss().groups,
                    bad_flag_col=cg["bad_flag"],
                )

        passed_row = result[result["Group"] == "✅ Passed All Rules"]
        passed_cnt = float(passed_row["Incremental Declined"].iloc[0]) if not passed_row.empty else 0.0
        declined   = total_apps - passed_cnt

        # ── KPI row ────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Applications",  f"{int(total_apps):,}")
        k2.metric(f"Total Declined {unit}",
                  f"{int(declined):,}",   f"{declined/total_apps*100:.1f}%" if total_apps else "—")
        k3.metric(f"Passed ({unit})",
                  f"{int(passed_cnt):,}", f"{passed_cnt/total_apps*100:.1f}%" if total_apps else "—")
        k4.metric("Groups",              len(ss().group_order))
        k5.metric("Overall Bad Rate",    f"{overall_br:.2f}%")

        if mode == "aggregated":
            st.caption(
                f"Bad rate = sum(**{cg['bad_num']}**) ÷ sum(**{cg['bad_denom'] or cg['count']}**) × 100"
            )

        st.plotly_chart(
            build_waterfall_chart(result, total_apps, overall_br),
            use_container_width=True,
        )

        st.subheader("Group-level Results")

        def _hl(row):
            return ["background-color: #e8f5e9"] * len(row) if "Passed" in str(row["Group"]) else [""] * len(row)

        st.dataframe(
            result.style
                  .format({"Incremental Declined": "{:,.0f}", "Cumulative Declined": "{:,.0f}",
                            "Bad Count": "{:,.0f}", "Bad Rate %": "{:.2f}", "Decline Rate %": "{:.3f}"})
                  .apply(_hl, axis=1)
                  .background_gradient(subset=["Bad Rate %"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )

        st.divider()
        st.subheader("Group Rules Reference")
        for gname in ss().group_order:
            rules = ss().groups.get(gname, [])
            with st.expander(f"{gname}  ({len(rules)} rules)"):
                st.write(", ".join(rules) if rules else "_empty_")

        st.download_button(
            "⬇️  Export Results as CSV",
            result.to_csv(index=False),
            file_name="waterfall_results.csv",
            mime="text/csv",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers used in main()
# ──────────────────────────────────────────────────────────────────────────────
def _load_data(df: pd.DataFrame) -> None:
    """Store df, auto-detect roles, clear stale state, init groups."""
    ss().df         = df
    ss().col_config = auto_detect_roles(df)
    for k in [k for k in ss() if k.startswith("catf_")]:
        del ss()[k]
    _init_groups(_col_groups(ss().col_config)["rules"])
    ss().data_loaded = True


def _render_landing() -> None:
    st.info("👈  Choose a data source in the sidebar to begin.")
    st.markdown(
        """
### How it works
1. **Load data** — generate dummy data (record-level or aggregated) or upload a CSV
2. **Review column roles** in the **Column Setup** tab — the app auto-detects everything; fix any mistakes
3. **Filter** by any categorical column in the sidebar
4. **Group** rules in the **Rule Grouping** tab
5. **Analyze** the incremental waterfall in the **Waterfall Analysis** tab

---

### Two supported data modes

| | Record-level | Aggregated |
|---|---|---|
| **Each row represents** | 1 application | Many applications |
| **Total apps** | row count | `sum(count)` |
| **Bad rate** | `mean(bad_flag)` | `sum(bad_num) / sum(bad_denom)` |
| **Key roles** | `bad_flag` | `count`, `bad_num`, `bad_denom` |

---

### Auto-detection logic (CSV upload)

| Role | Detection signal |
|---|---|
| `bad_flag` | Binary 0/1 + name in `{bad_flag, target, label, default, …}` |
| `count` | Non-binary int/float with name containing `app_count`, `volume`, `obs`, `freq`, … |
| `bad_num` | Non-binary int/float with name containing `bad_count`, `n_bad`, `defaults`, … |
| `bad_denom` | Non-binary int/float with name containing `denom`, `eligible`, … |
| `rule` | Binary 0/1 (not matched above) |
| `categorical` | String ≤ 50 unique values, or int named `year`/`quarter`/`month` |
| `date` | `datetime64` dtype, or name contains `date`/`time`/`vintage`/… |
| `ignore` | High-cardinality numerics, IDs, etc. |

All roles can be overridden in the **Column Setup** tab.
"""
    )


if __name__ == "__main__":
    main()
