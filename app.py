"""
Rule Waterfall Analyzer
=======================
Analyze the incremental decline impact of decision rules on applications.
Supports up to 200 rules × 3 million records with dynamic grouping UI.

Column roles (configurable in the ⚙️ Column Setup tab):
  • bad_flag    – binary 0/1 outcome variable (1 = bad account)
  • rule        – binary 0/1 decision rule indicator
  • date        – raw datetime column; not used directly for filtering
  • categorical – any column used as a sidebar filter (year, quarter, product, region…)
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

ROLE_OPTIONS   = ["rule", "bad_flag", "date", "categorical", "ignore"]
ROLE_HELP      = (
    "**rule** — binary 0/1 decision-rule indicator  \n"
    "**bad_flag** — binary 0/1 outcome / target variable  \n"
    "**date** — raw datetime column (not filtered directly)  \n"
    "**categorical** — low-cardinality column shown as a sidebar filter  \n"
    "**ignore** — excluded from all analysis"
)

# Keywords that identify special column types
_BAD_NAMES  = {"bad_flag", "bad", "target", "label", "default", "is_bad",
               "outcome", "delinquent", "charged_off", "event", "default_flag"}
_DATE_SUBS  = {"date", "_dt", "dt_", "timestamp", "time", "period",
               "vintage", "booking", "origination"}
_DATE_PARTS = {"year", "quarter", "month", "week", "day"}   # derived ints → categorical


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
    """Derive typed column lists from a role config dict."""
    return {
        "bad_flag":    next((c for c, r in col_config.items() if r == "bad_flag"), None),
        "date":        next((c for c, r in col_config.items() if r == "date"),     None),
        "rules":       sorted(c for c, r in col_config.items() if r == "rule"),
        "categoricals": [c for c, r in col_config.items() if r == "categorical"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Auto-detect column roles
# ──────────────────────────────────────────────────────────────────────────────
def auto_detect_roles(df: pd.DataFrame) -> dict[str, str]:
    """
    Heuristic role assignment — users can override in the Column Setup tab.

    Priority order:
      1. datetime dtype                          → date
      2. binary 0/1 + name matches bad_flag set → bad_flag
      3. name is a date-part (year/quarter/…)
         AND low-cardinality integer             → categorical   (not a rule!)
      4. binary 0/1                             → rule
      5. object / string, ≤ 50 unique values    → categorical
      6. integer 3–30 unique values             → categorical
      7. everything else                        → ignore
    """
    roles: dict[str, str] = {}

    for col in df.columns:
        series  = df[col].dropna()
        n_uniq  = int(df[col].nunique())
        dtype   = df[col].dtype
        name_l  = col.lower()

        is_dt   = pd.api.types.is_datetime64_any_dtype(dtype)
        is_int  = pd.api.types.is_integer_dtype(dtype)
        is_obj  = pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
        is_bin  = is_int and n_uniq <= 2 and (len(series) == 0 or series.isin([0, 1]).all())

        if is_dt:
            roles[col] = "date"

        elif any(sub in name_l for sub in _DATE_SUBS) and not is_int:
            # String/object column with a date-y name (e.g. "app_date" as string)
            roles[col] = "date"

        elif is_bin and name_l in _BAD_NAMES:
            roles[col] = "bad_flag"

        elif name_l in _DATE_PARTS and is_int and n_uniq <= 20:
            # year, quarter, month, day stored as integers → filter, not a rule
            roles[col] = "categorical"

        elif is_bin:
            # Default for binary columns that aren't bad_flag or date-parts
            roles[col] = "rule"

        elif is_obj and n_uniq <= 50:
            roles[col] = "categorical"

        elif is_int and 2 < n_uniq <= 30:
            roles[col] = "categorical"

        else:
            roles[col] = "ignore"

    return roles


# ──────────────────────────────────────────────────────────────────────────────
# Dummy data generation
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_dummy_data(n_records: int, n_rules: int, seed: int = 42) -> pd.DataFrame:
    """
    Realistic synthetic dataset:
      app_date     – datetime
      bad_flag     – binary outcome (~25 % bad)
      rule_NNN     – binary rule flags (correlated with bad_flag)
      product_type – categorical: Personal Loan / Auto Loan / Credit Card / Mortgage
      channel      – categorical: Online / Branch / Mobile / Partner
      region       – categorical: Northeast / Southeast / Midwest / Southwest / West
      year         – int (derived from app_date) → categorical filter
      quarter      – int (derived from app_date) → categorical filter
    """
    rng = np.random.default_rng(seed)

    # Dates
    t0    = pd.Timestamp("2022-01-01").value
    t1    = pd.Timestamp("2024-12-31").value
    dates = pd.to_datetime(rng.integers(t0, t1, n_records))

    # Outcome
    bad = rng.binomial(1, 0.25, n_records).astype(np.int8)

    # Rules (binary, correlated with bad)
    rule_dict: dict[str, np.ndarray] = {}
    for i in range(1, n_rules + 1):
        p_bad  = float(rng.uniform(0.08, 0.55))
        p_good = p_bad * float(rng.uniform(0.10, 0.45))
        probs  = np.where(bad == 1, p_bad, p_good)
        rule_dict[f"rule_{i:03d}"] = rng.binomial(1, probs).astype(np.int8)

    # Categorical columns
    product_types = rng.choice(
        ["Personal Loan", "Auto Loan", "Credit Card", "Mortgage"],
        size=n_records, p=[0.35, 0.25, 0.30, 0.10],
    )
    channels = rng.choice(
        ["Online", "Branch", "Mobile", "Partner"],
        size=n_records, p=[0.40, 0.20, 0.30, 0.10],
    )
    regions = rng.choice(
        ["Northeast", "Southeast", "Midwest", "Southwest", "West"],
        size=n_records,
    )

    df = pd.DataFrame(
        {
            "app_date":     dates,
            "bad_flag":     bad,
            "product_type": product_types,
            "channel":      channels,
            "region":       regions,
            **rule_dict,
        }
    )
    df["year"]    = df["app_date"].dt.year.astype(np.int16)
    df["quarter"] = df["app_date"].dt.quarter.astype(np.int8)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Waterfall logic
# ──────────────────────────────────────────────────────────────────────────────
def compute_waterfall(
    df: pd.DataFrame,
    group_order: list[str],
    groups: dict[str, list[str]],
    bad_flag_col: str,
) -> pd.DataFrame:
    """
    Incremental waterfall: each group captures only apps not already declined
    by a previous group.
    bad_rate % = bad accounts in bucket / total accounts in bucket × 100
    """
    n   = len(df)
    bad = df[bad_flag_col].to_numpy(dtype=np.int8)
    already_declined = np.zeros(n, dtype=bool)
    rows: list[dict] = []
    cumulative = 0

    for gname in group_order:
        rules = groups.get(gname, [])
        if not rules:
            continue

        fired       = df[rules].to_numpy(dtype=np.int8).any(axis=1)
        incremental = fired & ~already_declined
        already_declined |= fired

        cnt      = int(incremental.sum())
        bad_cnt  = int(bad[incremental].sum())
        bad_rate = bad_cnt / cnt * 100 if cnt else 0.0
        cumulative += cnt

        rows.append(
            {
                "Group":                gname,
                "# Rules":              len(rules),
                "Incremental Declined": cnt,
                "Cumulative Declined":  cumulative,
                "Bad Count":            bad_cnt,
                "Bad Rate %":           round(bad_rate, 2),
                "Decline Rate %":       round(cnt / n * 100, 3),
            }
        )

    passed     = int((~already_declined).sum())
    passed_bad = int(bad[~already_declined].sum())
    rows.append(
        {
            "Group":                "✅ Passed All Rules",
            "# Rules":              "—",
            "Incremental Declined": passed,
            "Cumulative Declined":  n,
            "Bad Count":            passed_bad,
            "Bad Rate %":           round(passed_bad / passed * 100 if passed else 0, 2),
            "Decline Rate %":       round(passed / n * 100, 3),
        }
    )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────────────────────────
def _bar_colors(bad_rates: list[float]) -> list[str]:
    return ["#c62828" if r >= 50 else "#ef6c00" if r >= 30 else "#2e7d32" for r in bad_rates]


def build_waterfall_chart(result: pd.DataFrame, total: int) -> go.Figure:
    groups    = result["Group"].tolist()
    counts    = result["Incremental Declined"].tolist()
    bad_rates = result["Bad Rate %"].tolist()

    x_labels = ["Total"] + groups[:-1] + ["✅ Passed"]
    measures  = ["absolute"] + ["relative"] * (len(groups) - 1) + ["absolute"]
    wf_y      = [total] + [-c for c in counts[:-1]] + [counts[-1]]
    wf_text   = [f"{total:,}"] + [f"−{c:,}" for c in counts[:-1]] + [f"{counts[-1]:,}"]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.60, 0.40],
        vertical_spacing=0.13,
        subplot_titles=(
            "Applications Declined per Group — incremental waterfall",
            "Bad Rate %  per Group  (bad accounts ÷ bucket size × 100)",
        ),
    )

    fig.add_trace(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=x_labels,
            y=wf_y,
            text=wf_text,
            textposition="outside",
            connector={"line": {"color": "#bdbdbd", "width": 1, "dash": "dot"}},
            increasing={"marker": {"color": "#1565c0"}},
            decreasing={"marker": {"color": "#c62828"}},
            totals={"marker": {"color": "#1565c0"}},
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=groups,
            y=bad_rates,
            text=[f"{r:.1f}%" for r in bad_rates],
            textposition="outside",
            marker_color=_bar_colors(bad_rates),
        ),
        row=2, col=1,
    )

    overall = result["Bad Count"].sum() / total * 100
    fig.add_hline(
        y=overall,
        line_dash="dash",
        line_color="#616161",
        annotation_text=f" Overall {overall:.1f}%",
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


def build_coverage_chart(df: pd.DataFrame, rule_cols: list[str]) -> go.Figure:
    cov = (df[rule_cols].sum() / len(df) * 100).sort_values(ascending=False).head(60)
    fig = go.Figure(
        go.Bar(
            x=cov.index, y=cov.values,
            marker_color="#5c6bc0",
            text=[f"{v:.1f}%" for v in cov.values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Rule Trigger Rate % — Top 60 (% of filtered records flagged)",
        xaxis_title="Rule", yaxis_title="% Triggered",
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
    if not name:
        return "Group name cannot be empty."
    if name in ss().groups:
        return f'A group named "{name}" already exists.'
    ss().groups[name] = list(rules)
    ss().group_order.append(name)
    for r in rules:
        if r in ss().ungrouped:
            ss().ungrouped.remove(r)
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
        if r not in ss().groups[name]:
            ss().groups[name].append(r)
        if r in ss().ungrouped:
            ss().ungrouped.remove(r)


def remove_rule_from_group(group: str, rule: str) -> None:
    if rule in ss().groups[group]:
        ss().groups[group].remove(rule)
    if rule not in ss().ungrouped:
        ss().ungrouped.append(rule)
        ss().ungrouped.sort()


def rename_group(old_name: str, new_name: str) -> str | None:
    new_name = new_name.strip()
    if not new_name:
        return "Name cannot be empty."
    if new_name == old_name:
        return None
    if new_name in ss().groups:
        return f'"{new_name}" already exists.'
    idx = ss().group_order.index(old_name)
    ss().group_order[idx] = new_name
    ss().groups[new_name] = ss().groups.pop(old_name)
    return None


def auto_split(n_groups: int) -> None:
    rules = list(ss().ungrouped)
    if not rules:
        return
    chunk = max(1, -(-len(rules) // n_groups))
    for i in range(0, len(rules), chunk):
        batch = rules[i : i + chunk]
        gname = base = f"Auto Group {i // chunk + 1}"
        sfx = 1
        while gname in ss().groups:
            gname = f"{base} ({sfx})"
            sfx += 1
        create_group(gname, batch)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: categorical filters
# ──────────────────────────────────────────────────────────────────────────────
def _render_sidebar_filters(df_all: pd.DataFrame, cat_cols: list[str]) -> None:
    """
    Render a multiselect for each categorical column.
    Selections are stored in ss() under key f"catf_{col}" — Streamlit persists
    them automatically across reruns via the widget key.
    """
    if not cat_cols:
        st.caption("No categorical columns detected. Assign the 'categorical' role in Column Setup.")
        return

    # Show year/quarter first if present, then rest alphabetically
    priority = [c for c in ["year", "quarter"] if c in cat_cols]
    others   = sorted(c for c in cat_cols if c not in priority)
    ordered  = priority + others

    for col in ordered:
        all_vals = sorted(df_all[col].dropna().unique().tolist(), key=lambda v: (str(v)))
        label    = col.replace("_", " ").title()

        # Special formatting for quarter
        fmt = (lambda q: f"Q{q}") if col == "quarter" else None

        # Initialise default (all selected) if not already in session state
        key = f"catf_{col}"
        if key not in ss():
            ss()[key] = all_vals

        if fmt:
            st.multiselect(label, all_vals, format_func=fmt, key=key)
        else:
            st.multiselect(label, all_vals, key=key)


def _apply_filters(df_all: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Apply all active categorical filter selections."""
    filt = pd.Series(True, index=df_all.index)
    for col in cat_cols:
        key = f"catf_{col}"
        selected = ss().get(key)
        if selected is not None and len(selected) < df_all[col].nunique():
            filt &= df_all[col].isin(selected)
    return df_all[filt].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Column Setup tab helper
# ──────────────────────────────────────────────────────────────────────────────
def _render_column_setup(df: pd.DataFrame) -> None:
    st.markdown("#### Column Role Configuration")
    st.markdown(
        "The app **auto-detected** a role for every column. "
        "Edit the **Role** column, then click **Apply** to confirm.\n\n"
        + ROLE_HELP
    )

    # Summary badges
    cfg = ss().col_config
    cg  = _col_groups(cfg)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rule columns",   len(cg["rules"]))
    m2.metric("Filter columns", len(cg["categoricals"]))
    m3.metric("Bad-flag col",   cg["bad_flag"] or "⚠️ None")
    m4.metric("Date col",       cg["date"]     or "None")
    m5.metric("Ignored",        sum(1 for r in cfg.values() if r == "ignore"))

    if not cg["bad_flag"]:
        st.error("No column is assigned the **bad_flag** role — bad-rate metrics will be unavailable.")

    st.divider()

    # Build editable table
    display = pd.DataFrame(
        [
            {
                "Column":        col,
                "Dtype":         str(df[col].dtype),
                "Unique Values": int(df[col].nunique()),
                "Sample":        ", ".join(str(v) for v in df[col].dropna().unique()[:4]),
                "Role":          role,
            }
            for col, role in cfg.items()
        ]
    )

    edited = st.data_editor(
        display,
        column_config={
            "Column":        st.column_config.TextColumn("Column",        disabled=True),
            "Dtype":         st.column_config.TextColumn("Dtype",         disabled=True),
            "Unique Values": st.column_config.NumberColumn("Unique Values", disabled=True),
            "Sample":        st.column_config.TextColumn("Sample Values", disabled=True),
            "Role": st.column_config.SelectboxColumn(
                "Role",
                options=ROLE_OPTIONS,
                required=True,
                help=ROLE_HELP,
            ),
        },
        hide_index=True,
        use_container_width=True,
        height=min(600, 55 + 35 * len(cfg)),
        key="col_setup_editor",
    )

    ca, cb, _ = st.columns([1, 1, 5])

    with ca:
        if st.button("✅ Apply Roles", type="primary", use_container_width=True):
            new_cfg = dict(zip(edited["Column"], edited["Role"]))

            old_cats  = {c for c, r in cfg.items() if r == "categorical"}
            new_cats  = {c for c, r in new_cfg.items() if r == "categorical"}
            old_rules = {c for c, r in cfg.items() if r == "rule"}
            new_rules = {c for c, r in new_cfg.items() if r == "rule"}

            # Drop filter state for columns that are no longer categorical
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
            # Clear all filter state
            for col in list(ss().keys()):
                if col.startswith("catf_"):
                    del ss()[col]
            ss().col_config = new_cfg
            _init_groups(sorted(c for c, r in new_cfg.items() if r == "rule"))
            st.toast("Auto-detect complete.", icon="🔍")
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("📊 Rule Waterfall Analyzer")
    st.caption("Incremental decline analysis · dynamic grouping · 200 rules × 3 M records")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Data Source")
        source = st.radio("source", ["🎲 Dummy Data", "📂 Upload CSV"], label_visibility="collapsed")
        st.divider()

        # ── Dummy data ────────────────────────────────────────────────────────
        if source == "🎲 Dummy Data":
            n_records = st.select_slider(
                "Records",
                options=[10_000, 50_000, 100_000, 500_000, 1_000_000, 3_000_000],
                value=100_000,
                format_func=lambda x: f"{x:,}",
            )
            n_rules = st.slider("Rules", min_value=5, max_value=200, value=20, step=5)

            if st.button("⚡ Generate Dataset", type="primary", use_container_width=True):
                with st.spinner(f"Generating {n_records:,} records…"):
                    df = generate_dummy_data(n_records, n_rules)
                ss().df         = df
                ss().col_config = auto_detect_roles(df)
                # Clear any stale filter state from a previous dataset
                for k in [k for k in ss() if k.startswith("catf_")]:
                    del ss()[k]
                _init_groups(_col_groups(ss().col_config)["rules"])
                ss().data_loaded = True
                st.success(f"Ready · {n_records:,} rows · {n_rules} rules")

        # ── CSV upload ────────────────────────────────────────────────────────
        else:
            uploaded = st.file_uploader("CSV file", type=["csv"])
            if uploaded:
                with st.spinner("Reading CSV…"):
                    df = pd.read_csv(uploaded, low_memory=False)
                    # Parse any datetime-looking columns
                    for col in df.columns:
                        if "date" in col.lower() or "time" in col.lower():
                            parsed = pd.to_datetime(df[col], errors="coerce")
                            if parsed.notna().mean() > 0.8:   # ≥ 80 % parseable
                                df[col] = parsed
                    # Derive year / quarter from the first datetime column if absent
                    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                    if dt_cols:
                        dc = dt_cols[0]
                        if "year"    not in df.columns:
                            df["year"]    = df[dc].dt.year.astype("Int16")
                        if "quarter" not in df.columns:
                            df["quarter"] = df[dc].dt.quarter.astype("Int8")

                ss().df         = df
                ss().col_config = auto_detect_roles(df)
                for k in [k for k in ss() if k.startswith("catf_")]:
                    del ss()[k]
                cg = _col_groups(ss().col_config)
                _init_groups(cg["rules"])
                ss().data_loaded = True
                st.success(
                    f"Loaded · {len(df):,} rows  \n"
                    f"{len(cg['rules'])} rules · "
                    f"{len(cg['categoricals'])} filter cols detected"
                )

        # ── Filters (categorical, dynamic) ────────────────────────────────────
        if ss().get("data_loaded"):
            st.divider()
            st.header("🔍 Filters")
            cg = _col_groups(ss().col_config)
            _render_sidebar_filters(ss().df, cg["categoricals"])

    # ── Guard ──────────────────────────────────────────────────────────────────
    if not ss().get("data_loaded"):
        st.info("👈  Configure a data source in the sidebar to begin.")
        st.markdown(
            """
### How it works
1. **Generate** dummy data (or upload your own CSV)
2. **Review** column roles in the **Column Setup** tab — the app auto-detects rules,
   bad-flag, dates and categorical filters; override any mistakes there
3. **Filter** by any categorical column in the sidebar
4. **Group** your rules in the **Rule Grouping** tab
5. **Analyze** the incremental waterfall in the **Waterfall Analysis** tab

#### CSV column requirements
| Role | Detection logic |
|------|----------------|
| `bad_flag` | Binary 0/1 column named: `bad_flag`, `target`, `label`, `default`, etc. |
| `rule` | Binary 0/1 column **not** matching bad_flag or date-part names |
| `date` | datetime64 dtype, or name contains `date`/`time`/`vintage`/… |
| `categorical` | String/object ≤ 50 unique values, OR int named `year`/`quarter`/`month` |
| `ignore` | Floats, high-cardinality ints, IDs, etc. |

All roles can be overridden in the **Column Setup** tab after loading data.
"""
        )
        return

    # ── Apply filters ──────────────────────────────────────────────────────────
    df_all: pd.DataFrame = ss().df
    cg                   = _col_groups(ss().col_config)
    fdf                  = _apply_filters(df_all, cg["categoricals"])

    bad_flag_col: str  = cg["bad_flag"] or ""
    all_rule_cols      = sorted(list(ss().ungrouped) + [r for g in ss().groups.values() for r in g])

    has_bad = bool(bad_flag_col and bad_flag_col in fdf.columns)

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
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Filtered Records", f"{len(fdf):,}")
        k2.metric("Rule Columns",     len(all_rule_cols))
        if has_bad:
            k3.metric("Bad Accounts", f"{int(fdf[bad_flag_col].sum()):,}")
            k4.metric("Bad Rate",     f"{fdf[bad_flag_col].mean() * 100:.2f}%")
        else:
            k3.metric("Bad Accounts", "—")
            k4.metric("Bad Rate",     "—")

        # Active filter summary
        active = [
            f"**{col.replace('_',' ').title()}**: {', '.join(str(v) for v in ss()[f'catf_{col}'])}"
            for col in cg["categoricals"]
            if f"catf_{col}" in ss() and len(ss()[f"catf_{col}"]) < df_all[col].nunique()
        ]
        if active:
            st.info("Active filters — " + "  ·  ".join(active))

        st.dataframe(fdf.head(2_000), use_container_width=True, height=340)

        if all_rule_cols:
            st.plotly_chart(build_coverage_chart(fdf, all_rule_cols), use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 3 — Rule Grouping
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
                if err:
                    st.error(err)
                else:
                    st.rerun()

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
            st.text_area(
                "", "\n".join(ss().ungrouped) if ss().ungrouped else "(none)",
                height=560, disabled=True, label_visibility="collapsed",
            )

        with left:
            n_grps = len(ss().group_order)
            st.markdown(f"**Groups in waterfall order ({n_grps})** — ⬆ ⬇ reorder · 🗑 delete")

            if not ss().group_order:
                st.info("No groups yet — create one above or use a bulk utility.")

            for idx, gname in enumerate(list(ss().group_order)):
                rules_here = ss().groups.get(gname, [])
                n_r = len(rules_here)
                with st.expander(f"**{idx + 1}.  {gname}**  ·  {n_r} rule{'s' if n_r != 1 else ''}", expanded=True):
                    ab1, ab2, ab3, ab4, _ = st.columns([1, 1, 1, 2, 4])
                    if ab1.button("⬆", key=f"up_{idx}", disabled=(idx == 0)):
                        move_group(gname, -1); st.rerun()
                    if ab2.button("⬇", key=f"dn_{idx}", disabled=(idx == n_grps - 1)):
                        move_group(gname, +1); st.rerun()
                    if ab3.button("🗑", key=f"del_{idx}"):
                        delete_group(gname); st.rerun()
                    with ab4:
                        new_gname = st.text_input(
                            "Rename", value=gname, key=f"ren_{idx}_{gname}",
                            label_visibility="collapsed",
                        )
                    if new_gname != gname:
                        err = rename_group(gname, new_gname)
                        if err: st.error(err)
                        else:   st.rerun()

                    if rules_here:
                        for rs in range(0, len(rules_here), 5):
                            chunk = rules_here[rs : rs + 5]
                            for col, rule in zip(st.columns(5), chunk):
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

        if not has_bad:
            st.error(
                "No **bad_flag** column is configured. "
                "Go to **Column Setup** and assign the `bad_flag` role to your outcome column."
            )
            st.stop()

        if ss().ungrouped:
            st.warning(
                f"⚠️  {len(ss().ungrouped)} rules are ungrouped and excluded from analysis. "
                "Use 'Group all remaining' in the Rule Grouping tab to include them."
            )

        with st.spinner("Computing waterfall…"):
            result = compute_waterfall(fdf, ss().group_order, ss().groups, bad_flag_col)

        total      = len(fdf)
        passed_row = result[result["Group"] == "✅ Passed All Rules"]
        passed_cnt = int(passed_row["Incremental Declined"].iloc[0]) if not passed_row.empty else 0
        declined   = total - passed_cnt

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Applications", f"{total:,}")
        k2.metric("Total Declined",     f"{declined:,}",    f"{declined / total * 100:.1f}%")
        k3.metric("Passed All Rules",   f"{passed_cnt:,}",  f"{passed_cnt / total * 100:.1f}%")
        k4.metric("Groups",             len(ss().group_order))
        k5.metric("Overall Bad Rate",   f"{fdf[bad_flag_col].mean() * 100:.2f}%")

        st.plotly_chart(build_waterfall_chart(result, total), use_container_width=True)

        st.subheader("Group-level Results")

        def _hl_passed(row):
            return ["background-color: #e8f5e9"] * len(row) if "Passed" in str(row["Group"]) else [""] * len(row)

        st.dataframe(
            result.style
                  .format({"Incremental Declined": "{:,.0f}", "Cumulative Declined": "{:,.0f}",
                            "Bad Count": "{:,.0f}", "Bad Rate %": "{:.2f}", "Decline Rate %": "{:.3f}"})
                  .apply(_hl_passed, axis=1)
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


if __name__ == "__main__":
    main()
