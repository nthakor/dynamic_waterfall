"""
Core analytics engine for the Rule Waterfall App.
Handles waterfall computation, bad rate calculation, and group-level aggregations.
Designed to be memory-efficient for up to 3M records.

Supports two data modes:
  - "row_level"  : each row = one application (original mode)
  - "aggregated" : each row = many applications; a count_col holds total apps per row
                   and bad_count_col holds bad apps per row.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Column role dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ColumnConfig:
    """Carries the user-confirmed role assignments for every column."""

    # Common roles
    date_col: Optional[str] = None  # datetime / date column
    rule_cols: List[str] = field(default_factory=list)  # binary rule indicators
    cat_cols: List[str] = field(default_factory=list)  # categorical filter cols
    ignored_cols: List[str] = field(default_factory=list)  # IDs, free-text, etc.

    # Row-level mode
    bad_col: Optional[str] = None  # binary 0/1 target flag

    # Aggregated mode
    data_mode: str = "row_level"  # "row_level" | "aggregated"
    count_col: Optional[str] = None  # total apps count per row
    bad_count_col: Optional[str] = None  # bad apps count per row


# ──────────────────────────────────────────────────────────────────────────────
# Keyword sets for heuristic detection
# ──────────────────────────────────────────────────────────────────────────────

_DATE_KEYWORDS = {
    "date",
    "time",
    "dt",
    "month",
    "year",
    "period",
    "vintage",
    "origination",
}
_BAD_KEYWORDS = {
    "bad",
    "default",
    "target",
    "chargoff",
    "charge_off",
    "delinq",
    "delinquent",
    "loss",
    "dpd",
    "outcome",
    "fail",
}
_RULE_KEYWORDS = {
    "rule",
    "policy",
    "flag",
    "indicator",
    "check",
    "filter",
    "criterion",
    "decline",
    "reject",
    "deny",
    "screen",
    "exclusion",
}
_COUNT_KEYWORDS = {
    "count",
    "cnt",
    "n_apps",
    "apps",
    "volume",
    "total",
    "pop",
    "population",
    "n_",
    "num",
    "application",
}
_BAD_CNT_KEYWORDS = {
    "bad",
    "default",
    "loss",
    "delinq",
    "cnt_bad",
    "n_bad",
    "cnt_def",
    "n_def",
    "bad_cnt",
    "def_cnt",
    "bad_count",
    "bad_n",
    "defaults",
    "losses",
}


# ──────────────────────────────────────────────────────────────────────────────
# Auto-detection helpers
# ──────────────────────────────────────────────────────────────────────────────


def _is_binary_01(series: pd.Series, sample: int = 50_000) -> bool:
    """True if column contains ONLY 0 and 1 (ignoring nulls)."""
    s = series.dropna()
    if len(s) == 0:
        return False
    if len(s) > sample:
        s = s.sample(sample, random_state=0)
    return set(s.unique()).issubset({0, 1, True, False, np.int8(0), np.int8(1)})


def _is_positive_integer_like(series: pd.Series, sample: int = 10_000) -> bool:
    """True if column looks like a non-negative integer count (no decimals, all >= 0)."""
    s = series.dropna()
    if len(s) == 0 or not pd.api.types.is_numeric_dtype(s):
        return False
    if len(s) > sample:
        s = s.sample(sample, random_state=0)
    return bool((s >= 0).all() and (s == s.astype(int)).all())


def detect_column_roles(df: pd.DataFrame) -> ColumnConfig:
    """
    Auto-detect role of each column using dtype + name heuristics.

    Priority order (first match wins per column):
      1. datetime dtype / date-keyword name      → date_col candidate
      2. Binary 0/1 + bad keyword                → bad_col candidate (row-level)
      3. Numeric non-binary + bad-count keyword  → bad_count_col candidate (aggregated)
      4. Numeric non-binary + count keyword      → count_col candidate (aggregated)
      5. Binary 0/1 (remaining)                  → rule_col
      6. object/category low-cardinality         → cat_col
      7. int/float low-cardinality (3-50 unique) → cat_col
      8. Everything else                         → ignored
    """
    date_candidates: List[str] = []
    bad_candidates: List[str] = []  # binary bad flag (row-level)
    bad_count_candidates: List[str] = []  # count of bad apps (aggregated)
    count_candidates: List[str] = []  # count of total apps (aggregated)
    rule_cols: List[str] = []
    cat_cols: List[str] = []
    ignored_cols: List[str] = []

    for col in df.columns:
        series = df[col]
        col_low = col.lower()

        # ── 1. Datetime ──────────────────────────────────────────────────────
        if pd.api.types.is_datetime64_any_dtype(series):
            date_candidates.append(col)
            continue

        if any(kw in col_low for kw in _DATE_KEYWORDS):
            if pd.api.types.is_object_dtype(series):
                try:
                    pd.to_datetime(series.dropna().iloc[:100])
                    date_candidates.append(col)
                    continue
                except Exception:
                    pass
            else:
                date_candidates.append(col)
                continue

        # ── 2. Binary 0/1 columns ────────────────────────────────────────────
        if pd.api.types.is_numeric_dtype(series) and _is_binary_01(series):
            if any(kw in col_low for kw in _BAD_KEYWORDS):
                bad_candidates.append(col)
            else:
                rule_cols.append(col)
            continue

        # ── 3 & 4. Non-binary numeric — count/bad-count candidates ───────────
        if pd.api.types.is_numeric_dtype(series) and not _is_binary_01(series):
            is_cnt = _is_positive_integer_like(series)
            if is_cnt and any(kw in col_low for kw in _BAD_CNT_KEYWORDS):
                bad_count_candidates.append(col)
                continue
            if is_cnt and any(kw in col_low for kw in _COUNT_KEYWORDS):
                count_candidates.append(col)
                continue
            # High-cardinality continuous numerics → ignored
            if series.nunique() > 50:
                ignored_cols.append(col)
            else:
                cat_cols.append(col)
            continue

        # ── 5. Categorical ───────────────────────────────────────────────────
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(
            series
        ):
            cat_cols.append(col) if series.nunique() <= 50 else ignored_cols.append(col)
            continue

        ignored_cols.append(col)

    # ── Pick best candidates ─────────────────────────────────────────────────
    date_col = date_candidates[0] if date_candidates else None
    bad_col = bad_candidates[0] if bad_candidates else None
    count_col = count_candidates[0] if count_candidates else None
    bad_count_col = bad_count_candidates[0] if bad_count_candidates else None

    # Remaining date candidates → ignored
    for extra in date_candidates[1:]:
        ignored_cols.append(extra)

    # ── Decide data mode ─────────────────────────────────────────────────────
    # Auto-select aggregated if we found count columns AND no binary bad flag
    if count_col and bad_count_col and not bad_col:
        data_mode = "aggregated"
    else:
        data_mode = "row_level"

    return ColumnConfig(
        date_col=date_col,
        bad_col=bad_col,
        rule_cols=sorted(rule_cols),
        cat_cols=cat_cols,
        ignored_cols=ignored_cols,
        data_mode=data_mode,
        count_col=count_col,
        bad_count_col=bad_count_col,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    """Load parquet dataset."""
    return pd.read_parquet(path, engine="pyarrow")


def enrich_date_cols(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add _year / _quarter helper columns from the configured date column."""
    df = df.copy()
    if date_col and date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df["_year"] = dt.dt.year.astype("Int16")
        df["_quarter"] = dt.dt.quarter.astype("Int8")
    return df


def filter_data(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    years: Optional[List[int]] = None,
    quarters: Optional[List[int]] = None,
    cat_filters: Optional[Dict[str, List]] = None,
) -> pd.DataFrame:
    """Apply date + categorical filters."""
    mask = pd.Series(True, index=df.index)
    if date_col and date_col in df.columns:
        if years:
            mask &= df["_year"].isin(years)
        if quarters:
            mask &= df["_quarter"].isin(quarters)
    if cat_filters:
        for col, values in cat_filters.items():
            if values and col in df.columns:
                mask &= df[col].isin(values)
    return df[mask]


# ──────────────────────────────────────────────────────────────────────────────
# Waterfall computation  (dual-mode)
# ──────────────────────────────────────────────────────────────────────────────


def compute_waterfall(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    group_order: List[str],
    # Row-level params
    bad_col: Optional[str] = None,
    # Aggregated params
    data_mode: str = "row_level",
    count_col: Optional[str] = None,
    bad_count_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Waterfall logic — dual mode:

    Row-level  : remaining_mask tracks individual rows.
                 declined  = count of rows where any rule=1 (among remaining)
                 bad_dec   = count of those rows where bad_col=1

    Aggregated : same mask semantics, but metrics use weighted sums:
                 declined  = sum(count_col)   where any rule=1 (among remaining)
                 bad_dec   = sum(bad_count_col) where any rule=1 (among remaining)
    """
    is_agg = (
        data_mode == "aggregated"
        and count_col
        and bad_count_col
        and count_col in df.columns
        and bad_count_col in df.columns
    )

    remaining_mask = pd.Series(True, index=df.index)
    rows: List[dict] = []

    # ── Totals ───────────────────────────────────────────────────────────────
    if is_agg:
        total_apps = int(df[count_col].sum())
        total_bad = int(df[bad_count_col].sum())
    else:
        total_apps = len(df)
        total_bad = int(df[bad_col].sum()) if bad_col and bad_col in df.columns else 0

    # ── Per-group pass ───────────────────────────────────────────────────────
    for grp in group_order:
        rules = groups.get(grp, [])
        valid_rules = [r for r in rules if r in df.columns]
        if not valid_rules:
            continue

        active_df = df[remaining_mask]
        declined_mask = active_df[valid_rules].any(axis=1)
        declined_df = active_df[declined_mask]

        if is_agg:
            n_declined = int(declined_df[count_col].sum())
            n_bad_dec = int(declined_df[bad_count_col].sum())
        else:
            n_declined = len(declined_df)
            n_bad_dec = (
                int(declined_df[bad_col].sum())
                if bad_col and bad_col in df.columns
                else 0
            )

        n_good_dec = n_declined - n_bad_dec
        bad_rate = (n_bad_dec / n_declined * 100) if n_declined > 0 else 0.0

        # Advance remaining pool
        remaining_mask = remaining_mask & ~declined_mask.reindex(
            df.index, fill_value=False
        )

        if is_agg:
            n_remaining = int(df.loc[remaining_mask, count_col].sum())
        else:
            n_remaining = int(remaining_mask.sum())

        rows.append(
            {
                "group": grp,
                "rules_in_group": ", ".join(valid_rules),
                "rule_count": len(valid_rules),
                "pool_before": (
                    int(active_df[count_col].sum())
                    if is_agg
                    else int(active_df.shape[0])
                ),
                "declined": n_declined,
                "bad_declined": n_bad_dec,
                "good_declined": n_good_dec,
                "bad_rate_pct": round(bad_rate, 2),
                "remaining": n_remaining,
            }
        )

    # ── Approved (remaining) ─────────────────────────────────────────────────
    remaining_df = df[remaining_mask]
    if is_agg:
        n_remaining = int(remaining_df[count_col].sum())
        n_bad_remain = int(remaining_df[bad_count_col].sum())
    else:
        n_remaining = len(remaining_df)
        n_bad_remain = (
            int(remaining_df[bad_col].sum()) if bad_col and bad_col in df.columns else 0
        )

    bad_rate_rem = (n_bad_remain / n_remaining * 100) if n_remaining > 0 else 0.0

    result = pd.DataFrame(rows)
    if not result.empty:
        result["cumulative_declined"] = result["declined"].cumsum()
        result["pct_of_total"] = (result["declined"] / total_apps * 100).round(2)

    summary = {
        "total_apps": total_apps,
        "total_bad": total_bad,
        "total_declined": int(result["declined"].sum()) if not result.empty else 0,
        "n_approved": n_remaining,
        "bad_rate_approved": round(bad_rate_rem, 2),
        "bad_rate_overall": round(total_bad / total_apps * 100, 2) if total_apps else 0,
        "data_mode": "aggregated" if is_agg else "row_level",
    }
    return result, summary


# ──────────────────────────────────────────────────────────────────────────────
# Rule-level stats  (dual-mode)
# ──────────────────────────────────────────────────────────────────────────────


def rule_level_stats(
    df: pd.DataFrame,
    rule_cols: List[str],
    # Row-level
    bad_col: Optional[str] = None,
    # Aggregated
    data_mode: str = "row_level",
    count_col: Optional[str] = None,
    bad_count_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute hit rate and bad rate for each rule. Supports both data modes."""
    is_agg = (
        data_mode == "aggregated"
        and count_col
        and bad_count_col
        and count_col in df.columns
        and bad_count_col in df.columns
    )
    total_wt = int(df[count_col].sum()) if is_agg else len(df)

    rows = []
    for col in rule_cols:
        if col not in df.columns:
            continue
        hit = df[col] == 1
        if is_agg:
            n_hit = int(df.loc[hit, count_col].sum())
            n_bad_hit = int(df.loc[hit, bad_count_col].sum())
        else:
            n_hit = int(hit.sum())
            n_bad_hit = (
                int((df.loc[hit, bad_col] == 1).sum())
                if bad_col and bad_col in df.columns
                else 0
            )

        br = (n_bad_hit / n_hit * 100) if n_hit > 0 else 0.0
        rows.append(
            {
                "rule": col,
                "hit_count": n_hit,
                "hit_rate_pct": round(n_hit / total_wt * 100, 2) if total_wt else 0,
                "bad_rate_pct": round(br, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("hit_count", ascending=False)
