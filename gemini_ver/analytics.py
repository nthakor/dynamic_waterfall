"""
Core analytics engine for the Rule Waterfall App.
Handles waterfall computation, bad rate calculation, and group-level aggregations.
Designed to be memory-efficient for up to 3M records.

Key addition: detect_column_roles() auto-classifies every column by heuristic
so the app is not tied to any specific column naming convention.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Column role dataclass
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ColumnConfig:
    """Carries the user-confirmed role assignments for every column."""

    date_col: Optional[str] = None  # datetime / date column
    bad_col: Optional[str] = None  # binary target / bad flag
    rule_cols: List[str] = field(default_factory=list)  # binary rule indicators
    cat_cols: List[str] = field(default_factory=list)  # categorical filter cols
    ignored_cols: List[str] = field(default_factory=list)  # IDs, free-text, etc.


# ──────────────────────────────────────────────────────────────────────────────
# Auto-detection
# ──────────────────────────────────────────────────────────────────────────────

# Keywords used for heuristic matching (all lower-case)
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
    "flag",
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
_ID_KEYWORDS = {"id", "key", "uuid", "index", "idx", "number", "num", "ref", "code"}


def _is_binary_01(series: pd.Series, sample: int = 50_000) -> bool:
    """Return True if the column contains ONLY 0 and 1 values (ignoring nulls)."""
    s = series.dropna()
    if len(s) == 0:
        return False
    if len(s) > sample:
        s = s.sample(sample, random_state=0)
    unique = set(s.unique())
    return unique.issubset({0, 1, True, False, np.int8(0), np.int8(1)})


def detect_column_roles(df: pd.DataFrame) -> ColumnConfig:
    """
    Auto-detect the role of each column using dtype + name heuristics.

    Priority order (first match wins per column):
      1. datetime dtype  → date_col candidate
      2. Name matches date keywords + parseable → date_col candidate
      3. Binary 0/1 + name matches bad keywords → bad_col candidate
      4. Binary 0/1 + name matches rule keywords → rule_col
      5. Binary 0/1 (no keyword match) → rule_col (treat as unlabelled rule)
      6. object / category + low cardinality (≤ 50 unique) → cat_col
      7. int/float low cardinality (3–50 unique) → cat_col
      8. Everything else → ignored (IDs, free-text, continuous numerics)
    """
    date_candidates = []
    bad_candidates = []
    rule_cols = []
    cat_cols = []
    ignored_cols = []

    for col in df.columns:
        series = df[col]
        col_low = col.lower()

        # ── 1. Datetime dtype ───────────────────────────────────────────────
        if pd.api.types.is_datetime64_any_dtype(series):
            date_candidates.append(col)
            continue

        # ── 2. String that looks like a date ────────────────────────────────
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

        # ── 3 & 4 & 5. Binary columns ───────────────────────────────────────
        if pd.api.types.is_numeric_dtype(series) and _is_binary_01(series):
            if any(kw in col_low for kw in _BAD_KEYWORDS):
                bad_candidates.append(col)
            else:
                # Includes explicit rule keywords AND unnamed binary columns
                rule_cols.append(col)
            continue

        # ── 6. Categorical object/category ──────────────────────────────────
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(
            series
        ):
            n_unique = series.nunique()
            if n_unique <= 50:
                cat_cols.append(col)
            else:
                ignored_cols.append(col)
            continue

        # ── 7. Low-cardinality int/float → treat as categorical ─────────────
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique()
            if 3 <= n_unique <= 50:
                cat_cols.append(col)
            else:
                # High cardinality numeric → ID or continuous → ignore
                ignored_cols.append(col)
            continue

        ignored_cols.append(col)

    # Choose best single date / bad candidates
    date_col = date_candidates[0] if date_candidates else None
    bad_col = bad_candidates[0] if bad_candidates else None

    # Remaining date/bad candidates that weren't chosen → ignored
    for extra in date_candidates[1:]:
        ignored_cols.append(extra)

    return ColumnConfig(
        date_col=date_col,
        bad_col=bad_col,
        rule_cols=sorted(rule_cols),
        cat_cols=cat_cols,
        ignored_cols=ignored_cols,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    """Load parquet dataset with minimal footprint."""
    df = pd.read_parquet(path, engine="pyarrow")
    return df


def enrich_date_cols(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add year / quarter columns derived from the configured date column."""
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
    cat_filters: Optional[Dict[str, List]] = None,  # {col: [selected_values]}
) -> pd.DataFrame:
    """Apply date + categorical filters and return filtered DataFrame."""
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
# Waterfall computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_waterfall(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    group_order: List[str],
    bad_col: str = "is_bad",
) -> tuple:
    """
    Waterfall logic:
      - Start with all records as 'active pool'
      - For each group (in order), compute # applications declined by ANY rule
        in the group among those NOT already declined by a prior group
      - Track remaining apps after each group

    bad_col  — name of the binary 0/1 target column
    """
    remaining_mask = pd.Series(True, index=df.index)
    rows = []

    total_apps = len(df)
    total_bad = int(df[bad_col].sum()) if bad_col in df.columns else 0

    for grp in group_order:
        rules = groups.get(grp, [])
        if not rules:
            continue

        valid_rules = [r for r in rules if r in df.columns]
        if not valid_rules:
            continue

        active_df = df[remaining_mask]
        declined_mask = active_df[valid_rules].any(axis=1)

        declined_df = active_df[declined_mask]
        n_declined = len(declined_df)
        n_bad_dec = int(declined_df[bad_col].sum()) if bad_col in df.columns else 0
        n_good_dec = n_declined - n_bad_dec
        bad_rate = (n_bad_dec / n_declined * 100) if n_declined > 0 else 0.0

        # Advance the pool
        remaining_mask = remaining_mask & ~declined_mask.reindex(
            df.index, fill_value=False
        )
        n_remaining = int(remaining_mask.sum())

        rows.append(
            {
                "group": grp,
                "rules_in_group": ", ".join(valid_rules),
                "rule_count": len(valid_rules),
                "pool_before": int(active_df.shape[0]),
                "declined": n_declined,
                "bad_declined": n_bad_dec,
                "good_declined": n_good_dec,
                "bad_rate_pct": round(bad_rate, 2),
                "remaining": n_remaining,
            }
        )

    # Approved summary
    remaining_df = df[remaining_mask]
    n_remaining = len(remaining_df)
    n_bad_remain = int(remaining_df[bad_col].sum()) if bad_col in df.columns else 0
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
    }
    return result, summary


# ──────────────────────────────────────────────────────────────────────────────
# Rule-level stats
# ──────────────────────────────────────────────────────────────────────────────


def rule_level_stats(
    df: pd.DataFrame,
    rule_cols: List[str],
    bad_col: str = "is_bad",
) -> pd.DataFrame:
    """Compute hit rate and bad rate for each individual rule."""
    rows = []
    for col in rule_cols:
        if col not in df.columns:
            continue
        hit = df[col] == 1
        n_hit = int(hit.sum())
        n_bad_hit = (
            int((df.loc[hit, bad_col] == 1).sum()) if bad_col in df.columns else 0
        )
        br = (n_bad_hit / n_hit * 100) if n_hit > 0 else 0.0
        rows.append(
            {
                "rule": col,
                "hit_count": n_hit,
                "hit_rate_pct": round(n_hit / len(df) * 100, 2),
                "bad_rate_pct": round(br, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("hit_count", ascending=False)
