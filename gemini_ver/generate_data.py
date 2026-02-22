"""
Dummy dataset generator for the Rule Waterfall App.
Generates a realistic dataset mimicking credit/application decisioning.
Supports up to 200 rules and millions of records.

Includes categorical columns so the app can demonstrate categorical filtering.
"""

import numpy as np
import pandas as pd
import os


def generate_dummy_dataset(
    n_records: int = 100_000,
    n_rules: int = 30,
    seed: int = 42,
    output_path: str = "data/applications.parquet",
) -> pd.DataFrame:
    """
    Generate a synthetic dataset of loan/application records with binary rule flags.

    Columns:
      - app_id        : unique application identifier
      - application_date : application date (for year/quarter filtering)
      - is_default    : 1 if the account is 'bad' (defaulted, etc.), 0 if good
      - product_type  : categorical — loan product category
      - channel       : categorical — origination channel
      - state         : categorical — US state (10 values)
      - risk_tier     : categorical — internal risk classification
      - rule_XXX      : binary 0/1 — 1 if rule XXX declined the application
    """
    np.random.seed(seed)

    print(f"Generating {n_records:,} records with {n_rules} rules...")

    # ── Date column ──────────────────────────────────────────────────────────
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2024-12-31")
    days = (end - start).days
    dates = start + pd.to_timedelta(
        np.random.randint(0, days, size=n_records), unit="D"
    )

    # ── Bad / target flag (overall bad rate ~15%) ────────────────────────────
    is_default = np.random.binomial(1, 0.15, size=n_records).astype(np.int8)

    # ── Categorical columns ──────────────────────────────────────────────────
    product_types = ["Personal Loan", "Credit Card", "Auto Loan", "Mortgage", "HELOC"]
    channels = ["Online", "Branch", "Broker", "Phone", "Partner"]
    states = ["CA", "TX", "FL", "NY", "IL", "OH", "GA", "NC", "PA", "AZ"]
    risk_tiers = ["Prime", "Near-Prime", "Sub-Prime", "Deep-Sub-Prime"]

    # Bad rate varies meaningfully by category (makes filters interesting)
    product_bad_rates = {
        "Personal Loan": 0.20,
        "Credit Card": 0.18,
        "Auto Loan": 0.12,
        "Mortgage": 0.05,
        "HELOC": 0.08,
    }
    channel_bad_rates = {
        "Online": 0.16,
        "Branch": 0.10,
        "Broker": 0.22,
        "Phone": 0.19,
        "Partner": 0.14,
    }
    tier_bad_rates = {
        "Prime": 0.04,
        "Near-Prime": 0.12,
        "Sub-Prime": 0.28,
        "Deep-Sub-Prime": 0.45,
    }

    product_arr = np.random.choice(product_types, size=n_records)
    channel_arr = np.random.choice(channels, size=n_records)
    state_arr = np.random.choice(states, size=n_records)
    risk_arr = np.random.choice(risk_tiers, size=n_records, p=[0.40, 0.30, 0.22, 0.08])

    # ── Rule flags ───────────────────────────────────────────────────────────
    base_rates = np.random.uniform(0.02, 0.25, size=n_rules)
    lift = np.random.uniform(1.5, 4.0, size=n_rules)

    rule_cols = {}
    for i in range(n_rules):
        p_good = base_rates[i]
        p_bad = min(base_rates[i] * lift[i], 0.98)
        prob = np.where(is_default == 1, p_bad, p_good)
        rule_cols[f"rule_{i + 1:03d}"] = np.random.binomial(1, prob).astype(np.int8)

    # ── Build DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "app_id": np.arange(1, n_records + 1, dtype=np.int32),
            "application_date": dates,
            "is_default": is_default,
            "product_type": product_arr,
            "channel": channel_arr,
            "state": state_arr,
            "risk_tier": risk_arr,
            **rule_cols,
        }
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    size_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"Dataset saved to '{output_path}'  ({size_mb:.1f} MB in memory)")
    return df


if __name__ == "__main__":
    df = generate_dummy_dataset(n_records=100_000, n_rules=30)
    print(df.head())
    print(f"\nBad rate: {df['is_default'].mean():.2%}")
    rule_cols = [c for c in df.columns if c.startswith("rule_")]
    print(f"Rule hit rates (sample):\n{df[rule_cols[:5]].mean()}")
    print(
        f"\nCategorical columns:\n{df[['product_type', 'channel', 'state', 'risk_tier']].nunique()}"
    )
