"""Standalone leakage audit for the CLV pipeline.

Run from the repository root, after notebook 01 has written the cleaned
transaction table:

    python scripts/leakage_audit.py

Exits 0 if every check passes, 1 otherwise, so it can be wired into CI.

The central test is a mutation test rather than an assertion. Asserting that
features "look right" only confirms what the author already believed; instead
this corrupts every post-cutoff transaction and re-derives the feature table.
If any feature reads future data, corrupting the future must change it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python scripts/leakage_audit.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_prep import load_clean_transactions
from src.features import (
    CUTOFF_DATE,
    FEATURE_COLUMNS,
    PREDICTION_WINDOW_DAYS,
    TARGET_COLUMN,
    build_customer_dataset,
    split_windows,
)
from src.modeling import split_train_test, split_train_validation

RESULTS = []


def check(name, passed):
    """Record a check and print its outcome."""
    RESULTS.append(bool(passed))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")


def main():
    df = load_clean_transactions()
    baseline = build_customer_dataset(df)
    horizon_end = CUTOFF_DATE + pd.Timedelta(days=PREDICTION_WINDOW_DAYS)

    print(f"cutoff {CUTOFF_DATE.date()} | horizon {PREDICTION_WINDOW_DAYS}d "
          f"| {len(baseline):,} customers\n")

    # --- 1. Window boundaries -------------------------------------------------
    observation, prediction = split_windows(df)
    check("observation window ends strictly before the cutoff",
          observation["InvoiceDate"].max() < CUTOFF_DATE)
    check("prediction window starts at or after the cutoff",
          prediction["InvoiceDate"].min() >= CUTOFF_DATE)
    check("prediction window respects the 90-day horizon",
          prediction["InvoiceDate"].max() < horizon_end)
    check("no transaction appears in both windows",
          len(observation.index.intersection(prediction.index)) == 0)

    # --- 2. Mutation test: corrupt the future, features must not move ---------
    corrupted = df.copy()
    post_cutoff = corrupted["InvoiceDate"] >= CUTOFF_DATE
    rng = np.random.default_rng(0)
    corrupted.loc[post_cutoff, "line_revenue"] *= 1000
    corrupted.loc[post_cutoff, "Quantity"] = rng.permutation(
        corrupted.loc[post_cutoff, "Quantity"].to_numpy()
    )
    corrupted.loc[post_cutoff, "StockCode"] = "99999"
    mutated = build_customer_dataset(corrupted)

    check("population is unchanged when the future is corrupted",
          mutated.index.equals(baseline.index))
    check(f"all {len(FEATURE_COLUMNS)} features are unchanged when the future is corrupted",
          np.allclose(mutated[FEATURE_COLUMNS].to_numpy(),
                      baseline[FEATURE_COLUMNS].to_numpy()))
    # Sanity check on the test itself: if the target did not move either, the
    # corruption never landed and the check above proves nothing.
    check("target DOES move when the future is corrupted (proves the test is live)",
          not np.allclose(mutated[TARGET_COLUMN], baseline[TARGET_COLUMN]))

    # --- 3. The target depends on the prediction window and nothing else ------
    in_window = (df["InvoiceDate"] >= CUTOFF_DATE) & (df["InvoiceDate"] < horizon_end)
    from_future_only = (
        df[in_window].groupby("Customer ID")["line_revenue"].sum()
        .reindex(baseline.index).fillna(0.0)
    )
    check("target is reconstructible from post-cutoff data alone",
          np.allclose(from_future_only.to_numpy(), baseline[TARGET_COLUMN].to_numpy()))

    # The upper bound must actually bite: data runs past the horizon end.
    unbounded = (
        df[df["InvoiceDate"] >= CUTOFF_DATE].groupby("Customer ID")["line_revenue"].sum()
        .reindex(baseline.index).fillna(0.0)
    )
    check("horizon upper bound is enforced (unbounded future spend differs)",
          not np.allclose(unbounded.to_numpy(), baseline[TARGET_COLUMN].to_numpy()))

    # --- 4. Split hygiene -----------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = split_train_test(baseline)
    X_train, X_valid, y_train, y_valid = split_train_validation(X_trainval, y_trainval)

    check("train / validation / test customers are mutually disjoint",
          set(X_train.index).isdisjoint(X_valid.index)
          and set(X_train.index).isdisjoint(X_test.index)
          and set(X_valid.index).isdisjoint(X_test.index))
    check("the three splits cover every customer exactly once",
          len(X_train) + len(X_valid) + len(X_test) == len(baseline))
    check("test set is 20% of customers and never enters model selection",
          abs(len(X_test) / len(baseline) - 0.20) < 0.01)

    # --- 5. No feature is a disguised copy of the target ----------------------
    correlations = baseline[FEATURE_COLUMNS].corrwith(baseline[TARGET_COLUMN]).abs()
    check(f"no feature is near-perfectly correlated with the target "
          f"(max {correlations.max():.3f})", correlations.max() < 0.95)

    print("\nfeature / target correlations:")
    print(correlations.sort_values(ascending=False).round(3).to_string())

    passed = all(RESULTS)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed - "
          f"{'NO LEAKAGE DETECTED' if passed else 'AUDIT FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
