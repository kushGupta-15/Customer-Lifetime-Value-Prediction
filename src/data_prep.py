"""Loading and cleaning for the UCI Online Retail II transaction data.

The cleaning sequence is fixed and order-dependent: dropping rows without a
`Customer ID` first removes ~23% of rows, which changes what the later steps
see (most non-product stock codes disappear at this step alone).
"""

from pathlib import Path

import pandas as pd

# Project paths, resolved relative to this file so notebooks work from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_EXCEL_PATH = DATA_DIR / "online_retail_II.xlsx"
CLEAN_TRANSACTIONS_PATH = DATA_DIR / "transactions_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Product codes in this dataset are a 5-digit number with an optional letter
# suffix (e.g. 85048, 79323P, 15056BL). Anything else is a candidate for being
# a non-product entry; every all-numeric code in the raw file is exactly 5 digits,
# so the fixed width is a real property of the data rather than a guess.
PRODUCT_CODE_PATTERN = r"\d{5}[A-Z]*"

# Verified non-product codes. Derived by running `summarize_stockcode_candidates`
# on the post-step-4 data and reading every candidate's Description: these are
# postage/carriage, manual adjustments, bank charges, discounts and test entries.
# Deliberately excluded from this list are SP1002 ("KID'S CHALKBOARD/EASEL") and
# PADS ("PADS TO MATCH ALL CUSHIONS"), which are real products with non-standard codes.
NON_PRODUCT_CODES = frozenset(
    {
        "POST",          # POSTAGE
        "DOT",           # DOTCOM POSTAGE
        "C2",            # CARRIAGE
        "M",             # Manual (also appears lowercase as "m")
        "D",             # Discount
        "ADJUST",        # Adjustment by john on ...
        "ADJUST2",       # Adjustment by Peter on ...
        "BANK CHARGES",  # Bank Charges
        "TEST001",       # This is a test product.
        "TEST002",       # This is a test product.
    }
)


def load_raw_transactions(path=RAW_EXCEL_PATH):
    """Load and concatenate both sheets of the Online Retail II workbook.

    Both sheets are used because the spec covers Dec 2009 - Dec 2011, and the
    2009-2010 sheet supplies the purchase history that makes tenure features
    meaningful for customers active near the cutoff.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. See README for download instructions "
            "(UCI ML Repository, dataset ID 502)."
        )
    # Invoice and StockCode are read as strings: both contain letters, and
    # letting pandas infer types would mangle codes like "47503J" or drop
    # the leading "C" that marks a cancellation.
    sheets = pd.read_excel(path, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    return pd.concat(sheets.values(), ignore_index=True)


def normalize_stock_code(series):
    """Strip whitespace and upper-case stock codes so equal codes compare equal.

    Needed because one raw code is stored as "47503J " and several use lowercase
    suffixes ("72349b"), which would otherwise look like distinct products and
    would wrongly fall outside the product-code pattern.
    """
    return series.astype(str).str.strip().str.upper()


def summarize_stockcode_candidates(df):
    """Return a per-code summary of stock codes that fail the product pattern.

    This is the programmatic step behind NON_PRODUCT_CODES: it surfaces every
    candidate with its descriptions and revenue so each can be judged as product
    or non-product, rather than trusting a hardcoded list.
    """
    code = normalize_stock_code(df["StockCode"])
    candidates = df.loc[~code.str.fullmatch(PRODUCT_CODE_PATTERN)].assign(stock_code=code)
    if candidates.empty:
        return pd.DataFrame(columns=["n_rows", "n_customers", "revenue", "descriptions"])

    revenue = candidates["Quantity"] * candidates["Price"]
    summary = (
        candidates.assign(revenue=revenue)
        .groupby("stock_code")
        .agg(
            n_rows=("Quantity", "size"),
            n_customers=("Customer ID", "nunique"),
            revenue=("revenue", "sum"),
            descriptions=(
                "Description",
                lambda s: " | ".join(sorted({str(x) for x in s.dropna()})[:3]),
            ),
        )
        .sort_values("revenue", ascending=False)
    )
    summary["removed"] = summary.index.isin(NON_PRODUCT_CODES)
    return summary


def clean_transactions(df):
    """Apply the cleaning steps in order.

    Returns (cleaned_df, log_df) where log_df records the row count remaining
    after each step, so the cost of every rule is visible rather than implied.
    """
    log = []

    def record(step, frame):
        log.append({"step": step, "rows": len(frame), "customers": _n_customers(frame)})

    record("0. raw (both sheets)", df)

    # 1. A customer-level target cannot be built from rows with no customer, and
    #    these cannot be recovered - they are guest/aggregated till transactions.
    df = df.dropna(subset=["Customer ID"])
    record("1. drop missing Customer ID", df)

    # 2. Exact duplicates are re-exported line items, not genuine repeat sales;
    #    keeping them would inflate frequency and monetary features.
    df = df.drop_duplicates()
    record("2. drop exact duplicates", df)

    # 3. Cancellations ("C" invoices) and non-positive quantities are returns,
    #    not purchases. They are dropped rather than netted off, because the
    #    target is gross future spend over a short 90-day window.
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    record("3. drop cancellations and Quantity <= 0", df)

    # 4. Zero and negative prices are administrative corrections, not sales.
    df = df[df["Price"] > 0]
    record("4. drop Price <= 0", df)

    # 5. Remove verified non-product codes (postage, adjustments, bank charges,
    #    tests) so monetary features reflect merchandise rather than fees.
    df = df.assign(StockCode=normalize_stock_code(df["StockCode"]))
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]
    record("5. drop non-product stock codes", df)

    # 6. High-value customers are deliberately retained: they are the customers
    #    the CLV model exists to find. Skew is absorbed by the log target and
    #    by tree models, so trimming them would remove the signal we want.
    record("6. keep high-value outliers (no rows dropped)", df)

    # 7. Types and the derived line-level revenue used by every monetary feature.
    df = df.assign(
        InvoiceDate=pd.to_datetime(df["InvoiceDate"]),
        **{"Customer ID": df["Customer ID"].astype("int64")},
    )
    df = df.assign(line_revenue=df["Quantity"] * df["Price"])
    record("7. type fixes and line_revenue", df)

    log_df = pd.DataFrame(log)
    log_df["rows_dropped"] = (-log_df["rows"].diff()).fillna(0).astype("int64")
    return df.reset_index(drop=True), log_df


def _n_customers(frame):
    """Distinct customer count, tolerating the pre-dropna stage where it is NaN."""
    if "Customer ID" not in frame:
        return 0
    return int(frame["Customer ID"].nunique())


def load_clean_transactions(path=CLEAN_TRANSACTIONS_PATH):
    """Read the cleaned transaction table written by notebook 01.

    Notebooks 02 and 03 use this instead of re-parsing the 45 MB workbook,
    which takes roughly 90 seconds per run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {path}. Run notebooks/01_cleaning_eda.ipynb first."
        )
    return pd.read_csv(path, parse_dates=["InvoiceDate"], dtype={"Invoice": str, "StockCode": str})
