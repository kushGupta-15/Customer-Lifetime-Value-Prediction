# Customer Lifetime Value (CLV) Prediction

Predicting how much each existing customer will spend over the **next 90 days**, from their
transaction history alone, using classical machine learning on the UCI Online Retail II dataset.

## Project Highlights

- **Leakage-free temporal target construction** — a single cutoff date parameter separates the
  feature window from the prediction window, asserted in code rather than assumed
- **11 engineered customer-level features** across RFM, tenure/cadence and behavioural breadth
- **Comparison of four classical ML models** — Linear Regression, Random Forest, XGBoost, LightGBM
- **Optuna hyperparameter optimization** — 50 trials against 5-fold cross-validated MAE
- **SHAP explainability** — global beeswarm and a single-customer waterfall
- **Decile lift analysis** — the top predicted decile captures **54.2%** of realised 90-day revenue

---

## Problem

A retailer wants to know which of its existing customers are worth spending money on *now*.
Given every transaction up to **2011-09-01**, predict the total revenue each customer will
generate in the following **90 days**.

The target is continuous and non-negative, so this is framed as **regression**. Outputs support
tiered retention campaigns, acquisition budget caps, and identification of high-history /
low-forecast customers for win-back.

## Data

**UCI Online Retail II** — a UK online retailer, Dec 2009 to Dec 2011, ~1.07M transaction rows
across two Excel sheets (both are used).

The raw data is **not committed**. To download:

1. Fetch the dataset from the [UCI ML Repository, dataset ID 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
   — direct link: `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`
2. Unzip it and place `online_retail_II.xlsx` in `data/`.

```bash
curl -L -o data/online_retail_ii.zip https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
unzip data/online_retail_ii.zip -d data/ && rm data/online_retail_ii.zip
```

## Approach

### Target construction (leakage-free)

| Parameter | Value |
|---|---|
| Cutoff date | 2011-09-01 |
| Observation window | all valid transactions strictly before the cutoff |
| Prediction window | cutoff → cutoff + 90 days |
| Population | customers with ≥ 1 valid purchase before the cutoff (**5,224**) |
| Target | `future_clv` = Σ (Quantity × Price) in the prediction window |
| No-purchase customers | target = 0, retained (**57.2%** of the population) |

Models train on `log1p(future_clv)` and predictions are inverted with `expm1`, then clipped at
zero since future spend cannot be negative; **all metrics are reported on the original monetary
scale**. Features read only pre-cutoff data, and the scaler for the linear baseline is fit inside
a `Pipeline` on training folds only.

### Evaluation protocol

Each split has exactly one job:

| Split | Share | Customers | Used for |
|---|---|---:|---|
| **train** | 64% | 3,343 | fitting the candidates; 5-fold CV inside it drives Optuna |
| **validation** | 16% | 836 | choosing between the four candidates — nothing else |
| **test** | 20% | 1,045 | touched once, at the very end, on the final tuned model |

Model *selection* happens on validation, never on test. Choosing a model on the test set would
spend it: the winner would be whichever model happened to suit those particular customers, and the
final metric would stop being an honest estimate of unseen performance. Optuna likewise optimises
5-fold CV MAE **within the training split only**.

Once both decisions are settled — which architecture, which hyperparameters — the final model is
refit on **train + validation** together. Holding validation back at that point would waste data
for no remaining benefit. The test set is read exactly once, at the end.

### Cleaning

Applied in a fixed order, with the row count logged after each step — 1,067,371 → **776,596** rows
(72.8% retained), 5,852 customers.

| Step | Rows remaining | Dropped |
|---|---:|---:|
| Raw (both sheets) | 1,067,371 | — |
| 1. Drop missing `Customer ID` | 824,364 | 243,007 |
| 2. Drop exact duplicates | 797,885 | 26,479 |
| 3. Drop cancellations & `Quantity ≤ 0` | 779,495 | 18,390 |
| 4. Drop `Price ≤ 0` | 779,425 | 70 |
| 5. Drop non-product stock codes | 776,596 | 2,829 |
| 6. Keep high-value outliers | 776,596 | 0 |
| 7. Type fixes + `line_revenue` | 776,596 | 0 |

Non-product codes are found **programmatically** — every code failing the product pattern
(5 digits, optional letter suffix) is surfaced with its descriptions and revenue, then judged
individually. Removed: `POST`, `DOT`, `C2`, `M`, `D`, `ADJUST`, `ADJUST2`, `BANK CHARGES`,
`TEST001`, `TEST002` (postage, carriage, manual adjustments, bank charges, discounts, tests).
Deliberately **kept**: `SP1002` and `PADS`, which have non-standard codes but are real merchandise.

High-value customers are **not** removed — they are precisely what the model exists to find.

### The 11 features

| Group | Features |
|---|---|
| RFM | `recency_days`, `frequency`, `monetary_total`, `monetary_avg_order` |
| Tenure & temporal | `tenure_days`, `avg_days_between_orders`, `active_months`, `average_monthly_spend` |
| Behavioural | `n_unique_products`, `avg_basket_size`, `has_single_purchase` |

All computed exclusively from observation-window data.

---

## Results

### Step 1 — model selection, on the validation set

Four candidates at sensible defaults, trained on the train split and scored on **validation**
(the test set plays no part here):

| Model | MAE ↓ | RMSE | R² |
|---|---:|---:|---:|
| **Random Forest** | **491.48** | 3,704.94 | 0.271 |
| LightGBM | 507.71 | 3,709.06 | 0.269 |
| XGBoost | 527.05 | 3,557.83 | 0.328 |
| Linear Regression | 1.22 × 10¹⁰ | 3.53 × 10¹¹ | −6.6 × 10¹⁵ |

Random Forest wins on **MAE**, the selection criterion.

**Why the linear baseline explodes.** This is a real property of the model, not a bug. A linear
model extrapolates without bound: one validation customer has an `avg_basket_size` of 87,167
against a training maximum of 24,746 (99th percentile: 1,377), so the model predicts a log-target
of **29.95** when training values span only [0, 11.68]. `expm1(29.95) ≈ 1.0 × 10¹³`, and that one
customer dominates the mean error. Random Forest cannot do this — its largest prediction is
£23,074, because a forest can only average leaf values it actually saw. That is a concrete argument
for tree models on skewed commercial data with a log-transformed target, and it only surfaces on
held-out data.

Among the three tree models there is still a tension worth naming: the lowest-MAE model is not the
best on RMSE or R². Both of those square the error and are therefore dominated by a handful of very
large customers, while a model that hedges toward the mean scores well on them and worse for the
typical customer. MAE is the right criterion because the model ranks and budgets **per customer**.

### Step 2 — tuning, on the training split

Optuna, 50 trials, minimising 5-fold CV MAE **within the training split**. Best CV MAE: **386.55**.

Best parameters (`models/best_params.json`): `n_estimators=450`, `max_depth=22`,
`min_samples_leaf=3`, `min_samples_split=3`, `max_features=0.782`.

### Step 3 — final evaluation, on the test set (once)

Refit on train + validation (4,179 customers), then evaluated a single time. The untuned baseline
is refit on the same data, so the comparison isolates the hyperparameters rather than the amount of
training data:

| | MAE ↓ | RMSE | R² |
|---|---:|---:|---:|
| Random Forest (defaults) | 337.30 | 1,529.84 | 0.464 |
| **Random Forest (tuned)** | **332.13** | 1,427.95 | 0.533 |

**Context for the MAE.** With 57% of customers spending nothing, "predict zero for everyone" is a
genuinely competitive baseline — and because the target is non-negative, its MAE is exactly the
test-set mean spend, **£453.61**. The tuned model reaches **£332.13**, a **26.8%** improvement.

*On the gap between validation and test MAE:* validation MAE (491) is higher than test MAE (332)
because the two splits are not equally hard — mean actual spend is £625.97 in validation versus
£453.61 in test. Validation numbers are only ever compared against each other, so this does not
affect model selection; the test figure is the one that estimates real-world performance.

### Decile lift

Ranking test customers by predicted CLV and measuring what each decile actually spent:

| Decile | Mean predicted | Mean actual | % of realised revenue | Lift |
|---:|---:|---:|---:|---:|
| 1 | 1,640.34 | 2,446.47 | **54.2%** | 5.39× |
| 2 | 169.52 | 574.04 | 12.6% | 1.27× |
| 3 | 61.11 | 437.49 | 9.7% | 0.96× |
| 4 | 26.27 | 375.20 | 8.2% | 0.83× |
| 5 | 12.67 | 209.10 | 4.6% | 0.46× |
| 6–10 | ≤ 6.23 | ≤ 196.49 | 10.7% | ≤ 0.43× |

The ordering is **strictly monotonic** across all ten deciles. This is the result that matters
commercially: individual forecasts are imprecise, but the *ranking* is reliable, and targeting the
top 10% of customers reaches over half of all revenue at risk.

---

## Key figures

| Figure | What it shows |
|---|---|
| [`02_revenue_pareto.png`](reports/figures/02_revenue_pareto.png) | Top 10% of customers = 64% of revenue; top 20% = 77% |
| [`04_monthly_revenue.png`](reports/figures/04_monthly_revenue.png) | Monthly revenue with the cutoff and 90-day window marked |
| [`05_target_distribution.png`](reports/figures/05_target_distribution.png) | Target on raw and `log1p` scales |
| [`06_recency_by_future_spend.png`](reports/figures/06_recency_by_future_spend.png) | Return rate falls from 74.9% (0–30 days) to 10.5% (365+ days) |
| [`12_decile_lift.png`](reports/figures/12_decile_lift.png) | Actual spend and revenue share per predicted decile |
| [`13_shap_beeswarm.png`](reports/figures/13_shap_beeswarm.png) | Global feature impact on predicted log CLV |

All 14 figures are in [`reports/figures/`](reports/figures/).

---

## Business insights

**1 — Value is concentrated, and the model finds it.** The top predicted decile captures **54.2%**
of realised 90-day revenue at **5.4× lift**; the top two deciles capture **66.8%**. A retention
budget aimed at the top decile reaches over half the value at risk while contacting 10% of the base.

**2 — Recency dominates, and it collapses at around 90 days.** `recency_days` and
`average_monthly_spend` are the two largest SHAP contributors. Observed return rate falls from
**74.9%** within 30 days to **56.9%** by 61–90 days, then to **10.5%** past a year. Practically: a
customer silent for more than a quarter belongs in a reactivation flow, not a routine newsletter.

**3 — High history, low forecast is the priority win-back list.** Customers in the top quartile of
historic spend whose predicted 90-day CLV falls in the bottom half form a proven-but-lapsing
segment: **15 customers** in the test set (1.4%) carrying **£173,432** of historic spend, at a
median recency of **356 days**. That is roughly £11,600 of past value per customer with almost
nothing forecast — the place where intervention has the most to save. Unlike a plain recency list,
it is filtered to customers who were actually worth having. The definition is deliberately strict;
loosening either threshold widens the list for a larger campaign.

**4 — Median predicted CLV is a per-customer acquisition-cost ceiling: £8.31.** Paying more than
the median predicted 90-day value to acquire a comparable customer only pays back if the
relationship extends well past the 90-day horizon, which makes it a conservative cap by
construction.

*Reading the median correctly:* the predicted distribution is heavily right-skewed — **57% of
customers spend nothing** in the window — so the median (**£8.31**) and the mean (**£193.08**)
answer different questions.
The **median** describes the *typical* customer and is the right basis for a per-customer cap. The
**mean** is pulled upward by a small number of high-value customers and reflects average monetary
value across the base, which is the relevant figure when acquisition spend is recovered across a
whole cohort rather than per individual. Both are reported in notebook 03; they should not be
substituted for one another.

**Caveat on all four:** the prediction window (Sep–Nov 2011) sits inside the retailer's
pre-Christmas ramp-up, so absolute pound figures are seasonally inflated. The rankings and relative
comparisons are the durable part; levels would need re-estimating on another window before setting
budgets.

---

## Repository structure

```
clv-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                          # gitignored; see download instructions above
├── notebooks/
│   ├── 01_cleaning_eda.ipynb      # cleaning + target-free EDA
│   ├── 02_features_and_target.ipynb
│   └── 03_modeling_and_shap.ipynb
├── src/
│   ├── data_prep.py               # loading and cleaning
│   ├── features.py                # cutoff, target construction, 11 features
│   └── modeling.py                # split, compare, tune, evaluate, save
├── scripts/
│   └── leakage_audit.py           # standalone verification, exits non-zero on failure
├── models/                        # final_model.joblib, best_params.json
└── reports/figures/               # all 14 exported figures
```

Logic lives in `src/`; the notebooks carry the narrative.

### Cached intermediates

Two CSVs are written into `data/` as the notebooks run:

| File | Written by | Purpose |
|---|---|---|
| `transactions_clean.csv` | notebook 01 | cleaned transactions, so 02 and 03 skip the ~90 s Excel parse |
| `customer_features.csv` | notebook 02 | the customer-level modelling table |

Both are **regenerable artifacts**, not source data: they are gitignored along with the rest of
`data/`, and deleting them costs only the time to re-run the notebook that produces them. Nothing
in the pipeline treats them as authoritative — re-running 01 → 02 → 03 rebuilds both from the raw
workbook.

### Verifying no leakage

```bash
python scripts/leakage_audit.py     # after notebook 01 has run
```

13 checks covering window boundaries, split hygiene, and feature/target independence. The central
one is a mutation test: it corrupts **every** post-cutoff transaction (×1000 revenue, shuffled
quantities, overwritten stock codes) and re-derives the feature table. All 11 features come back
bit-identical while the target moves — so no feature can be reading the future. Exits non-zero on
failure, so it can be wired into CI.

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Download the data (above), then run the notebooks **in order** — 01 → 02 → 03. Notebook 01 writes
the cleaned transaction table and 02 writes the customer feature table, so each notebook depends on
the previous one having run.

Every random seed is fixed at **42** (split, models, Optuna sampler), so results reproduce exactly.

**Runtime:** ~2.5 min for notebook 01 (parsing the 45 MB workbook), ~30 s for 02, ~5 min for 03
(Optuna tuning ≈ 3 min, SHAP ≈ 40 s).

## Scope

Modelling and analysis only — no deployment, dashboards, MLOps, or inference pipelines. The final
model is persisted with `joblib` for inspection; there is no serving code.
