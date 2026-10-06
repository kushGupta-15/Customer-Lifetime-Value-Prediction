"""Model comparison, Optuna tuning, evaluation and persistence.

Every model is trained on log1p(future_clv) but scored on the original monetary
scale, because a mean absolute error expressed in log units is not something a
retention budget can be set from.

The evaluation protocol keeps each split to one job:

    train (64%)       fit the candidates; 5-fold CV inside it drives Optuna
    validation (16%)  choose between the four candidates - nothing else
    test (20%)        touched once, at the very end, on the final tuned model

Once the architecture and hyperparameters are settled, the final model is refit
on train + validation together: those decisions are already made, so holding
data back from the final fit would only waste it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_prep import PROJECT_ROOT
from src.features import FEATURE_COLUMNS, TARGET_COLUMN

RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # share of the train+validation portion, i.e. 16% of all customers
CV_FOLDS = 5
N_TRIALS = 50

MODELS_DIR = PROJECT_ROOT / "models"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"


def split_train_test(dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """Hold out the test set, before anything else looks at the data.

    A random split is correct here because temporal separation is already
    enforced by the target construction: every row's features precede the cutoff
    and every row's target follows it, so splitting by time as well would only
    shrink the data without removing any leakage.

    Returns the train+validation portion and the test portion. The test portion
    is not touched again until the single final evaluation.
    """
    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def split_train_validation(X, y, validation_size=VALIDATION_SIZE, random_state=RANDOM_SEED):
    """Carve a validation set out of the train+validation portion.

    Model selection needs a held-out set that is *not* the test set. Choosing
    between candidates on the test set would spend it: the winner would be the
    model that happened to suit those particular customers, and the final metric
    would no longer be an honest estimate of performance on unseen data.
    """
    return train_test_split(X, y, test_size=validation_size, random_state=random_state)


def build_candidate_models(random_state=RANDOM_SEED):
    """The four candidates, at sensible defaults so the comparison is fair.

    Linear Regression is wrapped in a Pipeline with StandardScaler so the scaler
    is refit inside every CV fold and never sees validation data. The tree models
    split on thresholds and are scale-invariant, so scaling them would add a
    fitted step with no benefit.
    """
    candidates = {
        "LinearRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "RandomForest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
    }
    if XGBRegressor is not None:
        candidates["XGBoost"] = XGBRegressor(random_state=random_state, n_jobs=-1)
    if LGBMRegressor is not None:
        candidates["LightGBM"] = LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1)
    return candidates


def predict_monetary(model, X):
    """Predict on the original monetary scale.

    Inverts the training-time log1p with expm1, then clips at zero: a customer
    cannot spend a negative amount, so a negative prediction is a modelling
    artefact that would otherwise be scored as if it were a real forecast.
    """
    return np.clip(np.expm1(model.predict(X)), 0.0, None)


def fit_on_log_target(model, X_train, y_train):
    """Fit a model on log1p of the monetary target."""
    model.fit(X_train, np.log1p(y_train))
    return model


def regression_metrics(y_true, y_pred):
    """MAE, RMSE and R2 on the original monetary scale.

    MAE is the primary metric: it is in pounds, is not dominated by the handful
    of very large customers the way RMSE is, and answers the question a campaign
    owner actually asks — how far off is a typical per-customer forecast.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def compare_models(X_train, y_train, X_valid, y_valid, random_state=RANDOM_SEED):
    """Train each candidate on the train split and score it on the validation split.

    Selection happens here, on validation data, so that the test set stays
    unspent for the final estimate.

    Returns (results_df, fitted_models). No cross-validation at this stage: this
    is a coarse screen to pick one architecture, and CV is reserved for tuning it.
    """
    results, fitted = [], {}
    for name, model in build_candidate_models(random_state).items():
        fit_on_log_target(model, X_train, y_train)
        metrics = regression_metrics(y_valid, predict_monetary(model, X_valid))
        results.append({"model": name, **metrics})
        fitted[name] = model
    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    return results_df, fitted


def cross_val_mae(model_factory, X, y, n_splits=CV_FOLDS, random_state=RANDOM_SEED):
    """Mean K-fold CV MAE on the original monetary scale.

    The log1p transform is applied inside each fold and inverted before scoring,
    so the reported error is always in pounds.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_maes = []
    for train_idx, valid_idx in kfold.split(X):
        model = fit_on_log_target(model_factory(), X.iloc[train_idx], y.iloc[train_idx])
        preds = predict_monetary(model, X.iloc[valid_idx])
        fold_maes.append(mean_absolute_error(y.iloc[valid_idx], preds))
    return float(np.mean(fold_maes))


def suggest_params(trial, model_name):
    """Optuna search space, expressed in each library's own parameter names."""
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,  # subsample is inert in LightGBM unless bagging is enabled
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        }
    if model_name == "RandomForest":
        # A forest has no learning rate or column-subsample-per-tree, so the
        # spec's boosting space maps onto its nearest structural equivalents.
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    raise ValueError(f"No search space defined for {model_name!r}")


def build_tuned_model(model_name, params, random_state=RANDOM_SEED):
    """Instantiate a model of the winning type with the given hyperparameters."""
    if model_name == "XGBoost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed.")
        return XGBRegressor(random_state=random_state, n_jobs=-1, **params)
    if model_name == "LightGBM":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed.")
        return LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1, **params)
    if model_name == "RandomForest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    raise ValueError(f"Cannot tune {model_name!r}")


def tune_model(model_name, X_train, y_train, n_trials=N_TRIALS, random_state=RANDOM_SEED):
    """Optuna search minimising 5-fold CV MAE on the training set.

    Tuning is scored by cross-validation on the training split alone; the test
    set stays untouched so the final number is an honest held-out estimate.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = suggest_params(trial, model_name)
        return cross_val_mae(
            lambda: build_tuned_model(model_name, params, random_state),
            X_train,
            y_train,
            random_state=random_state,
        )

    # A fixed sampler seed makes the whole search reproducible, not just the model fit.
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        study_name=f"clv_{model_name.lower()}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def decile_lift(y_true, y_pred, n_deciles=10):
    """Actual mean spend per predicted-value decile.

    This is the metric that matches how the model is used: customers are ranked
    and the top slice is targeted, so what matters is whether the ranking holds,
    not whether each individual forecast is precise.
    """
    frame = pd.DataFrame({"actual": np.asarray(y_true), "predicted": np.asarray(y_pred)})
    # rank-then-cut rather than qcut on values, because heavy ties at zero
    # predictions would otherwise collapse several deciles into one bin.
    ranks = frame["predicted"].rank(method="first", ascending=False)
    frame["decile"] = pd.qcut(ranks, n_deciles, labels=range(1, n_deciles + 1)).astype(int)

    summary = frame.groupby("decile").agg(
        customers=("actual", "size"),
        mean_predicted=("predicted", "mean"),
        mean_actual=("actual", "mean"),
        total_actual=("actual", "sum"),
    )
    summary["pct_of_total_revenue"] = summary["total_actual"] / summary["total_actual"].sum() * 100
    summary["lift_vs_average"] = summary["mean_actual"] / frame["actual"].mean()
    return summary


def save_best_params(params, model_name, cv_mae, path=BEST_PARAMS_PATH):
    """Persist the winning hyperparameters, tagged with which model they belong to."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "cv_mae": cv_mae,
        "random_seed": RANDOM_SEED,
        "params": params,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_model(model, path=FINAL_MODEL_PATH):
    """Persist the fitted model with joblib."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
