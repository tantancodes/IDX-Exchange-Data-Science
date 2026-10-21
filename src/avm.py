"""Shared data loading, cleaning, and evaluation for the CRMLS AVM notebooks.

The cleaning rules, split logic, and metrics are ported from
notebooks/03_baseline_model.ipynb so that every later model is scored on the
same sales, the same monthly cutoffs, and the same price bounds as the
Week 4 linear baseline.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler, TargetEncoder)
from sklearn.tree import DecisionTreeRegressor

TARGET = "ClosePrice"
SKEWED = ["LivingArea", "LotSizeSquareFeet"]       # log-transformed in the log-log linear model
COUNTS = ["BedroomsTotal", "BathroomsTotalInteger"]
NUMERIC = SKEWED + COUNTS
CATEGORICAL = ["City", "PostalCode"]
FEATURES = NUMERIC + CATEGORICAL
LOW_Q, HIGH_Q = 0.005, 0.995
VAL_MONTH = "202602"
TEST_MONTH = "202603"
BACKTEST_MONTHS = ["202511", "202512", "202601"]
RANDOM_STATE = 42

# Prices above $100M on ordinary homes are data-entry errors
# (e.g. a $750K listing recorded as a $664M close).
MAX_PRICE = 100_000_000

# Domain plausibility bounds for a single-family home (fixed rules, not learned
# from data). Values outside these ranges are treated as missing, not dropped.
PLAUSIBLE = {"LivingArea": (300, 30_000),
             "BedroomsTotal": (1, 20),
             "BathroomsTotalInteger": (1, 20)}

# Listing-dependent and post-sale fields are never model inputs.
EXCLUDED = {"ListPrice", "OriginalListPrice", "DaysOnMarket",
            "CumulativeDaysOnMarket", "PurchaseContractDate",
            "ListingContractDate", "CloseDate", "ClosePrice"}
assert not EXCLUDED.intersection(FEATURES)

ModelFactory = Callable[[], RegressorMixin]


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------

def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Read every CRMLSSold file and fingerprint the pull.

    Returns the concatenated raw rows, a per-file snapshot table (name, rows,
    short SHA-256), and a 12-character fingerprint of the whole snapshot.
    """
    files = sorted(Path(data_dir).glob("CRMLSSold*.csv"))
    if not files:
        raise FileNotFoundError(f"No CRMLSSold files in {Path(data_dir).resolve()}")

    frames = []
    for f in files:
        part = pd.read_csv(f, low_memory=False, dtype={"PostalCode": "string"})
        part["SourceFile"] = f.name
        frames.append(part)
    raw = pd.concat(frames, ignore_index=True)

    snapshot = pd.DataFrame([
        {"file": f.name, "rows": int((raw["SourceFile"] == f.name).sum()),
         "sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:12]}
        for f in files
    ])
    fingerprint = hashlib.sha256("".join(snapshot["sha256"]).encode()).hexdigest()[:12]
    return raw, snapshot, fingerprint


def clean_text(series: pd.Series, width: int | None = None) -> pd.Series:
    """Upper-case and strip a text column; blanks become NaN."""
    s = series.astype("string").str.strip().str.upper()
    if width is not None:
        s = s.str[:width]
    s = s.mask(s.eq(""))
    # NumPy NaN rather than pandas NA is compatible with sklearn imputers.
    return pd.Series(s.astype(object).where(s.notna(), np.nan), index=series.index)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Apply the fixed Week 4 validity rules.

    Returns the cleaned sales, a table of rows removed at each step, and the
    number of physical values set to missing per column.
    """
    steps = []

    def log_step(name, before, after):
        steps.append({"step": name, "removed": before - after, "rows_left": after})

    df = raw.copy()
    previous = len(df)

    df = df.loc[(df["PropertyType"] == "Residential") &
                (df["PropertySubType"] == "SingleFamilyResidence")].copy()
    log_step("Single-family residences", previous, len(df))
    previous = len(df)

    # Ignore SourceFile so the same record exported in two monthly files is caught.
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "SourceFile"]).copy()
    log_step("Exact duplicates", previous, len(df))
    previous = len(df)

    df["CloseDate"] = pd.to_datetime(df["CloseDate"], errors="coerce")
    df = df.loc[df["CloseDate"].notna()].copy()
    log_step("Missing CloseDate", previous, len(df))
    previous = len(df)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.loc[df[TARGET] > 0].copy()
    log_step("Missing or non-positive ClosePrice", previous, len(df))
    previous = len(df)

    df = df.loc[df[TARGET] <= MAX_PRICE].copy()
    log_step("ClosePrice above $100M", previous, len(df))
    previous = len(df)

    if "ListingContractDate" in df.columns:
        listed = pd.to_datetime(df["ListingContractDate"], errors="coerce")
        df = df.loc[~(df["CloseDate"] < listed)].copy()
        log_step("CloseDate before listing date", previous, len(df))

    set_missing = {}
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Negative physical quantities are invalid; zero is often unreported.
        invalid = df[col] <= 0
        if col in PLAUSIBLE:
            low, high = PLAUSIBLE[col]
            invalid |= ~df[col].between(low, high) & df[col].notna()
        set_missing[col] = int(invalid.sum())
        df.loc[invalid, col] = np.nan

    df["City"] = clean_text(df["City"])
    df["PostalCode"] = clean_text(df["PostalCode"], width=5)
    df["CloseMonth"] = df["CloseDate"].dt.strftime("%Y%m")
    return df, pd.DataFrame(steps), set_missing


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric columns numeric and normalize categorical missing values."""
    x = frame[FEATURES].copy()
    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors="coerce").astype("float64")
    for col in CATEGORICAL:
        x[col] = clean_text(x[col])
    return x


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def _log_target(pipeline: Pipeline) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=pipeline, func=np.log,
                                      inverse_func=np.exp, check_inverse=False)


def _numeric_linear_pipeline(log_features: bool = False) -> Pipeline:
    steps = []
    if log_features:
        # log1p keeps NaN as NaN, so imputation still sees the missing values.
        steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps += [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]
    return Pipeline(steps)


def make_linear_model(log_target: bool = False) -> RegressorMixin:
    """Week 4 baseline: OLS with scaled numerics and one-hot city/ZIP.

    With ``log_target=True`` this is the log-log variant: log price regressed on
    log living area and log lot size.
    """
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),
    ])
    prep = ColumnTransformer([
        ("size", _numeric_linear_pipeline(log_features=log_target), SKEWED),
        ("counts", _numeric_linear_pipeline(), COUNTS),
        ("categorical", categorical, CATEGORICAL),
    ])
    pipeline = Pipeline([("preprocess", prep), ("regression", LinearRegression())])
    return _log_target(pipeline) if log_target else pipeline


LINEAR_MODELS: dict[str, ModelFactory] = {
    "Linear regression": lambda: make_linear_model(log_target=False),
    "Log-log linear regression": lambda: make_linear_model(log_target=True),
}


def make_tree_model(regressor: RegressorMixin, log_target: bool = False) -> RegressorMixin:
    """Tree-model pipeline: no scaling, median imputation, target-encoded city/ZIP.

    ``TargetEncoder.fit_transform`` cross-fits (5 folds), so during training a
    row's city/ZIP encoding is computed without its own price. Categories not
    seen in training are encoded as the training mean.
    """
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("encode", TargetEncoder(target_type="continuous",
                                 cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))),
    ])
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    pipeline = Pipeline([("preprocess", prep), ("regression", regressor)])
    return _log_target(pipeline) if log_target else pipeline


TREE_ESTIMATORS = {
    "Decision tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
}


def tree_factory(kind: str, params: dict, log_target: bool) -> ModelFactory:
    """Factory for a seeded decision tree or random forest with ``params``."""
    estimator = clone(TREE_ESTIMATORS[kind]).set_params(**params)
    # Clone on every call so a kept fitted model is never refit by a later cutoff.
    return lambda: make_tree_model(clone(estimator), log_target)


def describe_config(params: dict, log_target: bool) -> str:
    """Stable text label for a configuration, used in tables and CSVs."""
    target = "log price" if log_target else "price"
    return ", ".join([f"{k}={v}" for k, v in params.items()] + [f"target={target}"])


def run_grid(data: pd.DataFrame, eval_month: str, kind: str,
             grid: dict[str, list], score_train: bool = False) -> pd.DataFrame:
    """Fit every grid point (raw and log target) on one cutoff.

    Returns one row per configuration with its parameters, fit time, and
    filtered-scope metrics (plus training MdAPE and R² when ``score_train``).
    """
    configs = {}
    for log_target in (False, True):
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values))
            configs[describe_config(params, log_target)] = (params, log_target)

    run = run_cutoff(data, eval_month,
                     {label: tree_factory(kind, params, log_target)
                      for label, (params, log_target) in configs.items()},
                     score_train=score_train)
    rows = []
    for label, (params, log_target) in configs.items():
        res = run["results"][label]
        row = {"model": kind, "params": label,
               "target": "log price" if log_target else "price", **params,
               "fit_seconds": res["fit_seconds"], **res["filtered"]}
        if score_train:
            row["train_MdAPE_pct"] = res["train"]["MdAPE_pct"]
            row["train_R2"] = res["train"]["R2"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def score(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Week 4 metrics plus RMSE. Percentages are on a 0-100 scale."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        raise ValueError("No observations in evaluation subset")
    error = (y_pred - y_true) / y_true
    absolute_pct_error = np.abs(error)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE_USD": mean_absolute_error(y_true, y_pred),
        "RMSE_USD": root_mean_squared_error(y_true, y_pred),
        "MAPE_pct": 100 * absolute_pct_error.mean(),
        "MdAPE_pct": 100 * np.median(absolute_pct_error),
        "PPE10_pct": 100 * (absolute_pct_error <= 0.10).mean(),
        "PPE20_pct": 100 * (absolute_pct_error <= 0.20).mean(),
        "median_bias_pct": 100 * np.median(error),
        "negative_predictions": int((y_pred < 0).sum()),
        "N": len(y_true),
    }


def run_cutoff(data: pd.DataFrame, eval_month: str,
               model_factories: dict[str, ModelFactory],
               n_train_months: int | None = None,
               score_train: bool = False,
               keep_models: bool = False) -> dict:
    """Train on months before ``eval_month`` and score every model on it.

    Price bounds (0.5th/99.5th percentiles) come from the training months only;
    the training set is filtered with them and the same frozen bounds define the
    "filtered" evaluation scope. "all" scores every valid sale in the month.

    Each factory is called once, so a fresh unfitted model is built per cutoff.
    Fitted models are kept only when ``keep_models`` is set, because full-depth
    forests are large.
    """
    months = sorted(data["CloseMonth"].unique())
    prior_months = [month for month in months if month < eval_month]
    if n_train_months is not None:
        prior_months = prior_months[-n_train_months:]
    train = data.loc[data["CloseMonth"].isin(prior_months)]
    evaluate = data.loc[data["CloseMonth"] == eval_month]
    if train.empty or evaluate.empty:
        raise ValueError(f"Insufficient observations for {eval_month}")

    lower, upper = train[TARGET].quantile([LOW_Q, HIGH_Q])
    train_fit = train.loc[train[TARGET].between(lower, upper)]
    in_range = evaluate[TARGET].between(lower, upper)
    x_train = model_inputs(train_fit)
    x_eval = model_inputs(evaluate)

    results = {}
    for name, factory in model_factories.items():
        start = time.perf_counter()
        fitted = factory().fit(x_train, train_fit[TARGET])
        fit_seconds = time.perf_counter() - start
        pred = pd.Series(fitted.predict(x_eval), index=evaluate.index)
        result = {
            "pred": pred,
            "fit_seconds": fit_seconds,
            "filtered": score(evaluate.loc[in_range, TARGET], pred.loc[in_range]),
            "all": score(evaluate[TARGET], pred),
        }
        if score_train:
            result["train"] = score(train_fit[TARGET], fitted.predict(x_train))
        if keep_models:
            result["model"] = fitted
        results[name] = result

    return {"month": eval_month, "n_train": len(train_fit),
            "train_months": (prior_months[0], prior_months[-1]),
            "train_price_range": (train_fit[TARGET].min(), train_fit[TARGET].max()),
            "lower": lower, "upper": upper, "coverage": in_range.mean(),
            "evaluate": evaluate, "in_range": in_range, "results": results}


def split_name(month: str) -> str:
    return ("test" if month == TEST_MONTH
            else "validation" if month == VAL_MONTH else "backtest")


def metric_records(run: dict, params: dict[str, str] | None = None) -> list[dict]:
    """Flatten one ``run_cutoff`` result into rows matching baseline_metrics.csv."""
    params = params or {}
    records = []
    for name, res in run["results"].items():
        for scope in ("filtered", "all"):
            metrics = {k: v for k, v in res[scope].items()}
            records.append({"split": split_name(run["month"]), "month": run["month"],
                            "model": name, "scope": scope, "coverage": run["coverage"],
                            **metrics, "params": params.get(name, "")})
    return records


def paired_bootstrap_mdape_diff(y_true: np.ndarray, pred_a: np.ndarray,
                                pred_b: np.ndarray, n_resamples: int = 1000,
                                seed: int = RANDOM_STATE) -> np.ndarray:
    """Bootstrap distribution of MdAPE(a) - MdAPE(b) in percentage points.

    Both models are scored on the same resampled sales, so the interval reflects
    uncertainty in the difference rather than in each model separately.
    """
    y_true = np.asarray(y_true, dtype=float)
    ape_a = np.abs(np.asarray(pred_a, dtype=float) - y_true) / y_true
    ape_b = np.abs(np.asarray(pred_b, dtype=float) - y_true) / y_true
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_resamples, len(y_true)))
    return 100 * (np.median(ape_a[idx], axis=1) - np.median(ape_b[idx], axis=1))


def error_frame(run: dict, model_names: Iterable[str], scope: str = "filtered") -> pd.DataFrame:
    """Long table of per-sale signed and absolute percentage errors."""
    evaluate = run["evaluate"]
    if scope == "filtered":
        evaluate = evaluate.loc[run["in_range"]]
    frames = []
    for name in model_names:
        pred = run["results"][name]["pred"].loc[evaluate.index]
        signed = 100 * (pred - evaluate[TARGET]) / evaluate[TARGET]
        frames.append(pd.DataFrame({
            "model": name, "City": evaluate["City"], "LivingArea": evaluate["LivingArea"],
            "actual": evaluate[TARGET], "prediction": pred,
            "signed_pct": signed, "APE_pct": signed.abs(),
        }))
    return pd.concat(frames)


METRIC_LABELS = {
    "R2": "R²", "MAE_USD": "MAE ($)", "RMSE_USD": "RMSE ($)",
    "MAPE_pct": "MAPE (%)", "MdAPE_pct": "MdAPE (%)",
    "PPE10_pct": "Within 10% (%)", "PPE20_pct": "Within 20% (%)",
    "median_bias_pct": "Median bias (%)",
    "negative_predictions": "Negative predictions", "N": "N",
}


def metrics_table(run: dict, scope: str = "filtered",
                  columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Models as rows, metrics as columns, for one cutoff and scope."""
    table = pd.DataFrame({name: res[scope] for name, res in run["results"].items()}).T
    return table[list(columns)] if columns is not None else table


def format_table(table: pd.DataFrame, pct_decimals: int = 1) -> pd.DataFrame:
    """Readable copy: dollars with separators, percentages, R² to 3 decimals.

    Formatting is chosen from the column name (``*_USD``, ``*_pct``, ``R2``,
    counts), and metric columns are renamed with ``METRIC_LABELS``.
    """
    out = table.copy()
    for col in out.columns:
        name = str(col[-1] if isinstance(col, tuple) else col)
        values = pd.to_numeric(out[col], errors="coerce")
        if values.isna().all():
            continue
        if name.endswith("USD"):
            out[col] = values.map("{:,.0f}".format)
        elif name.endswith("pct"):
            out[col] = values.map(f"{{:.{pct_decimals}f}}".format)
        elif "R2" in name:
            out[col] = values.map("{:.3f}".format)
        elif name in {"N", "n", "negative_predictions"}:
            out[col] = values.astype(int).map("{:,}".format)
    return out.rename(columns=METRIC_LABELS)


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------

LINEAR, LOG_LINEAR = "Linear regression", "Log-log linear regression"
TREE, FOREST = "Decision tree", "Random forest"
MODEL_ORDER = [LINEAR, LOG_LINEAR, TREE, FOREST]
MODEL_COLORS = {LINEAR: "#9a9a9a", LOG_LINEAR: "#3b6ea8", TREE: "#e08a2e", FOREST: "#1f7a45"}
DISPLAY_NAMES = {LINEAR: "Linear regression (baseline)", LOG_LINEAR: "Log-log linear regression",
                 TREE: "Decision tree", FOREST: "Random forest"}


def assert_metrics_match(run: dict, saved: pd.DataFrame) -> None:
    """Check a rescored cutoff against saved metric rows at displayed precision.

    MdAPE and other percentages are compared to 1 decimal, R² to 3, MAE to the
    dollar, and N exactly, which is the precision used in the result tables.
    """
    precision = {"MdAPE_pct": 1, "MAPE_pct": 1, "PPE10_pct": 1, "PPE20_pct": 1,
                 "R2": 3, "MAE_USD": 0}
    for row in saved.itertuples():
        scored = run["results"][row.model][row.scope]
        assert scored["N"] == row.N, (row.model, row.scope, "N")
        for metric, digits in precision.items():
            assert round(scored[metric], digits) == round(getattr(row, metric), digits), \
                (row.model, row.scope, metric, scored[metric], getattr(row, metric))


def highlight_table(table: pd.DataFrame, formats: dict[str, str],
                    best: dict[str, str]) -> str:
    """HTML table with formatted values and the best value per column in bold.

    ``best`` maps a column to "min" or "max". Plain HTML is used because
    pandas' Styler needs jinja2, which is not a project dependency.
    """
    winners = {col: (table[col].idxmin() if rule == "min" else table[col].idxmax())
               for col, rule in best.items()}
    head = "".join(f"<th style='text-align:right;padding:6px 12px'>{col}</th>" for col in table.columns)
    rows = []
    for index in table.index:
        cells = []
        for col in table.columns:
            # Read cell by cell: iterrows would upcast integer columns to float.
            text = formats.get(col, "{}").format(table.at[index, col])
            style = "text-align:right;padding:6px 12px"
            if winners.get(col) == index:
                style += ";font-weight:bold;background:#dcefe3"
            cells.append(f"<td style='{style}'>{text}</td>")
        rows.append(f"<tr><th style='text-align:left;padding:6px 12px'>{index}</th>{''.join(cells)}</tr>")
    return ("<table style='border-collapse:collapse;font-size:14px'>"
            f"<thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def finish_chart(fig, ax, title: str, subtitle: str, source: str) -> None:
    """Takeaway title, explanatory subtitle, and a source note under the axes."""
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=10, color="#555555")
    fig.text(0.02, 0.01, source, fontsize=8.5, color="#666666", ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
