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
from dataclasses import dataclass
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
# Week 4–5 split, kept so notebooks 03–04 reproduce.
VAL_MONTH = "202602"
TEST_MONTH = "202603"
BACKTEST_MONTHS = ["202511", "202512", "202601"]
# Team split from Week 6 on: March (the Week 5 test month) becomes validation and
# April 2026 is the shared test month, scored once at the end.
TEAM_VAL_MONTH = "202603"
TEAM_TEST_MONTH = "202604"
TEAM_BACKTEST_MONTHS = ["202512", "202601", "202602"]
RANDOM_STATE = 42

# Prices above $100M on ordinary homes are data-entry errors
# (e.g. a $750K listing recorded as a $664M close).
MAX_PRICE = 100_000_000

# Domain plausibility bounds for a single-family home (fixed rules, not learned
# from data). Values outside these ranges are treated as missing, not dropped.
PLAUSIBLE = {"LivingArea": (300, 30_000),
             "BedroomsTotal": (1, 20),
             "BathroomsTotalInteger": (1, 20)}
EARLIEST_YEAR_BUILT = 1850
# A later record of the same listing that closes within this window at nearly
# the same price is the same sale reported again (Week 6 rule, off by default).
REREPORT_DAYS = 30
REREPORT_PRICE_SHARE = 0.02
# Generous box around California; coordinates outside it are typos or
# out-of-state listings and are set to missing.
CA_LATITUDE = (32.5, 42.1)
CA_LONGITUDE = (-124.5, -114.1)

# Listing-dependent, post-sale, and tax fields are never model inputs. Tax
# amounts can be reassessed after a sale, so they may encode the price.
EXCLUDED = {"ListPrice", "OriginalListPrice", "DaysOnMarket",
            "CumulativeDaysOnMarket", "PurchaseContractDate",
            "ListingContractDate", "ContractStatusChangeDate", "CloseDate",
            "ClosePrice", "TaxAnnualAmount", "TaxYear", "TaxAssessedValue"}
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
    return raw, snapshot, snapshot_fingerprint(snapshot)


def snapshot_fingerprint(snapshot: pd.DataFrame) -> str:
    """12-character fingerprint of the files listed in a snapshot table.

    Passing a subset of rows fingerprints just those files, which lets a
    notebook confirm the months it uses are unchanged after a new month lands.
    """
    return hashlib.sha256("".join(snapshot["sha256"]).encode()).hexdigest()[:12]


def clean_text(series: pd.Series, width: int | None = None) -> pd.Series:
    """Upper-case and strip a text column; blanks become NaN."""
    s = series.astype("string").str.strip().str.upper()
    if width is not None:
        s = s.str[:width]
    s = s.mask(s.eq(""))
    # NumPy NaN rather than pandas NA is compatible with sklearn imputers.
    return pd.Series(s.astype(object).where(s.notna(), np.nan), index=series.index)


def find_rereported_sales(df: pd.DataFrame) -> pd.Index:
    """Index of records that repeat an earlier sale of the same listing.

    A record is a repeat when an earlier kept record has the same ListingKey,
    closed at most REREPORT_DAYS before it, and has a price within
    REREPORT_PRICE_SHARE of it. Each record is compared with the last kept one,
    so a chain of three copies keeps only the first.
    """
    repeated = df.loc[df["ListingKey"].duplicated(keep=False)]
    repeated = repeated.sort_values(["ListingKey", "CloseDate"], kind="stable")
    drop = []
    for _, group in repeated.groupby("ListingKey", sort=False):
        kept_date = kept_price = None
        for index, date, price in zip(group.index, group["CloseDate"], group[TARGET]):
            if (kept_date is not None and (date - kept_date).days <= REREPORT_DAYS
                    and abs(price - kept_price) <= REREPORT_PRICE_SHARE * kept_price):
                drop.append(index)
            else:
                kept_date, kept_price = date, price
    return pd.Index(drop)


def clean(raw: pd.DataFrame, drop_rereported: bool = False
          ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Apply the fixed Week 4 validity rules.

    Returns the cleaned sales, a table of rows removed at each step, and the
    number of physical values set to missing per column.

    ``drop_rereported`` (Week 6 on) also drops later copies of a re-reported
    sale; see ``find_rereported_sales``. It is off by default so notebooks 03
    and 04 reproduce their frozen results. The dropped rows are kept in
    ``steps.attrs["rereported"]`` for logging.
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
    previous = len(df)

    rereported = df.iloc[:0]
    if drop_rereported:
        rereported = df.loc[find_rereported_sales(df)]
        df = df.drop(rereported.index)
        log_step(f"Re-reported sale (same listing, within {REREPORT_DAYS} days "
                 f"and {REREPORT_PRICE_SHARE:.0%} of an earlier close)", previous, len(df))

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

    # Week 6 fields. These only add columns, so the Week 4–5 inputs are unchanged.
    df["SaleYear"] = df["CloseDate"].dt.year.astype("float64")
    df["YearBuilt"] = pd.to_numeric(df["YearBuilt"], errors="coerce")
    invalid = df["YearBuilt"].notna() & ~df["YearBuilt"].between(EARLIEST_YEAR_BUILT, df["SaleYear"])
    set_missing["YearBuilt"] = int(invalid.sum())
    df.loc[invalid, "YearBuilt"] = np.nan

    for col in ("Latitude", "Longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    inside = df["Latitude"].between(*CA_LATITUDE) & df["Longitude"].between(*CA_LONGITUDE)
    invalid = df[["Latitude", "Longitude"]].notna().any(axis=1) & ~inside
    set_missing["Coordinates"] = int(invalid.sum())
    df.loc[~inside, ["Latitude", "Longitude"]] = np.nan

    df["City"] = clean_text(df["City"])
    df["PostalCode"] = clean_text(df["PostalCode"], width=5)
    df["CloseMonth"] = df["CloseDate"].dt.strftime("%Y%m")
    steps = pd.DataFrame(steps)
    steps.attrs["rereported"] = rereported[["ListingKey", "CloseDate", TARGET]].assign(
        CloseMonth=rereported["CloseDate"].dt.strftime("%Y%m"))
    return df, steps, set_missing


# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------

# Engineered columns and the raw columns each one is computed from. They are
# built row by row inside the model pipeline, so nothing is learned from data.
ENGINEERED = {
    "BedBathRatio": ("BedroomsTotal", "BathroomsTotalInteger"),
    "LivingAreaPerBedroom": ("LivingArea", "BedroomsTotal"),
    "LotToLivingAreaRatio": ("LotSizeSquareFeet", "LivingArea"),
    "AgeAtSale": ("YearBuilt", "SaleYear"),
}
# Size-like columns that the log-log linear model logs; other numerics are scaled as is.
LOGGED_IN_LINEAR = {"LivingArea", "LotSizeSquareFeet", "LivingAreaPerBedroom",
                    "LotToLivingAreaRatio"}
TEXT_INPUTS = {"City", "PostalCode", "SchoolDistrict"}


@dataclass(frozen=True)
class FeatureSet:
    """Named model features: numeric (raw or engineered) and categorical."""
    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    def __post_init__(self):
        assert not EXCLUDED.intersection(self.numeric + self.categorical), self.name

    @property
    def inputs(self) -> list[str]:
        """Raw columns the pipeline needs, in a stable order."""
        cols = []
        for col in self.numeric + self.categorical:
            for raw in ENGINEERED.get(col, (col,)):
                if raw not in cols:
                    cols.append(raw)
        return cols

    def plus(self, name: str, *groups: str) -> FeatureSet:
        numeric, categorical = list(self.numeric), list(self.categorical)
        for group in groups:
            add_numeric, add_categorical = FEATURE_GROUPS[group]
            numeric += [c for c in add_numeric if c not in numeric]
            categorical += [c for c in add_categorical if c not in categorical]
        return FeatureSet(name, tuple(numeric), tuple(categorical))


FEATURE_GROUPS = {
    "ratios": (("BedBathRatio", "LivingAreaPerBedroom", "LotToLivingAreaRatio"), ()),
    "age": (("AgeAtSale",), ()),
    "coordinates": (("Latitude", "Longitude"), ()),
    "school district": ((), ("SchoolDistrict",)),
}
BASE_FEATURES = FeatureSet("A", tuple(NUMERIC), tuple(CATEGORICAL))
EXTENDED_FEATURES = BASE_FEATURES.plus("B", *FEATURE_GROUPS)
# Dataset B without the three ratios, which added nothing in the Week 6 ablation.
FEATURES_WITHOUT_RATIOS = BASE_FEATURES.plus("B without ratios", "age", "coordinates",
                                             "school district")


def add_engineered_features(x: pd.DataFrame) -> pd.DataFrame:
    """Append every engineered column whose raw inputs are present.

    Division by a missing denominator gives NaN, which the pipeline imputes.
    Zero counts and areas were already set to missing during cleaning.
    """
    x = x.copy()
    if {"BedroomsTotal", "BathroomsTotalInteger"} <= set(x.columns):
        x["BedBathRatio"] = x["BedroomsTotal"] / x["BathroomsTotalInteger"]
    if {"LivingArea", "BedroomsTotal"} <= set(x.columns):
        x["LivingAreaPerBedroom"] = x["LivingArea"] / x["BedroomsTotal"]
    if {"LotSizeSquareFeet", "LivingArea"} <= set(x.columns):
        x["LotToLivingAreaRatio"] = x["LotSizeSquareFeet"] / x["LivingArea"]
    if {"YearBuilt", "SaleYear"} <= set(x.columns):
        x["AgeAtSale"] = x["SaleYear"] - x["YearBuilt"]
    return x


def model_inputs(frame: pd.DataFrame, features: FeatureSet = BASE_FEATURES) -> pd.DataFrame:
    """Raw input columns for ``features``: numerics as float, text normalized."""
    x = frame[features.inputs].copy()
    for col in x.columns:
        if col in TEXT_INPUTS:
            x[col] = clean_text(x[col])
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").astype("float64")
    return x


# --------------------------------------------------------------------------
# School districts
# --------------------------------------------------------------------------

def load_school_districts(shapefile: Path):
    """CA School District Areas 2024-25 as a GeoDataFrame in WGS84.

    geopandas is imported here so notebooks that never join districts do not
    need it. Seven polygons in the source file are invalid and are repaired.
    """
    import geopandas as gpd

    districts = gpd.read_file(shapefile)[["CDCode", "DistrictNa", "DistrictTy",
                                          "CountyName", "geometry"]]
    districts["geometry"] = districts.geometry.make_valid()
    return districts.to_crs(epsg=4326)


def assign_school_districts(latitude: pd.Series, longitude: pd.Series,
                            districts) -> pd.DataFrame:
    """One school-district label per point.

    Elementary and high school districts overlap outside unified districts, so
    a point inside a unified district gets that district's CDS code, and any
    other point gets its elementary and high school codes joined as
    "elementary|high" (either part may be "-"). Missing coordinates, and points
    outside every district, get NaN.
    """
    import geopandas as gpd

    located = latitude.notna() & longitude.notna()
    points = gpd.GeoDataFrame(
        index=latitude.index[located],
        geometry=gpd.points_from_xy(longitude[located], latitude[located]), crs=4326)
    hits = gpd.sjoin(points, districts, how="inner", predicate="within")
    # Ties (a point on a shared edge or in an overlapping sliver) go to the lowest code.
    hits = hits.reset_index().sort_values(["index", "CDCode"])
    first = hits.drop_duplicates(["index", "DistrictTy"]).set_index(["index", "DistrictTy"])

    found = set(first.index.get_level_values("DistrictTy"))

    def by_type(kind, column):
        # A single point usually matches one or two district types, not all three.
        if kind not in found:
            return pd.Series(np.nan, index=latitude.index, dtype=object)
        return first.xs(kind, level="DistrictTy")[column].reindex(latitude.index)

    unified_code, unified_name = by_type("Unified", "CDCode"), by_type("Unified", "DistrictNa")
    elem_code, elem_name = by_type("Elementary", "CDCode"), by_type("Elementary", "DistrictNa")
    high_code, high_name = by_type("High", "CDCode"), by_type("High", "DistrictNa")

    split = unified_code.isna() & (elem_code.notna() | high_code.notna())
    code = unified_code.copy()
    code[split] = elem_code[split].fillna("-") + "|" + high_code[split].fillna("-")
    name = unified_name.copy()
    name[split] = elem_name[split].fillna("-") + " / " + high_name[split].fillna("-")
    kind = pd.Series(np.where(unified_code.notna(), "Unified",
                              np.where(split, "Elementary + high", None)),
                     index=latitude.index)
    return pd.DataFrame({"SchoolDistrict": code, "SchoolDistrictName": name,
                         "SchoolDistrictType": kind})


DISTRICT_LOCATION = ["ListingKey", "Latitude", "Longitude"]


def attach_school_districts(df: pd.DataFrame, cache_path: Path,
                            shapefile: Path | None = None) -> tuple[pd.DataFrame, int]:
    """Add school-district columns to cleaned sales, using a per-location cache.

    The cache holds one row per listing location (ListingKey plus coordinates,
    since some keys appear at more than one location). Locations missing from
    it are joined against ``shapefile`` and appended, so adding a month only
    joins its new listings. Returns the merged sales and the number of
    locations newly joined.
    """
    cache_path = Path(cache_path)
    locations = df[DISTRICT_LOCATION].drop_duplicates()
    if cache_path.exists():
        cache = pd.read_csv(cache_path, dtype={"SchoolDistrict": str})
        new = locations.merge(cache[DISTRICT_LOCATION], how="left", indicator=True)
        new = new.loc[new.pop("_merge") == "left_only"]
    else:
        cache, new = None, locations
    new = new.reset_index(drop=True)
    if len(new):
        if shapefile is None:
            raise FileNotFoundError(f"{len(new)} locations are not in {cache_path}; pass the shapefile")
        labels = assign_school_districts(new["Latitude"], new["Longitude"],
                                         load_school_districts(shapefile))
        joined = pd.concat([new, labels], axis=1)
        cache = joined if cache is None else pd.concat([cache, joined], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(cache_path, index=False)
        cache = pd.read_csv(cache_path, dtype={"SchoolDistrict": str})
    merged = df.merge(cache, on=DISTRICT_LOCATION, how="left", validate="many_to_one",
                      indicator=True)
    assert (merged.pop("_merge") == "both").all() and len(merged) == len(df)
    return merged, len(new)


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


def _with_features(prep: ColumnTransformer, regressor: RegressorMixin,
                   log_target: bool) -> RegressorMixin:
    """Engineered columns, then column-wise preprocessing, then the model.

    The first step only adds columns, and the ColumnTransformer picks the ones a
    feature set uses, so the Week 5 feature set gives the same model as before.
    """
    pipeline = Pipeline([("features", FunctionTransformer(add_engineered_features)),
                         ("preprocess", prep), ("regression", regressor)])
    return _log_target(pipeline) if log_target else pipeline


def make_linear_model(log_target: bool = False,
                      features: FeatureSet = BASE_FEATURES) -> RegressorMixin:
    """Week 4 baseline: OLS with scaled numerics and one-hot categoricals.

    With ``log_target=True`` this is the log-log variant: log price regressed on
    the logs of the size-like columns (living area and lot size in Week 4).
    """
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),
    ])
    size = [c for c in features.numeric if c in LOGGED_IN_LINEAR]
    other = [c for c in features.numeric if c not in LOGGED_IN_LINEAR]
    prep = ColumnTransformer([
        ("size", _numeric_linear_pipeline(log_features=log_target), size),
        ("counts", _numeric_linear_pipeline(), other),
        ("categorical", categorical, list(features.categorical)),
    ])
    return _with_features(prep, LinearRegression(), log_target)


LINEAR_MODELS: dict[str, ModelFactory] = {
    "Linear regression": lambda: make_linear_model(log_target=False),
    "Log-log linear regression": lambda: make_linear_model(log_target=True),
}


def make_tree_model(regressor: RegressorMixin, log_target: bool = False,
                    features: FeatureSet = BASE_FEATURES) -> RegressorMixin:
    """Tree-model pipeline: no scaling, median imputation, target-encoded categoricals.

    ``TargetEncoder.fit_transform`` cross-fits (5 folds), so during training a
    row's city/ZIP/district encoding is computed without its own price.
    Categories not seen in training are encoded as the training mean.
    """
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("encode", TargetEncoder(target_type="continuous",
                                 cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))),
    ])
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), list(features.numeric)),
        ("categorical", categorical, list(features.categorical)),
    ])
    return _with_features(prep, regressor, log_target)


TREE_ESTIMATORS = {
    "Decision tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
}
LIGHTGBM, XGBOOST = "LightGBM", "XGBoost"


def new_estimator(kind: str) -> RegressorMixin:
    """Unfitted, seeded estimator for a model name.

    LightGBM and XGBoost are imported only when asked for, so notebooks that
    never use them (03–05) do not need the libraries or their OpenMP runtime.
    """
    if kind in TREE_ESTIMATORS:
        return clone(TREE_ESTIMATORS[kind])
    if kind == LIGHTGBM:
        from lightgbm import LGBMRegressor
        # deterministic + row-wise histograms make repeated fits give identical trees.
        return LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
                             deterministic=True, force_row_wise=True)
    if kind == XGBOOST:
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
    raise KeyError(kind)


def tree_factory(kind: str, params: dict, log_target: bool,
                 features: FeatureSet = BASE_FEATURES) -> ModelFactory:
    """Factory for a seeded tree ensemble (or single tree) with ``params``."""
    estimator = new_estimator(kind).set_params(**params)
    # Clone on every call so a kept fitted model is never refit by a later cutoff.
    return lambda: make_tree_model(clone(estimator), log_target, features)


def describe_config(params: dict, log_target: bool) -> str:
    """Stable text label for a configuration, used in tables and CSVs."""
    target = "log price" if log_target else "price"
    return ", ".join([f"{k}={v}" for k, v in params.items()] + [f"target={target}"])


def run_grid(data: pd.DataFrame, eval_month: str, kind: str,
             grid: dict[str, list], score_train: bool = False,
             log_targets: tuple[bool, ...] = (False, True),
             features: FeatureSet = BASE_FEATURES) -> pd.DataFrame:
    """Fit every grid point (for each target in ``log_targets``) on one cutoff.

    Returns one row per configuration with its parameters, fit time, and
    filtered-scope metrics (plus training MdAPE and R² when ``score_train``).
    """
    configs = {}
    for log_target in log_targets:
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values))
            configs[describe_config(params, log_target)] = (params, log_target)

    run = run_cutoff(data, eval_month,
                     {label: tree_factory(kind, params, log_target, features)
                      for label, (params, log_target) in configs.items()},
                     score_train=score_train, inputs=features)
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
               keep_models: bool = False,
               inputs: FeatureSet = BASE_FEATURES) -> dict:
    """Train on months before ``eval_month`` and score every model on it.

    Price bounds (0.5th/99.5th percentiles) come from the training months only;
    the training set is filtered with them and the same frozen bounds define the
    "filtered" evaluation scope. "all" scores every valid sale in the month.

    Each factory is called once, so a fresh unfitted model is built per cutoff.
    Fitted models are kept only when ``keep_models`` is set, because full-depth
    forests are large. ``inputs`` sets the raw columns passed to every model;
    each pipeline selects the ones its own feature set uses, so models with
    different feature sets can share one cutoff and the same rows.
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
    x_train = model_inputs(train_fit, inputs)
    x_eval = model_inputs(evaluate, inputs)

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


def fit_through(data: pd.DataFrame, last_month: str, factory: ModelFactory,
                inputs: FeatureSet = BASE_FEATURES) -> tuple[RegressorMixin, dict]:
    """Fit one model on every month up to and including ``last_month``.

    Uses the same rule as ``run_cutoff``: price bounds are the training
    months' 0.5th/99.5th percentiles and the model is fitted only on sales
    inside them. Used for the deployed model once evaluation is finished.
    """
    train = data.loc[data["CloseMonth"] <= last_month]
    lower, upper = train[TARGET].quantile([LOW_Q, HIGH_Q])
    train_fit = train.loc[train[TARGET].between(lower, upper)]
    model = factory().fit(model_inputs(train_fit, inputs), train_fit[TARGET])
    months = sorted(train_fit["CloseMonth"].unique())
    return model, {"train_months": [months[0], months[-1]], "n_train": int(len(train_fit)),
                   "price_bounds": [float(lower), float(upper)], "train_fit": train_fit}


def split_name(month: str, val_month: str = VAL_MONTH, test_month: str = TEST_MONTH) -> str:
    return ("test" if month == test_month
            else "validation" if month == val_month else "backtest")


def metric_records(run: dict, params: dict[str, str] | None = None,
                   val_month: str = VAL_MONTH, test_month: str = TEST_MONTH) -> list[dict]:
    """Flatten one ``run_cutoff`` result into rows matching baseline_metrics.csv."""
    params = params or {}
    split = split_name(run["month"], val_month, test_month)
    records = []
    for name, res in run["results"].items():
        for scope in ("filtered", "all"):
            metrics = {k: v for k, v in res[scope].items()}
            records.append({"split": split, "month": run["month"],
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
MODEL_COLORS = {LINEAR: "#9a9a9a", LOG_LINEAR: "#3b6ea8", TREE: "#e08a2e", FOREST: "#1f7a45",
                LIGHTGBM: "#7b3fa0", XGBOOST: "#c0392b"}
DISPLAY_NAMES = {LINEAR: "Linear regression (baseline)", LOG_LINEAR: "Log-log linear regression",
                 TREE: "Decision tree", FOREST: "Random forest",
                 LIGHTGBM: "LightGBM", XGBOOST: "XGBoost"}


def saved_size(model, compress: int = 3) -> dict[str, float]:
    """Size and load time of ``model`` saved with joblib, as the app will save it."""
    import io

    import joblib

    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=compress)
    size_mb = buffer.tell() / 1e6
    buffer.seek(0)
    start = time.perf_counter()
    joblib.load(buffer)
    return {"size_mb": size_mb, "load_seconds": time.perf_counter() - start}


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
