"""Train the deployed AVM and write everything the Streamlit app loads.

Run from the repository root:

    python -m src.train

The model is the configuration frozen in results/final_model_config.json (lock
commit 8d41984), refit on every month through April 2026 now that the April
test is finished. The accuracy figures the app shows come from the April test
of the model trained through March, not from this refit, which has no held-out
month left to be scored on.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avm  # noqa: E402  (the app imports avm the same way, so the saved pipeline unpickles)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
DISTRICT_SHAPEFILE = ROOT / "data" / "external" / "school_districts_2024_25" / "DistrictAreas2425.shp"
DISTRICT_CACHE = ROOT / "data" / "processed" / "school_district_by_location.csv"
RESULTS_DIR = ROOT / "results"
APP_DIR = ROOT / "models" / "app"

EXPECTED_FINGERPRINT = "66cab54017ab"
LOCK_COMMIT = "8d41984"
LAST_MONTH = avm.TEAM_TEST_MONTH
RANGE_COLUMNS = ["LivingArea", "LotSizeSquareFeet", "BedroomsTotal", "BathroomsTotalInteger", "YearBuilt"]
# About 50 m; small enough that district labels barely change, large enough to shrink the file.
SIMPLIFY_DEGREES = 0.0005


def load_sales() -> tuple[pd.DataFrame, str]:
    raw, _, fingerprint = avm.load_raw(DATA_DIR)
    assert fingerprint == EXPECTED_FINGERPRINT, f"Data pull changed: {fingerprint}"
    df, _, _ = avm.clean(raw, drop_rereported=True)
    df, _ = avm.attach_school_districts(df, DISTRICT_CACHE, DISTRICT_SHAPEFILE)
    df["County"] = avm.clean_text(df["CountyOrParish"])
    return df, fingerprint


def final_model() -> tuple[dict, avm.FeatureSet, avm.ModelFactory]:
    config = json.loads((RESULTS_DIR / "final_model_config.json").read_text())
    features = avm.FeatureSet(config["features"]["name"], tuple(config["features"]["numeric"]),
                              tuple(config["features"]["categorical"]))
    factory = avm.tree_factory(config["model"], config["params"], config["log_target"], features)
    return config, features, factory


def check_against_april_test(df, config, features, factory) -> float:
    """Refit through March with this code and compare with the scored April predictions."""
    model, _ = avm.fit_through(df, avm.TEAM_VAL_MONTH, factory, features)
    april = df.loc[df["CloseMonth"] == LAST_MONTH]
    scored = pd.read_csv(RESULTS_DIR / "april_predictions.csv")
    assert len(scored) == len(april) and np.allclose(scored["ClosePrice"], april[avm.TARGET])
    pred = model.predict(avm.model_inputs(april, features))
    return float(np.max(np.abs(pred - scored[f"pred_{config['model']}"]) / scored[f"pred_{config['model']}"]))


def zip_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ZIP → median coordinates, county and district; and ZIP × city sale counts."""
    zips = df.loc[df["PostalCode"].astype("string").str.fullmatch(r"\d{5}").fillna(False)]

    def most_common(series):
        series = series.dropna()
        return series.mode().iat[0] if len(series) else None

    grouped = zips.groupby("PostalCode")
    lookup = pd.DataFrame({
        "sales": grouped.size(),
        "latitude": grouped["Latitude"].median(),
        "longitude": grouped["Longitude"].median(),
        "county": grouped["County"].agg(most_common),
        "school_district": grouped["SchoolDistrict"].agg(most_common),
        "school_district_name": grouped["SchoolDistrictName"].agg(most_common),
    }).reset_index()
    cities = (zips.dropna(subset=["City"]).groupby(["PostalCode", "City"]).size().rename("sales")
          .reset_index().sort_values(["PostalCode", "sales"], ascending=[True, False]))
    return lookup, cities


def error_bands(model_name: str, n_bands: int = 10) -> pd.DataFrame:
    """Empirical April error by predicted price, from the through-March model's test predictions.

    Bands are deciles of the predicted price, because the app only knows the
    prediction. Ratios are sale price ÷ prediction.
    """
    homes = pd.read_csv(RESULTS_DIR / "april_predictions.csv")
    homes = homes.loc[homes["in_price_bounds"]]
    pred, actual = homes[f"pred_{model_name}"], homes["ClosePrice"]
    ratio, ape = actual / pred, 100 * (pred - actual).abs() / actual
    edges = pred.quantile(np.linspace(0, 1, n_bands + 1)).to_numpy().copy()
    edges[0], edges[-1] = 0, np.inf
    band = pd.cut(pred, edges, labels=False, include_lowest=True)
    grouped = pd.DataFrame({"band": band, "ratio": ratio, "ape": ape}).groupby("band")
    table = pd.DataFrame({
        "pred_low": edges[:-1], "pred_high": edges[1:], "n": grouped.size(),
        "MdAPE_pct": grouped["ape"].median(),
        "PPE10_pct": grouped["ape"].apply(lambda s: 100 * (s <= 10).mean()),
        "PPE20_pct": grouped["ape"].apply(lambda s: 100 * (s <= 20).mean()),
        **{f"ratio_q{q}": grouped["ratio"].quantile(q / 100) for q in (10, 25, 75, 90)},
    })
    return table.reset_index(drop=True)


def simplified_districts(df: pd.DataFrame) -> tuple[object, float]:
    """Simplified boundary layer for the app, and its label agreement with the full layer."""
    districts = avm.load_school_districts(DISTRICT_SHAPEFILE)
    simple = districts.copy()
    simple["geometry"] = simple.geometry.simplify(SIMPLIFY_DEGREES, preserve_topology=True).make_valid()
    sample = df.loc[df["Latitude"].notna()].sample(20_000, random_state=avm.RANDOM_STATE)
    labels = avm.assign_school_districts(sample["Latitude"], sample["Longitude"], simple)["SchoolDistrict"]
    agreement = float((labels.fillna("NA") == sample["SchoolDistrict"].fillna("NA")).mean())
    return simple, agreement


def main() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    df, fingerprint = load_sales()
    config, features, factory = final_model()
    print(f"{len(df):,} cleaned sales, snapshot {fingerprint}; model {config['model']}, "
          f"features {features.name}")

    max_diff = check_against_april_test(df, config, features, factory)
    assert max_diff < 1e-6, f"Through-March refit does not reproduce the April test ({max_diff:.2e})"
    print(f"Through-March refit reproduces the scored April predictions (max relative diff {max_diff:.1e})")

    start = time.perf_counter()
    model, info = avm.fit_through(df, LAST_MONTH, factory, features)
    fit_seconds = time.perf_counter() - start
    train_fit = info.pop("train_fit")
    assert list(model.feature_names_in_) == features.inputs
    model_path = APP_DIR / "avm_model.joblib"
    joblib.dump(model, model_path, compress=3)
    size = avm.saved_size(model)
    print(f"Fitted on {info['n_train']:,} sales {info['train_months'][0]}–{info['train_months'][1]} "
          f"in {fit_seconds:.1f} s; saved {size['size_mb']:.1f} MB, loads in {size['load_seconds']:.2f} s")

    lookup, cities = zip_tables(df)
    lookup.to_csv(APP_DIR / "zip_lookup.csv", index=False)
    cities.to_csv(APP_DIR / "zip_cities.csv", index=False)
    bands = error_bands(config["model"])
    bands.to_csv(APP_DIR / "error_bands.csv", index=False)
    simple, agreement = simplified_districts(df)
    boundary_path = APP_DIR / "school_districts.geojson"
    boundary_path.unlink(missing_ok=True)
    simple.to_file(boundary_path, driver="GeoJSON", COORDINATE_PRECISION=5)
    print(f"{len(lookup):,} ZIPs, {len(cities):,} ZIP-city pairs, {len(bands)} error bands; simplified "
          f"boundaries {boundary_path.stat().st_size / 1e6:.1f} MB, {agreement:.2%} label agreement")

    summary = pd.read_csv(RESULTS_DIR / "metrics_summary.csv", dtype={"month": str})
    test = summary.query("split == 'test' and model == @config['model']").set_index("scope")
    context = json.loads((RESULTS_DIR / "april_test_context.json").read_text())
    homes = pd.read_csv(RESULTS_DIR / "april_predictions.csv")
    top_cut = homes["ClosePrice"].quantile(0.99)
    top = homes.loc[homes["ClosePrice"] >= top_cut]
    top_signed = 100 * (top[f"pred_{config['model']}"] - top["ClosePrice"]) / top["ClosePrice"]
    ranges = {col: {"p1": float(train_fit[col].quantile(0.01)), "p99": float(train_fit[col].quantile(0.99)),
                    "min": float(train_fit[col].min()), "max": float(train_fit[col].max())}
              for col in RANGE_COLUMNS}
    metadata = {
        "model": config["model"], "params": config["params"], "log_target": config["log_target"],
        "features": config["features"], "inputs": features.inputs,
        "trained_through": LAST_MONTH, **info,
        "train_price_range": [float(train_fit[avm.TARGET].min()), float(train_fit[avm.TARGET].max())],
        "valuation_year": int(train_fit["SaleYear"].max()),
        "input_ranges": ranges,
        "evaluation": {
            "source": "April 2026 test of the same configuration trained through March 2026",
            "test_month": context["test_month"], "train_months": context["train_months"],
            "price_bounds": context["price_bounds"], "coverage": context["coverage"],
            **{scope: {k: float(test.loc[scope, k]) for k in
                       ("MdAPE_pct", "PPE10_pct", "PPE20_pct", "R2", "MAE_USD", "N", "median_bias_pct")}
               for scope in ("filtered", "all")},
            "top_1pct": {"price_cut": float(top_cut), "n": int(len(top)),
                         "median_signed_error_pct": float(top_signed.median()),
                         "n_above_train_max": int((top["ClosePrice"] > context["train_price_max"]).sum())},
        },
        "lock_commit": LOCK_COMMIT, "data_snapshot": fingerprint,
        "check_through_march_max_relative_diff": max_diff,
        "district_boundary_simplify_degrees": SIMPLIFY_DEGREES,
        "district_label_agreement": agreement,
        "model_file_mb": size["size_mb"], "load_seconds": size["load_seconds"],
        "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (APP_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (APP_DIR / "feature_list.json").write_text(json.dumps(features.inputs, indent=2) + "\n")
    print(f"Wrote {model_path.relative_to(ROOT)} and the app lookups to {APP_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
