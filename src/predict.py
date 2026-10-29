"""Estimate sale prices with the deployed model (build it first with python -m src.train).

One home:

    python -m src.predict --zip 92253 --city "La Quinta" --living-area 1800 \\
        --lot-size 6500 --beds 3 --baths 2 --year-built 1985

Many homes, from a CSV with columns PostalCode, City, LivingArea,
LotSizeSquareFeet, BedroomsTotal, BathroomsTotalInteger, YearBuilt and,
optionally, Latitude and Longitude:

    python -m src.predict --csv homes.csv --out estimates.csv

Missing coordinates are filled with the ZIP's median sale location, and the
school district is looked up from the coordinates, as in the app.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avm  # noqa: E402
from config import APP_DIR  # noqa: E402

HOME_COLUMNS = ["PostalCode", "City", "LivingArea", "LotSizeSquareFeet", "BedroomsTotal",
                "BathroomsTotalInteger", "YearBuilt"]


@dataclass
class Artifacts:
    model: object
    metadata: dict
    features: avm.FeatureSet
    zips: pd.DataFrame
    districts: object


def load_artifacts(app_dir: Path = APP_DIR) -> Artifacts:
    """Load the saved model and lookups, and check they belong together."""
    import geopandas

    metadata = json.loads((app_dir / "metadata.json").read_text())
    feature_list = json.loads((app_dir / "feature_list.json").read_text())
    model = joblib.load(app_dir / "avm_model.joblib")
    # A mismatch means the model and its saved feature list come from different runs.
    assert feature_list == list(model.feature_names_in_) == metadata["inputs"], feature_list
    features = avm.FeatureSet(metadata["features"]["name"], tuple(metadata["features"]["numeric"]),
                              tuple(metadata["features"]["categorical"]))
    assert features.inputs == feature_list
    zips = pd.read_csv(app_dir / "zip_lookup.csv", dtype={"PostalCode": str, "school_district": str})
    districts = geopandas.read_file(app_dir / "school_districts.geojson")
    return Artifacts(model, metadata, features, zips.set_index("PostalCode"), districts)


def prepare(homes: pd.DataFrame, art: Artifacts) -> pd.DataFrame:
    """Home descriptions → the raw columns the pipeline expects, plus notes columns.

    Applies the training plausibility rules (bad values become missing), fills
    missing coordinates from the ZIP, and looks up the school district.
    """
    frame = homes.copy()
    for column in ("Latitude", "Longitude"):
        if column not in frame:
            frame[column] = np.nan
    frame["PostalCode"] = frame["PostalCode"].astype("string").str.strip().str[:5]
    frame["City"] = avm.clean_text(frame["City"])
    for column in HOME_COLUMNS[2:] + ["Latitude", "Longitude"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    for column, (low, high) in avm.PLAUSIBLE.items():
        frame.loc[~frame[column].between(low, high), column] = np.nan
    frame.loc[frame["LotSizeSquareFeet"] <= 0, "LotSizeSquareFeet"] = np.nan
    frame["SaleYear"] = float(art.metadata["valuation_year"])
    frame.loc[~frame["YearBuilt"].between(avm.EARLIEST_YEAR_BUILT, frame["SaleYear"]), "YearBuilt"] = np.nan

    given = frame["Latitude"].notna() & frame["Longitude"].notna()
    known_zip = frame["PostalCode"].isin(art.zips.index)
    fill = ~given & known_zip
    frame.loc[fill, "Latitude"] = frame.loc[fill, "PostalCode"].map(art.zips["latitude"]).to_numpy()
    frame.loc[fill, "Longitude"] = frame.loc[fill, "PostalCode"].map(art.zips["longitude"]).to_numpy()
    inside = (frame["Latitude"].between(*avm.CA_LATITUDE) & frame["Longitude"].between(*avm.CA_LONGITUDE))
    frame.loc[~inside, ["Latitude", "Longitude"]] = np.nan
    frame["coordinates_from"] = np.where(~inside, "missing", np.where(given, "given", "ZIP median"))

    labels = avm.assign_school_districts(frame["Latitude"], frame["Longitude"], art.districts)
    frame["SchoolDistrict"] = labels["SchoolDistrict"]
    frame["SchoolDistrictName"] = labels["SchoolDistrictName"]
    frame["zip_in_training"] = known_zip
    return frame


def predict(homes: pd.DataFrame, art: Artifacts) -> pd.DataFrame:
    frame = prepare(homes, art)
    x = avm.model_inputs(frame, art.features)
    assert list(x.columns) == list(art.model.feature_names_in_)
    frame["estimate"] = art.model.predict(x)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, help="CSV of homes (see module docstring)")
    parser.add_argument("--out", type=Path, help="where to write estimates for --csv")
    for flag, help_text in [("--zip", "ZIP code"), ("--city", "city"), ("--living-area", "sq ft"),
                            ("--lot-size", "sq ft"), ("--beds", "bedrooms"), ("--baths", "bathrooms"),
                            ("--year-built", "year"), ("--lat", "latitude"), ("--lon", "longitude")]:
        parser.add_argument(flag, help=help_text)
    args = parser.parse_args()

    if args.csv:
        homes = pd.read_csv(args.csv, dtype={"PostalCode": str})
    else:
        missing = [f for f in ("zip", "city", "living_area", "beds", "baths", "year_built")
                   if getattr(args, f) is None]
        if missing:
            parser.error("give --csv, or describe one home; missing: " + ", ".join(missing))
        homes = pd.DataFrame([{"PostalCode": args.zip, "City": args.city, "LivingArea": args.living_area,
                               "LotSizeSquareFeet": args.lot_size, "BedroomsTotal": args.beds,
                               "BathroomsTotalInteger": args.baths, "YearBuilt": args.year_built,
                               "Latitude": args.lat, "Longitude": args.lon}])

    result = predict(homes, load_artifacts())
    columns = HOME_COLUMNS + ["Latitude", "Longitude", "coordinates_from", "SchoolDistrictName",
                              "zip_in_training", "estimate"]
    if args.out:
        result[columns].to_csv(args.out, index=False)
        print(f"Wrote {len(result):,} estimates to {args.out}")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(result[columns].assign(estimate=result["estimate"].map("${:,.0f}".format)).to_string(index=False))


if __name__ == "__main__":
    main()
