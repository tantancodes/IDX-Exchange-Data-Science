"""Streamlit app for the IDX Exchange California single-family AVM.

Build the model and lookups first, then launch from the repository root:

    python -m src.train
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import avm  # noqa: E402  (the saved pipeline refers to avm.add_engineered_features)
import predict as predictor  # noqa: E402  (same input preparation as python -m src.predict)

APP_DIR = ROOT / "models" / "app"
RESULTS_DIR = ROOT / "results"
REQUIRED = ["avm_model.joblib", "metadata.json", "feature_list.json", "zip_lookup.csv",
            "zip_cities.csv", "error_bands.csv", "school_districts.geojson"]
FEW_SALES = 20

st.set_page_config(page_title="California home value estimate", layout="wide")


def dollars(value: float) -> str:
    return f"${value / 1e6:,.2f}M" if value >= 1e6 else f"${value:,.0f}"


def md_dollars(value: float) -> str:
    """Dollar amount for markdown, where a pair of $ signs would start a math block."""
    return dollars(value).replace("$", "\\$")


@st.cache_resource
def load_artifacts():
    # Includes the check that the saved feature list matches the model's expected inputs.
    return predictor.load_artifacts(APP_DIR)


@st.cache_data
def load_tables():
    cities = pd.read_csv(APP_DIR / "zip_cities.csv", dtype={"PostalCode": str})
    bands = pd.read_csv(APP_DIR / "error_bands.csv")
    summary = pd.read_csv(RESULTS_DIR / "metrics_summary.csv", dtype={"month": str})
    return cities, bands, summary


missing = [name for name in REQUIRED if not (APP_DIR / name).exists()]
if missing:
    st.error(f"Missing app files in models/app/: {', '.join(missing)}. "
             "Run `python -m src.train` from the repository root first.")
    st.stop()

ART = load_artifacts()
META, ZIPS = ART.metadata, ART.zips
CITIES, BANDS, SUMMARY = load_tables()
EVAL = META["evaluation"]
VALUATION_YEAR = META["valuation_year"]


def range_warnings(values: dict[str, float]) -> list[str]:
    labels = {"LivingArea": "living area", "LotSizeSquareFeet": "lot size",
              "BedroomsTotal": "bedrooms", "BathroomsTotalInteger": "bathrooms", "YearBuilt": "year built"}
    notes = []
    for column, value in values.items():
        if value is None or np.isnan(value):
            continue
        r = META["input_ranges"][column]
        fmt = "{:.0f}" if column == "YearBuilt" else "{:,.0f}"
        shown = {k: fmt.format(v) for k, v in {**r, "value": value}.items()}
        if value < r["min"] or value > r["max"]:
            notes.append(f"{labels[column].capitalize()} ({shown['value']}) is outside every home in the training "
                         f"data ({shown['min']}–{shown['max']}). Treat the estimate with caution.")
        elif value < r["p1"] or value > r["p99"]:
            notes.append(f"{labels[column].capitalize()} ({shown['value']}) is rarer than 98% of training homes "
                         f"(typical range {shown['p1']}–{shown['p99']}).")
    return notes


def estimate_page():
    st.title("Estimate a home's value")
    st.caption("California single-family homes. Enter what you know about the property; "
               f"the estimate is for a sale in {VALUATION_YEAR}.")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        zip_options = ZIPS.sort_values("sales", ascending=False).index.tolist()
        default_zip = zip_options[0]
        zip_code = st.selectbox(
            "ZIP code", sorted(zip_options), index=sorted(zip_options).index(default_zip),
            format_func=lambda z: f"{z} ({ZIPS.at[z, 'county'].title() if pd.notna(ZIPS.at[z, 'county']) else '?'} "
                                  f"County, {ZIPS.at[z, 'sales']:,} sales)")
        city_options = CITIES.loc[CITIES["PostalCode"] == zip_code, "City"].tolist()
        city = st.selectbox("City", city_options, format_func=str.title,
                            help="Cities that appear with this ZIP in the sales data, most common first.")
        c1, c2 = st.columns(2)
        living_area = c1.number_input("Living area (sq ft)", min_value=300, max_value=30_000, value=1_800, step=50)
        lot_size = c2.number_input("Lot size (sq ft)", min_value=0, max_value=5_000_000, value=6_500, step=100,
                                   help="Enter 0 if unknown.")
        c3, c4, c5 = st.columns(3)
        beds = c3.number_input("Bedrooms", min_value=1, max_value=20, value=3)
        baths = c4.number_input("Bathrooms", min_value=1, max_value=20, value=2)
        year_built = c5.number_input("Year built", min_value=1850, max_value=VALUATION_YEAR, value=1985)

        exact = st.checkbox("I know the exact coordinates",
                            help="Otherwise the ZIP's median sale location is used for latitude, "
                                 "longitude and school district.")
        latitude, longitude = float(ZIPS.at[zip_code, "latitude"]), float(ZIPS.at[zip_code, "longitude"])
        if exact:
            c6, c7 = st.columns(2)
            latitude = c6.number_input("Latitude", value=round(latitude, 5), format="%.5f")
            longitude = c7.number_input("Longitude", value=round(longitude, 5), format="%.5f")

    home = pd.DataFrame([{
        "PostalCode": zip_code, "City": city, "LivingArea": living_area, "LotSizeSquareFeet": lot_size,
        "BedroomsTotal": beds, "BathroomsTotalInteger": baths, "YearBuilt": year_built,
        "Latitude": latitude if exact else np.nan, "Longitude": longitude if exact else np.nan,
    }])
    result = predictor.predict(home, ART).iloc[0]
    estimate = float(result["estimate"])
    notes = []
    if result["coordinates_from"] == "missing":
        notes.append("The coordinates are outside California, so they are treated as missing, "
                     "as in training.")

    band = BANDS.loc[(BANDS["pred_low"] <= estimate) & (estimate < BANDS["pred_high"])].iloc[0]
    notes += range_warnings({"LivingArea": living_area, "LotSizeSquareFeet": lot_size or np.nan,
                             "BedroomsTotal": beds, "BathroomsTotalInteger": baths, "YearBuilt": year_built})
    if ZIPS.at[zip_code, "sales"] < FEW_SALES:
        notes.append(f"This ZIP has only {ZIPS.at[zip_code, 'sales']} sales in the data, so the estimate "
                     "leans on city, district and location more than on the ZIP itself.")
    ceiling = META["price_bounds"][1]
    top = EVAL["top_1pct"]
    if estimate >= 0.75 * ceiling:
        notes.append(f"This estimate is close to the highest price the model was trained on "
                     f"({md_dollars(ceiling)}); it cannot predict above that. In the April test, homes that sold "
                     f"for {md_dollars(top['price_cut'])} or more were under-predicted by a median of "
                     f"{-top['median_signed_error_pct']:.0f}%.")

    with right:
        st.metric("Estimated value", dollars(round(estimate, -3)))
        st.markdown(
            f"**Typical error for homes estimated in this range ({md_dollars(band['pred_low'])}"
            f"{' and up' if np.isinf(band['pred_high']) else ' – ' + md_dollars(band['pred_high'])}):** "
            f"{band['MdAPE_pct']:.1f}%.  \n"
            f"In the April 2026 test, half of these homes sold between **{md_dollars(estimate * band['ratio_q25'])}** "
            f"and **{md_dollars(estimate * band['ratio_q75'])}**, and 80% between "
            f"{md_dollars(estimate * band['ratio_q10'])} and {md_dollars(estimate * band['ratio_q90'])}.")
        st.caption("These ranges are empirical: they describe how far sale prices fell from the model's "
                   "estimates for similarly priced homes in a past month. They are not a guarantee for "
                   "this home.")
        for note in notes:
            st.warning(note)

        st.subheader("What the model used")
        used = pd.DataFrame({"Value": [
            f"{zip_code}, {city.title()}",
            ("missing" if result["coordinates_from"] == "missing" else
             f"{result['Latitude']:.4f}, {result['Longitude']:.4f}"
             + (" (ZIP median)" if result["coordinates_from"] == "ZIP median" else "")),
            result["SchoolDistrictName"] if isinstance(result["SchoolDistrictName"], str) else "not found",
            f"{VALUATION_YEAR - year_built} years (built {year_built}, valued in {VALUATION_YEAR})",
            f"{living_area:,} sq ft, lot {lot_size:,} sq ft" if lot_size else f"{living_area:,} sq ft, lot unknown",
            f"{beds} bed / {baths} bath",
        ]}, index=["Location", "Coordinates", "School district", "Age at sale", "Size", "Rooms"])
        st.table(used)


def comparison_page():
    st.title("Model comparison")
    st.caption("Every model was frozen before April 2026 was scored, then trained on all sales through "
               "March 2026 and scored once on April. Source: `results/metrics_summary.csv`.")
    scope = st.radio("Sales scored", ["filtered", "all"], horizontal=True,
                     format_func=lambda s: {"filtered": "Inside the training price range (98.9%)",
                                            "all": "All April sales"}[s])
    names = {**avm.DISPLAY_NAMES, avm.FOREST: "Random forest (Week 6 features)",
             avm.XGBOOST: "XGBoost (deployed)"}
    april = (SUMMARY.query("split == 'test' and scope == @scope")
             .assign(Model=lambda t: t["model"].map(lambda m: names.get(m, m)),
                     MAE_USD=lambda t: t["MAE_USD"].round())
             .sort_values("MdAPE_pct"))
    st.subheader("April 2026 test")
    st.dataframe(
        april[["Model", "features", "MdAPE_pct", "PPE10_pct", "PPE20_pct", "R2", "MAE_USD", "N"]],
        hide_index=True, width="stretch",
        column_config={
            "features": "Features",
            "MdAPE_pct": st.column_config.NumberColumn("Typical error (MdAPE)", format="%.2f%%"),
            "PPE10_pct": st.column_config.NumberColumn("Within 10%", format="%.1f%%"),
            "PPE20_pct": st.column_config.NumberColumn("Within 20%", format="%.1f%%"),
            "R2": st.column_config.NumberColumn("R²", format="%.3f"),
            "MAE_USD": st.column_config.NumberColumn("Mean error ($)", format="localized"),
            "N": st.column_config.NumberColumn("Homes", format="localized"),
        })
    st.caption("Features: A is the Week 4–5 set (living area, lot size, bedrooms, bathrooms, city, ZIP). "
               "“B without ratios” adds age at sale, latitude, longitude and school district.")

    left, right = st.columns(2, gap="large")
    bars = alt.Chart(april).mark_bar().encode(
        x=alt.X("MdAPE_pct:Q", title="Typical error, MdAPE (%)"),
        y=alt.Y("Model:N", sort="x", title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.condition(alt.datum.model == META["model"], alt.value("#c0392b"), alt.value("#9aa7b4")),
        tooltip=["Model", alt.Tooltip("MdAPE_pct:Q", format=".2f", title="MdAPE (%)")])
    left.markdown("**Typical error on April 2026**")
    left.altair_chart(bars, width="stretch")

    months = SUMMARY.query("scope == @scope").assign(
        Model=lambda t: t["model"].map(lambda m: names.get(m, m)),
        Month=lambda t: pd.to_datetime(t["month"], format="%Y%m"))
    lines = alt.Chart(months).mark_line(point=True).encode(
        x=alt.X("yearmonth(Month):T", title=None),
        y=alt.Y("MdAPE_pct:Q", title="Typical error, MdAPE (%)", scale=alt.Scale(zero=True)),
        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", columns=2)),
        tooltip=["Model", alt.Tooltip("yearmonth(Month):T", title="Month"),
                 alt.Tooltip("MdAPE_pct:Q", format=".2f", title="MdAPE (%)")])
    right.markdown("**Stability across months** (Dec–Feb backtests, March validation, April test)")
    right.altair_chart(lines, width="stretch")


def how_it_works_page():
    f, a = EVAL["filtered"], EVAL["all"]
    baseline = SUMMARY.query("split == 'test' and scope == 'filtered' and model == @avm.LINEAR")["MdAPE_pct"].iat[0]
    st.title("How it works")
    st.markdown(f"""
### What it estimates
The likely sale price of a California single-family home from its size, rooms, age and location.
It works for homes that are not for sale: it never uses a list price or anything else that only
exists once a home is on the market.

### Data
Closed single-family sales from CRMLS, the California Regional Multiple Listing Service, from
January 2024 through April 2026: 309,039 sales after cleaning. Cleaning drops impossible records
(no sale price, prices above \\$100M, close dates before the listing date), exact duplicates, and
sales reported twice under the same listing within 30 days at nearly the same price. Implausible
sizes, room counts, build years and out-of-state coordinates are treated as unknown rather than
dropped. The data itself is licensed and is not part of this project's repository.

### What the model uses
Living area, lot size, bedrooms, bathrooms, age at sale, latitude and longitude, city, ZIP code,
and school district. The district comes from matching the home's coordinates against the
California School District Areas 2024-25 boundaries. City, ZIP and district are converted to the
average sale price of their area in the training data, computed so that no sale ever sees its own
price. Without exact coordinates, the app uses the ZIP code's median sale location.

### What it never uses
List price, original list price, days on market, price reductions, contract and closing dates, and
tax amounts. These are either only known once a home is listed or sold, or they can encode the sale
price itself (tax assessments can be reset by a sale).

### How it was tested
Models were always trained on earlier months and tested on a later one, never on a random mix of
dates. December 2025 to February 2026 were used as backtests, March 2026 to choose between models,
and April 2026 as the final test. The model choice was locked in a commit (`{META['lock_commit']}`)
before April was scored, and April was scored once.

### Where the accuracy numbers come from
**The model running in this app was refit on all sales through April 2026 after the test was
finished, with the same frozen settings. The accuracy numbers in this app come from the April 2026
test of the same configuration trained through March 2026. The refit itself has not been scored,
because there is no later month left to test it on.** Training on the extra month is standard once
evaluation is done. We expect its accuracy to be close to the tested model's, but that is an
expectation, not a measurement.

On the April test, among the {f['N']:,.0f} homes inside the training price range ({EVAL['coverage']:.1%} of April
sales), the typical error was **{f['MdAPE_pct']:.1f}%** of the sale price. {f['PPE10_pct']:.0f}% of homes were priced within
10% and {f['PPE20_pct']:.0f}% within 20%. Including every April sale, the typical error was {a['MdAPE_pct']:.1f}%.
The Week 4 baseline, a straight linear regression on size, rooms, city and ZIP, had a typical error of
{baseline:.1f}%.

### Limitations
- **The most expensive homes.** The model cannot predict above {md_dollars(META['price_bounds'][1])}, the top of its
  training range. In the April test, homes that sold for {md_dollars(EVAL['top_1pct']['price_cut'])} or more were
  under-predicted by a median of {-EVAL['top_1pct']['median_signed_error_pct']:.0f}%.
- **Older homes and some markets.** Error roughly doubles from new homes to homes over 75 years
  old, and is highest in the Bay Area, especially Oakland.
- **No time trend.** The model does not know prices move over time. In April it was a median
  {-f['median_bias_pct']:.1f}% low, consistent with a rising spring market.
- **Condition is unknown.** Renovations, views, pools and condition are not inputs, so two homes
  with the same size and location get the same estimate.
- **Coordinates.** With the ZIP median location, the school district and fine location may not
  match the actual home.

### Rebuild
From the repository root, with the CRMLS files in `data/raw/`:
`python -m src.train` then `streamlit run app.py`. Notebooks 05–07 hold the full analysis.
""")


pages = st.navigation([
    st.Page(estimate_page, title="Estimate", icon=":material/home:", default=True),
    st.Page(comparison_page, title="Model comparison", icon=":material/bar_chart:", url_path="model-comparison"),
    st.Page(how_it_works_page, title="How it works", icon=":material/info:", url_path="how-it-works"),
])
pages.run()
