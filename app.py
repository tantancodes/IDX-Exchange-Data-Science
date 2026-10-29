"""Streamlit app for the IDX Exchange California single-family AVM.

Build the model and lookups first, then launch from the repository root:

    python -m src.train
    streamlit run app.py
"""

from __future__ import annotations

import html
import math
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import avm  # noqa: E402  (the saved pipeline refers to avm.add_engineered_features)
import explain  # noqa: E402
import predict as predictor  # noqa: E402  (same input preparation as python -m src.predict)

APP_DIR = ROOT / "models" / "app"
RESULTS_DIR = ROOT / "results"
STYLESHEET = ROOT / "assets" / "style.css"
REQUIRED = ["avm_model.joblib", "metadata.json", "feature_list.json", "zip_lookup.csv",
            "zip_cities.csv", "error_bands.csv", "school_districts.geojson"]
FEW_SALES = 20
TITLE = "California Home Value Estimator"
INK, SOFT, FAINT, LINE = "#0a0a0a", "#6b6b6b", "#d4d4d4", "#e5e5e5"

st.set_page_config(page_title=TITLE, page_icon=":material/home:", layout="wide",
                   initial_sidebar_state="collapsed")


def dollars(value: float) -> str:
    return f"${value / 1e6:,.2f}M" if value >= 1e6 else f"${value:,.0f}"


def short_dollars(value: float) -> str:
    return f"${value / 1e6:.2f}M" if abs(value) >= 1e6 else f"${value / 1e3:,.0f}k"


def signed_pct(value: float, decimals: int = 0) -> str:
    # A real minus sign lines up with the plus sign in proportional fonts.
    return f"{value:+.{decimals}f}%".replace("-", "−")


@st.cache_resource
def load_artifacts():
    # Includes the check that the saved feature list matches the model's expected inputs.
    return predictor.load_artifacts(APP_DIR)


@st.cache_resource
def load_tables():
    cities = pd.read_csv(APP_DIR / "zip_cities.csv", dtype={"PostalCode": str})
    bands = pd.read_csv(APP_DIR / "error_bands.csv")
    summary = pd.read_csv(RESULTS_DIR / "metrics_summary.csv", dtype={"month": str})
    return cities, bands, summary


@st.cache_resource
def stylesheet() -> str:
    return STYLESHEET.read_text()


@alt.theme.register("avm", enable=True)
def avm_theme() -> alt.theme.ThemeConfig:
    font = "Geist, system-ui, sans-serif"
    axis = {"labelColor": SOFT, "titleColor": SOFT, "labelFont": font, "titleFont": font,
            "labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 400,
            "gridColor": "#efefef", "domainColor": LINE, "tickColor": LINE}
    return {"config": {
        "background": "transparent", "font": font, "view": {"stroke": None},
        "axis": axis, "axisY": {"domain": False, "ticks": False},
        "title": {"anchor": "start", "font": font, "fontSize": 15, "fontWeight": 500, "color": INK,
                  "subtitleColor": SOFT, "subtitleFont": font, "subtitleFontSize": 13, "offset": 14},
        "legend": {"labelColor": SOFT, "titleColor": SOFT, "labelFont": font, "titleFont": font},
        "text": {"font": font, "fontSize": 12, "color": INK},
        "bar": {"color": INK}, "line": {"color": INK, "strokeWidth": 1.5},
    }}


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
# Range bar scale, shared by every estimate so wider bands look wider.
RATIO_DOMAIN = (math.floor(BANDS["ratio_q10"].min() * 20) / 20, math.ceil(BANDS["ratio_q90"].max() * 20) / 20)


@st.cache_resource
def district_shapes(codes: tuple[str, ...]) -> dict:
    shapes = ART.districts.loc[ART.districts["CDCode"].isin(codes), ["DistrictNa", "geometry"]]
    return shapes.__geo_interface__


def page_head(eyebrow: str, title: str, lede: str) -> None:
    st.html(f'<header class="page-head fade-in"><p class="eyebrow">{eyebrow}</p>'
            f'<h1>{title}</h1><p class="lede">{lede}</p></header>')


def section_head(eyebrow: str, title: str, text: str = "") -> None:
    body = f"<p>{text}</p>" if text else ""
    st.html(f'<div class="section-head"><p class="eyebrow">{eyebrow}</p><h2>{title}</h2>{body}</div>')


def range_warnings(values: dict[str, float]) -> list[tuple[str, str]]:
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
            notes.append(("caution", f"{labels[column].capitalize()} ({shown['value']}) is outside every home in "
                                     f"the training data ({shown['min']}–{shown['max']}). Treat the estimate "
                                     "with caution."))
        elif value < r["p1"] or value > r["p99"]:
            notes.append(("note", f"{labels[column].capitalize()} ({shown['value']}) is rarer than 98% of "
                                  f"training homes (typical range {shown['p1']}–{shown['p99']})."))
    return notes


def notices_html(notes: list[tuple[str, str]]) -> str:
    if not notes:
        return ""
    items = "".join(f'<div class="notice {kind}" role="note"><span class="tag">'
                    f'{"Caution" if kind == "caution" else "Note"}</span><p>{html.escape(text)}</p></div>'
                    for kind, text in notes)
    return f'<div class="notices">{items}</div>'


def result_card(estimate: float, band: pd.Series, place: str, notes: list[tuple[str, str]]) -> str:
    shown = round(estimate, -3)
    lo, hi = RATIO_DOMAIN

    def at(ratio: float) -> float:
        return 100 * (ratio - lo) / (hi - lo)

    def seg(kind: str, a: float, b: float) -> str:
        left, width = f"{at(a):.2f}%", f"{at(b) - at(a):.2f}%"
        return (f'<div class="range-seg {kind}" data-left="{left}" data-width="{width}" '
                f'style="left:{left};width:{width}"></div>')

    ticks = "".join(f'<span style="left:{at(1 + p / 100):.2f}%">{"0" if p == 0 else signed_pct(p)}</span>'
                    for p in range(-30, 31, 10) if 4 <= at(1 + p / 100) <= 96)
    q = {k: band[f"ratio_q{k}"] for k in (10, 25, 75, 90)}
    at_q = {k: dollars(round(estimate * v, -3)) for k, v in q.items()}
    band_range = (f"{short_dollars(band['pred_low'])} and up" if np.isinf(band["pred_high"])
                  else f"{short_dollars(band['pred_low'])}–{short_dollars(band['pred_high'])}")
    if band["pred_low"] == 0:
        band_range = f"under {short_dollars(band['pred_high'])}"
    pct = {k: signed_pct(100 * (v - 1)) for k, v in q.items()}
    element_id = f"est-{int(shown)}-{int(band['pred_low'])}"
    return f"""
<section class="card result" aria-labelledby="{element_id}-label">
  <p class="eyebrow" id="{element_id}-label">Estimated sale price, {VALUATION_YEAR}</p>
  <p class="place">{html.escape(place)}</p>
  <span class="estimate" id="{element_id}" data-value="{shown:.0f}" aria-hidden="true">{dollars(shown)}</span>
  <span class="sr-only" style="position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)"
        aria-live="polite">Estimated value {dollars(shown)}</span>
  <div class="range" role="img" aria-label="Middle half of April test sales: {at_q[25]} to
       {at_q[75]}. 80 percent: {at_q[10]} to {at_q[90]}.">
    <div class="range-track">{seg("outer", q[10], q[90])}{seg("inner", q[25], q[75])}
      <div class="range-mark" style="left:{at(1):.2f}%"></div></div>
    <div class="range-axis">{ticks}</div>
  </div>
  <ul class="range-legend">
    <li><span class="swatch inner"></span><span class="label">Middle half</span>
        <span class="value">{at_q[25]} – {at_q[75]}</span></li>
    <li><span class="swatch outer"></span><span class="label">80% of sales</span>
        <span class="value">{at_q[10]} – {at_q[90]}</span></li>
  </ul>
  <p class="plain">In the April 2026 test, half of the homes we estimated at {band_range} sold within
    {pct[25]} to {pct[75]} of our estimate, and 80% within {pct[10]} to {pct[90]}.
    The typical miss in this price range was {band['MdAPE_pct']:.1f}%.</p>
  <p class="small-note">These ranges are empirical: they describe how far sale prices fell from the model's
    estimates for similarly priced homes in a past month. They are not a guarantee for this home.</p>
  {notices_html(notes)}
</section>
<script>
(function () {{
  var el = document.getElementById("{element_id}");
  if (!el || el.dataset.done) return;
  el.dataset.done = "1";
  var state = window.__avm || (window.__avm = {{}});
  var card = el.closest(".result");
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var target = Number(el.dataset.value), from = state.value, segs = card.querySelectorAll(".range-seg");
  if (!reduce && from === undefined) card.classList.add("enter");
  if (!reduce && state.segs) {{
    segs.forEach(function (s, i) {{ if (state.segs[i]) {{ s.style.left = state.segs[i][0]; s.style.width = state.segs[i][1]; }} }});
    requestAnimationFrame(function () {{ requestAnimationFrame(function () {{
      segs.forEach(function (s) {{ s.style.left = s.dataset.left; s.style.width = s.dataset.width; }});
    }}); }});
  }}
  state.segs = Array.prototype.map.call(segs, function (s) {{ return [s.dataset.left, s.dataset.width]; }});
  state.value = target;
  if (reduce || from === target) return;
  var final = el.textContent, start = from === undefined ? 0 : from;
  var duration = from === undefined ? 1100 : 500, t0 = performance.now();
  function show(v) {{
    el.textContent = v >= 1e6 ? "$" + (v / 1e6).toFixed(2) + "M"
                              : "$" + (Math.round(v / 1000) * 1000).toLocaleString("en-US");
  }}
  function tick(now) {{
    var t = Math.min(1, (now - t0) / duration), eased = 1 - Math.pow(1 - t, 3);
    if (t < 1) {{ show(start + (target - start) * eased); requestAnimationFrame(tick); }}
    else el.textContent = final;
  }}
  show(start);
  requestAnimationFrame(tick);
}})();
</script>"""


def facts_card(rows: dict[str, str]) -> str:
    items = "".join(f"<dt>{k}</dt><dd>{html.escape(v)}</dd>" for k, v in rows.items())
    return f'<section class="card"><p class="eyebrow">What the model used</p><dl class="facts">{items}</dl></section>'


def waterfall(bias: float, effects: pd.Series) -> alt.LayerChart:
    order = effects.reindex(effects.abs().sort_values(ascending=False).index)
    level = math.exp(bias)
    rows = [{"step": "Starting point", "start": level, "end": level, "kind": "total",
             "label": short_dollars(level), "change": ""}]
    for name, term in order.items():
        new = level * math.exp(term)
        effect = 100 * math.expm1(term)
        rows.append({"step": name, "start": level, "end": new, "kind": "up" if term >= 0 else "down",
                     "label": signed_pct(effect, 1 if abs(effect) < 10 else 0),
                     "change": ("+" if new >= level else "−") + short_dollars(abs(new - level))})
        level = new
    rows.append({"step": "Estimate", "start": level, "end": level, "kind": "total",
                 "label": short_dollars(level), "change": ""})
    data = pd.DataFrame(rows).assign(right=lambda t: t[["start", "end"]].max(axis=1))
    low, high = data[["start", "end"]].min().min(), data[["start", "end"]].max().max()
    pad = (high - low) * 0.06 or high * 0.05
    steps = data["step"].tolist()

    base = alt.Chart(data).encode(
        y=alt.Y("step:N", sort=steps, title=None, axis=alt.Axis(labelLimit=180, labelPadding=10,
                                                                 labelFontSize=13, labelColor=INK)),
        tooltip=[alt.Tooltip("step:N", title="Part"), alt.Tooltip("label:N", title="Effect"),
                 alt.Tooltip("change:N", title="Dollars, in this order")])
    x = alt.X("start:Q", title=None, scale=alt.Scale(domain=[low - pad, high + pad * 4], zero=False, nice=False),
              axis=alt.Axis(format="$.2s", tickCount=4, grid=True))
    color = alt.Color("kind:N", scale=alt.Scale(domain=["up", "down", "total"], range=[INK, "#a3a3a3", INK]),
                      legend=None)
    bars = base.transform_filter("datum.kind != 'total'").mark_bar(height=16, cornerRadius=2).encode(
        x=x, x2="end:Q", color=color)
    totals = base.transform_filter("datum.kind == 'total'").mark_tick(thickness=2.5, size=24).encode(
        x=x, color=color)
    labels = base.mark_text(align="left", dx=8, fontSize=12).encode(
        x="right:Q", text="label:N",
        color=alt.condition("datum.kind == 'down'", alt.value(SOFT), alt.value(INK)))
    return (bars + totals + labels).properties(height=36 * len(data))


def district_map(codes: tuple[str, ...], latitude: float, longitude: float) -> pdk.Deck:
    shapes = district_shapes(codes)
    xs, ys = [longitude], [latitude]
    for feature in shapes["features"]:
        x0, y0, x1, y1 = feature["bbox"] if "bbox" in feature else _bounds(feature["geometry"])
        xs += [x0, x1]
        ys += [y0, y1]
    span_x, span_y = max(max(xs) - min(xs), 0.02), max(max(ys) - min(ys), 0.02)
    # Web Mercator: 256 px per 360° at zoom 0. Leave some margin around the district.
    zoom = min(math.log2(420 * 360 / (256 * span_x)), math.log2(300 * 170 / (256 * span_y)))
    view = pdk.ViewState(latitude=(min(ys) + max(ys)) / 2, longitude=(min(xs) + max(xs)) / 2,
                         zoom=max(5, min(zoom - 0.3, 13)))
    layers = [
        pdk.Layer("GeoJsonLayer", shapes, stroked=True, filled=True, pickable=True,
                  get_fill_color=[10, 10, 10, 14], get_line_color=[10, 10, 10, 210],
                  line_width_min_pixels=1.5),
        pdk.Layer("ScatterplotLayer", [{"lon": longitude, "lat": latitude}], get_position=["lon", "lat"],
                  get_fill_color=[10, 10, 10], get_line_color=[255, 255, 255], stroked=True,
                  radius_min_pixels=7, line_width_min_pixels=2.5, get_radius=1),
    ]
    return pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style="light",
                    tooltip={"text": "{DistrictNa}"})


def _bounds(geometry: dict) -> tuple[float, float, float, float]:
    coords = np.array([pt for poly in (geometry["coordinates"] if geometry["type"] == "MultiPolygon"
                                       else [geometry["coordinates"]]) for ring in poly for pt in ring])
    return coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()


def estimate_page():
    page_head("California single-family homes", TITLE,
              f"What a home would likely sell for in {VALUATION_YEAR}, from a model trained on "
              f"{META['n_train']:,} CRMLS sales and tested on April 2026.")

    left, right = st.columns([1, 1.3], gap="large")
    with left, st.container(border=True, key="inputs"):
        st.html('<p class="eyebrow">The home</p>')
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

        exact = st.toggle("I know the exact coordinates",
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
    frame = predictor.predict(home, ART)
    result = frame.iloc[0]
    estimate = float(result["estimate"])
    notes = []
    if result["coordinates_from"] == "missing":
        notes.append(("caution", "The coordinates are outside California, so they are treated as missing, "
                                 "as in training."))

    band = BANDS.loc[(BANDS["pred_low"] <= estimate) & (estimate < BANDS["pred_high"])].iloc[0]
    notes += range_warnings({"LivingArea": living_area, "LotSizeSquareFeet": lot_size or np.nan,
                             "BedroomsTotal": beds, "BathroomsTotalInteger": baths, "YearBuilt": year_built})
    if ZIPS.at[zip_code, "sales"] < FEW_SALES:
        notes.append(("note", f"This ZIP has only {ZIPS.at[zip_code, 'sales']} sales in the data, so the "
                              "estimate leans on city, district and location more than on the ZIP itself."))
    ceiling = META["price_bounds"][1]
    top = EVAL["top_1pct"]
    if estimate >= 0.75 * ceiling:
        notes.append(("caution", f"This estimate is close to the highest price the model was trained on "
                                 f"({dollars(ceiling)}); it cannot predict above that. In the April test, homes "
                                 f"that sold for {dollars(top['price_cut'])} or more were under-predicted by a "
                                 f"median of {-top['median_signed_error_pct']:.0f}%."))

    district = result["SchoolDistrictName"] if isinstance(result["SchoolDistrictName"], str) else None
    with right:
        st.html(result_card(estimate, band, f"{zip_code} · {city.title()}" + (f" · {district}" if district else ""),
                            notes), unsafe_allow_javascript=True)
        st.html(facts_card({
            "Location": f"{zip_code}, {city.title()}",
            "Coordinates": ("missing" if result["coordinates_from"] == "missing" else
                            f"{result['Latitude']:.4f}, {result['Longitude']:.4f}"
                            + (" (ZIP median)" if result["coordinates_from"] == "ZIP median" else "")),
            "School district": district or "not found",
            "Age at sale": f"{VALUATION_YEAR - year_built} years (built {year_built}, valued in {VALUATION_YEAR})",
            "Size": f"{living_area:,} sq ft, lot {lot_size:,} sq ft" if lot_size else f"{living_area:,} sq ft, lot unknown",
            "Rooms": f"{beds} bed / {baths} bath",
        }))

    bias, effects = explain.contributions(ART.model, avm.model_inputs(frame, ART.features))
    chart_col, map_col = st.columns([1.25, 1], gap="large")
    with chart_col:
        section_head("Why this price", "What moved the estimate",
                     "Read top to bottom. The model starts from its average price, and each bar is how much "
                     "that part of the home pushed the estimate up (dark) or down (grey).")
        st.altair_chart(waterfall(bias, effects), theme=None, width="stretch")
        st.html('<p class="small-note">Effects come from XGBoost\'s per-feature contributions on the log price, '
                "so the percentages multiply. Dollar amounts in the tooltip depend on the order shown.</p>")
    with map_col:
        section_head("Where it is", district or "School district not found",
                     "The dot is the home, at the ZIP's median sale location unless you entered coordinates. "
                     "The outline is its school district.")
        if result["coordinates_from"] == "missing":
            st.html(notices_html([("note", "No map: the coordinates are treated as missing.")]))
        else:
            codes = tuple(str(result["SchoolDistrict"]).split("|")) if district else ()
            st.pydeck_chart(district_map(codes, result["Latitude"], result["Longitude"]), height=380,
                            alt=f"Map of {district or 'the home location'}")
            st.html('<p class="small-note">Boundaries: California School District Areas 2024–25. '
                    "Map tiles © CARTO, © OpenStreetMap contributors.</p>")


MODEL_NAMES = {**avm.DISPLAY_NAMES, avm.FOREST: "Random forest", "Random forest (Week 5 features)":
               "Random forest, Week 5 features"}


def scoreboard(april: pd.DataFrame) -> str:
    rows = []
    for _, r in april.iterrows():
        final = r["model"] == META["model"]
        tag = '<span class="pill">In this app</span>' if final else ""
        rows.append(
            f'<tr class="{"is-final" if final else ""}"><td class="model">{html.escape(r["Model"])}{tag}</td>'
            f'<td class="features">{html.escape(r["features"])}</td><td>{r["MdAPE_pct"]:.2f}%</td>'
            f'<td>{r["PPE10_pct"]:.1f}%</td><td>{r["PPE20_pct"]:.1f}%</td><td>{r["R2"]:.3f}</td>'
            f'<td>${r["MAE_USD"]:,.0f}</td><td>{r["N"]:,.0f}</td></tr>')
    head = "".join(f"<th>{h}</th>" for h in ["Model", "Features", "Typical error", "Within 10%", "Within 20%",
                                             "R²", "Mean error", "Homes"])
    return (f'<div class="table-scroll fade-in" tabindex="0" role="region" aria-label="April 2026 scoreboard">'
            f'<table class="board"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def comparison_page():
    page_head("April 2026 test", "Model comparison",
              "Every model was frozen before April 2026 was scored, then trained on all sales through March "
              "2026 and scored once on April.")
    scope = st.segmented_control(
        "Sales scored", ["filtered", "all"], default="filtered", required=True, key="scope",
        format_func=lambda s: {"filtered": f"Inside the training price range ({EVAL['coverage']:.1%})",
                               "all": "All April sales"}[s])
    april = (SUMMARY.query("split == 'test' and scope == @scope")
             .assign(Model=lambda t: t["model"].map(lambda m: MODEL_NAMES.get(m, m)))
             .sort_values("MdAPE_pct"))
    by_model = april.set_index("model")["MdAPE_pct"]
    final, baseline = by_model[META["model"]], by_model[avm.LINEAR]
    best = by_model.idxmin()

    title = f"Gradient boosting cut the typical error from {baseline:.1f}% to {final:.1f}%"
    if best != META["model"]:
        detail = (f"{best} was {final - by_model[best]:.2f} points lower on April. {META['model']} was picked on "
                  "March under a rule set before tuning: within 0.1 points, take the smaller model. Switching "
                  "after seeing April would have turned the test into a selection set.")
    else:
        detail = f"{META['model']} had the lowest typical error of every model on April."
    section_head("Scoreboard", title, detail)
    st.html(scoreboard(april))
    st.html('<p class="small-note">Typical error is the median absolute percentage error (MdAPE). Features: A is '
            "the Week 4–5 set (living area, lot size, bedrooms, bathrooms, city, ZIP). “B without ratios” adds "
            "age at sale, latitude, longitude and school district. Source: results/metrics_summary.csv.</p>")

    forest_gain = by_model["Random forest (Week 5 features)"] - by_model[avm.FOREST]
    boost_gain = by_model[avm.FOREST] - final
    bars_title = f"Location features cut {forest_gain:.1f} points; boosting cut {boost_gain:.1f} more"
    months = (SUMMARY.query("scope == @scope")
              .assign(Model=lambda t: t["model"].map(lambda m: MODEL_NAMES.get(m, m)),
                      Month=lambda t: pd.to_datetime(t["month"], format="%Y%m").dt.strftime("%b %Y"),
                      final=lambda t: t["model"] == META["model"],
                      layer=lambda t: t["final"].astype(int)))
    spread = months.groupby("model")["MdAPE_pct"].agg(lambda s: s.max() - s.min()).max()

    left, right = st.columns(2, gap="large")
    with left:
        section_head("By model", bars_title)
        april_chart = april.assign(final=lambda t: t["model"] == META["model"],
                                   value=lambda t: t["MdAPE_pct"].map("{:.2f}%".format))
        base = alt.Chart(april_chart).encode(
            y=alt.Y("Model:N", sort=april["Model"].tolist(), title=None, axis=alt.Axis(labelLimit=220, labelFontSize=12.5,
                                                                    labelColor=INK)),
            tooltip=["Model", alt.Tooltip("MdAPE_pct:Q", format=".2f", title="Typical error (%)")])
        bars = base.mark_bar(height=18, cornerRadiusEnd=2).encode(
            x=alt.X("MdAPE_pct:Q", title="Typical error on April 2026 (%)", scale=alt.Scale(domain=[0, 20])),
            color=alt.condition("datum.final", alt.value(INK), alt.value(FAINT)))
        text = base.mark_text(align="left", dx=6).encode(
            x="MdAPE_pct:Q", text="value:N",
            color=alt.condition("datum.final", alt.value(INK), alt.value(SOFT)))
        st.altair_chart((bars + text).properties(height=300), theme=None, width="stretch")
    with right:
        section_head("By month", f"No model's error moved more than {spread:.1f} points from month to month")
        order = ["Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026"]
        x = alt.X("Month:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelExpr=
                  "{'Dec 2025':'Dec','Jan 2026':'Jan','Feb 2026':'Feb','Mar 2026':'Mar','Apr 2026':'Apr'}[datum.label]"))
        y = alt.Y("MdAPE_pct:Q", title="Typical error (%)", scale=alt.Scale(domain=[0, 20]))
        lines = alt.Chart(months).mark_line(point=alt.OverlayMarkDef(size=28)).encode(
            x=x, y=y, detail="Model:N",
            color=alt.condition("datum.final", alt.value(INK), alt.value("#bdbdbd")),
            strokeWidth=alt.condition("datum.final", alt.value(2.5), alt.value(1.25)),
            order=alt.Order("layer:Q"),
            tooltip=["Model", "Month", alt.Tooltip("MdAPE_pct:Q", format=".2f", title="Typical error (%)")])
        last = months[months["Month"] == "Apr 2026"].sort_values("MdAPE_pct").copy()
        # Spread end labels so close lines do not overlap.
        placed = []
        for value in last["MdAPE_pct"]:
            placed.append(max(value, placed[-1] + 1.05) if placed else value)
        last["label_y"] = placed
        labels = alt.Chart(last).mark_text(align="left", dx=10, fontSize=11.5).encode(
            x=x, y="label_y:Q", text="Model:N",
            color=alt.condition("datum.final", alt.value(INK), alt.value(SOFT)))
        chart = (lines + labels).properties(height=300)
        st.altair_chart(chart, theme=None, width="stretch")
        st.html('<p class="small-note">December to February are backtests, March chose the model, '
                "April is the test.</p>")


def how_it_works_page():
    f, a = EVAL["filtered"], EVAL["all"]
    baseline = SUMMARY.query("split == 'test' and scope == 'filtered' and model == @avm.LINEAR")["MdAPE_pct"].iat[0]
    page_head("Method", "How it works",
              "What the model estimates, the data behind it, how it was tested, and where it falls short.")

    stats = [("Typical error", f"{f['MdAPE_pct']:.1f}%", "April 2026, homes in the training price range"),
             ("Within 10%", f"{f['PPE10_pct']:.0f}%", "of April homes"),
             ("Within 20%", f"{f['PPE20_pct']:.0f}%", "of April homes"),
             ("Baseline", f"{baseline:.1f}%", "Week 4 linear regression"),
             ("Sales", "309,039", "after cleaning, Jan 2024 – Apr 2026")]
    st.html('<dl class="stats fade-in">' + "".join(
        f'<div class="stat"><dt>{k}</dt><dd>{v}<small>{s}</small></dd></div>' for k, v, s in stats) + "</dl>")

    section_head("Pipeline", "From raw listings to one April score")
    steps = [("CRMLS data", "Closed single-family sales, Jan 2024 – Apr 2026"),
             ("Cleaning", "Duplicates, impossible records and re-reported sales removed"),
             ("Features", "Size, rooms, age, location, city, ZIP, school district"),
             ("Model", "XGBoost on log price, settings frozen on March"),
             ("April test", f"Scored once: {f['MdAPE_pct']:.1f}% typical error")]
    st.html('<ol class="pipeline">' + "".join(f"<li><strong>{t}</strong><span>{d}</span></li>" for t, d in steps)
            + "</ol>")

    with st.container(key="article"):
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
- **The most expensive homes.** The model cannot predict above {dollars(META['price_bounds'][1]).replace('$', '\\$')}, the
  top of its training range. In the April test, homes that sold for
  {dollars(EVAL['top_1pct']['price_cut']).replace('$', '\\$')} or more were under-predicted by a median of
  {-EVAL['top_1pct']['median_signed_error_pct']:.0f}%.
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


pages = [
    st.Page(estimate_page, title="Estimate", default=True),
    st.Page(comparison_page, title="Model comparison", url_path="model-comparison"),
    st.Page(how_it_works_page, title="How it works", url_path="how-it-works"),
]
current = st.navigation(pages, position="hidden")
st.html(f"<style>{stylesheet()}</style>")
with st.container(horizontal=True, key="topbar", vertical_alignment="center", horizontal_alignment="distribute"):
    st.html(f'<span class="brand">{TITLE.removesuffix(" Estimator")} <span>Estimator</span></span>', width="content")
    with st.container(horizontal=True, gap="small", width="content", key="nav"):
        for page in pages:
            st.page_link(page)
current.run()
