"""Clean the raw CRMLS files and attach school districts.

Run from the repository root:

    python -m src.preprocess

Writes data/processed/sales_clean.csv (all months in data/raw, ignored by git)
and prints the rows removed at each cleaning step. The other entry points do
this step themselves, so running it is only needed to inspect the cleaned data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avm  # noqa: E402
from config import DATA_DIR, DISTRICT_CACHE, DISTRICT_SHAPEFILE, PROCESSED_DIR, ROOT  # noqa: E402

OUTPUT = PROCESSED_DIR / "sales_clean.csv"
COLUMNS = ["ListingKey", "CloseDate", "CloseMonth", "SaleYear", avm.TARGET, "LivingArea",
           "LotSizeSquareFeet", "BedroomsTotal", "BathroomsTotalInteger", "YearBuilt", "Latitude",
           "Longitude", "City", "PostalCode", "County", "SchoolDistrict", "SchoolDistrictName",
           "SchoolDistrictType", "SourceFile"]


def main() -> None:
    df, steps, fingerprint = avm.load_clean_sales(DATA_DIR, DISTRICT_CACHE, DISTRICT_SHAPEFILE)
    print(f"Snapshot {fingerprint}")
    print(steps.to_string(index=False))
    rereported = steps.attrs["rereported"]
    print(f"Re-reported copies removed: {len(rereported)}")
    print("Sales per close month (last six):")
    print(df["CloseMonth"].value_counts().sort_index().tail(6).to_string())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df):,} sales to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
