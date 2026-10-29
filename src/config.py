"""Project paths and fixed reference values shared by the src entry points."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DISTRICT_SHAPEFILE = ROOT / "data" / "external" / "school_districts_2024_25" / "DistrictAreas2425.shp"
DISTRICT_CACHE = PROCESSED_DIR / "school_district_by_location.csv"
RESULTS_DIR = ROOT / "results"
APP_DIR = ROOT / "models" / "app"

# Fingerprint of the 28 CRMLSSold files, January 2024 – April 2026.
EXPECTED_FINGERPRINT = "66cab54017ab"
# Commit that froze the final model before the April 2026 test.
LOCK_COMMIT = "8d41984"
