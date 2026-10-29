"""Score the frozen final model on a held-out month, the way notebook 07 does.

Run from the repository root:

    python -m src.evaluate                 # reproduces the April 2026 test
    python -m src.evaluate --month 202603  # any month: trains on all earlier months

The model trains on every month before --month (within the training months'
0.5th–99.5th percentile price bounds) and is scored on that month. For April
2026 the result is checked against results/metrics_summary.csv. This
reproduces the locked test; it is not a way to choose between models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avm  # noqa: E402
from config import RESULTS_DIR  # noqa: E402
from train import final_model, load_sales  # noqa: E402

COLUMNS = ["R2", "MAE_USD", "RMSE_USD", "MAPE_pct", "MdAPE_pct", "PPE10_pct", "PPE20_pct",
           "median_bias_pct", "N"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--month", default=avm.TEAM_TEST_MONTH, help="month to score, YYYYMM")
    month = parser.parse_args().month

    df, fingerprint = load_sales()
    config, features, factory = final_model()
    run = avm.run_cutoff(df, month, {config["model"]: factory}, inputs=features)
    result = run["results"][config["model"]]
    table = pd.DataFrame({scope: result[scope] for scope in ("filtered", "all")}).T[COLUMNS]
    print(f"{config['model']} ({features.name}), snapshot {fingerprint}: trained on {run['n_train']:,} "
          f"sales {run['train_months'][0]}–{run['train_months'][1]}, scored on {month}; "
          f"price bounds ${run['lower']:,.0f}–${run['upper']:,.0f}, coverage {run['coverage']:.1%}")
    print(avm.format_table(table, pct_decimals=2).to_string())

    if month == avm.TEAM_TEST_MONTH:
        saved = pd.read_csv(RESULTS_DIR / "metrics_summary.csv", dtype={"month": str}).query(
            "split == 'test' and model == @config['model']").set_index("scope")
        for scope in ("filtered", "all"):
            assert np.isclose(result[scope]["MdAPE_pct"], saved.loc[scope, "MdAPE_pct"], atol=1e-6)
            assert result[scope]["N"] == saved.loc[scope, "N"]
        print("Matches the April 2026 test in results/metrics_summary.csv")


if __name__ == "__main__":
    main()
