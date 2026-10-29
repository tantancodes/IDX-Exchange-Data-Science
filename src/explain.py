"""Per-feature contributions for the deployed XGBoost model.

XGBoost's pred_contribs gives one additive term per model column plus a bias,
on the model's own scale (log price here). The terms for a home add up to its
log estimate exactly, so exp(bias) is the starting point and each group of
columns multiplies it by exp(sum of that group's terms).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

# Model column (after preprocessing) -> plain name. Missing-value indicators go
# with the column they flag.
GROUPS = {
    "LivingArea": "Living area",
    "LotSizeSquareFeet": "Lot size",
    "BedroomsTotal": "Bedrooms and bathrooms",
    "BathroomsTotalInteger": "Bedrooms and bathrooms",
    "AgeAtSale": "Age",
    "Latitude": "Exact location",
    "Longitude": "Exact location",
    "City": "City and ZIP code",
    "PostalCode": "City and ZIP code",
    "SchoolDistrict": "School district",
}


def contributions(model, x: pd.DataFrame) -> tuple[float, pd.Series]:
    """Starting log price and the log-price effect of each feature group, for one home.

    ``model`` is the saved TransformedTargetRegressor and ``x`` the raw inputs
    from avm.model_inputs.
    """
    assert len(x) == 1, "one home at a time"
    pipeline = model.regressor_
    prepared = pipeline[:-1].transform(x)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    terms = pipeline[-1].get_booster().predict(xgb.DMatrix(np.asarray(prepared, dtype=float)),
                                               pred_contribs=True)[0]
    bias, terms = float(terms[-1]), pd.Series(terms[:-1], index=names)
    column = terms.index.str.split("__").str[1].str.removeprefix("missingindicator_")
    effects = terms.groupby(column.map(GROUPS)).sum()
    assert np.isclose(bias + effects.sum(), np.log(model.predict(x)[0]), atol=1e-4)
    return bias, effects
