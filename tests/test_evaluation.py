import pandas as pd

from electricity_demand.evaluation import (
    forecast_bias,
    rmse,
)


def test_perfect_forecast_has_zero_error():
    actual = pd.Series(
        [1.0, 2.0, 3.0]
    )

    prediction = actual.copy()

    assert rmse(actual, prediction) == 0.0
    assert forecast_bias(actual, prediction) == 0.0