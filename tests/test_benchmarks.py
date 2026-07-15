import pandas as pd

from electricity_demand.models.benchmarks import (
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)


def test_benchmark_forecast_lengths():
    train = pd.Series(range(104))

    forecast_index = pd.date_range(
        "2020-01-05",
        periods=10,
        freq="W-SUN",
    )

    assert len(
        mean_forecast(
            train,
            10,
            forecast_index,
        )
    ) == 10

    assert len(
        naive_forecast(
            train,
            10,
            forecast_index,
        )
    ) == 10

    assert len(
        seasonal_naive_forecast(
            train,
            10,
            52,
            forecast_index,
        )
    ) == 10