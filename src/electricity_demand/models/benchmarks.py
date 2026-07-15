import numpy as np
import pandas as pd


def mean_forecast(
    train: pd.Series,
    horizon: int,
    index: pd.Index,
) -> pd.Series:
    """
    Forecast all future observations using the training mean.
    """

    values = np.repeat(
        train.mean(),
        horizon,
    )

    return pd.Series(
        values,
        index=index,
        name="mean",
    )


def naive_forecast(
    train: pd.Series,
    horizon: int,
    index: pd.Index,
) -> pd.Series:
    """
    Forecast all future observations using the final training value.
    """

    values = np.repeat(
        train.iloc[-1],
        horizon,
    )

    return pd.Series(
        values,
        index=index,
        name="naive",
    )


def seasonal_naive_forecast(
    train: pd.Series,
    horizon: int,
    seasonality: int,
    index: pd.Index,
) -> pd.Series:
    """
    Forecast by repeating observations from the previous seasonal cycle.

    For weekly data, seasonality=52 corresponds to one year.
    """

    if len(train) < seasonality:
        raise ValueError(
            "Training data must contain at least one full seasonal cycle."
        )

    values = []

    for step in range(horizon):
        seasonal_position = -seasonality + (step % seasonality)
        values.append(train.iloc[seasonal_position])

    return pd.Series(
        values,
        index=index,
        name="seasonal_naive",
    )


def drift_forecast(
    train: pd.Series,
    horizon: int,
    index: pd.Index,
) -> pd.Series:
    """
    Forecast using a straight-line drift from the first training value
    to the final training value.
    """

    if len(train) < 2:
        raise ValueError(
            "Drift forecast requires at least two training observations."
        )

    slope = (
        train.iloc[-1] - train.iloc[0]
    ) / (len(train) - 1)

    steps = np.arange(
        1,
        horizon + 1,
    )

    values = (
        train.iloc[-1]
        + slope * steps
    )

    return pd.Series(
        values,
        index=index,
        name="drift",
    )