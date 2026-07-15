import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def rmse(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calculate root mean squared error.
    """

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def forecast_bias(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Calculate average signed forecast error.

    Positive bias means the model tends to over-forecast.
    """

    return float(
        np.mean(
            y_pred.values
            - y_true.values
        )
    )


def mase(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    seasonality: int = 52,
) -> float:
    """
    Calculate mean absolute scaled error.

    The scaling denominator uses the in-sample seasonal naive error.
    """

    if len(y_train) <= seasonality:
        raise ValueError(
            "Training series is too short for the selected seasonality."
        )

    seasonal_errors = np.abs(
        y_train.iloc[seasonality:].values
        - y_train.iloc[:-seasonality].values
    )

    scale = seasonal_errors.mean()

    if np.isclose(scale, 0):
        raise ValueError(
            "MASE scale is zero."
        )

    model_error = np.mean(
        np.abs(
            y_true.values
            - y_pred.values
        )
    )

    return float(
        model_error / scale
    )


def evaluate_forecast(
    name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: pd.Series,
    seasonality: int = 52,
) -> dict:
    """
    Evaluate a forecast using MAE, RMSE, MASE and bias.
    """

    y_pred = y_pred.reindex(
        y_true.index
    )

    aligned = pd.concat(
        [
            y_true.rename("actual"),
            y_pred.rename("forecast"),
        ],
        axis=1,
    ).dropna()

    if aligned.empty:
        raise ValueError(
            f"No aligned observations available for model: {name}"
        )

    actual = aligned["actual"]
    forecast = aligned["forecast"]

    return {
        "model": name,
        "MAE": float(
            mean_absolute_error(
                actual,
                forecast,
            )
        ),
        "RMSE": rmse(
            actual,
            forecast,
        ),
        "MASE": mase(
            actual,
            forecast,
            y_train,
            seasonality,
        ),
        "Bias": forecast_bias(
            actual,
            forecast,
        ),
    }