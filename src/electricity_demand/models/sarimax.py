import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarimax(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    order: tuple,
    seasonal_order: tuple,
    maxiter: int = 300,
):
    """
    Fit a SARIMAX model using exogenous regressors.
    """

    if y_train.empty:
        raise ValueError(
            "Training target series is empty."
        )

    if X_train.empty:
        raise ValueError(
            "Training exogenous data is empty."
        )

    if len(y_train) != len(X_train):
        raise ValueError(
            "Target and exogenous training data "
            "must have the same number of rows."
        )

    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(
        disp=False,
        maxiter=maxiter,
        method="lbfgs",
    )

    return fitted


def forecast_sarimax(
    fitted_model,
    X_test: pd.DataFrame,
    index: pd.Index,
    alpha: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Generate SARIMAX forecasts and confidence intervals.

    If observed future temperature is supplied in X_test,
    this is a conditional or explanatory forecast.
    """

    if X_test.empty:
        raise ValueError(
            "Test exogenous data is empty."
        )

    forecast_result = fitted_model.get_forecast(
        steps=len(X_test),
        exog=X_test,
    )

    prediction = forecast_result.predicted_mean
    prediction.index = index
    prediction.name = "sarimax"

    confidence_interval = forecast_result.conf_int(
        alpha=alpha
    )
    confidence_interval.index = index

    return prediction, confidence_interval