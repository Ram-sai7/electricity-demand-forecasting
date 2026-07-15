import itertools
import warnings

import numpy as np
import pandas as pd

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX


def search_sarima_orders(
    train: pd.Series,
    p_values=range(0, 7),
    d_values=range(0, 3),
    q_values=range(0, 7),
    seasonal_candidates=None,
    seasonal_period: int = 52,
    maxiter: int = 80,
) -> pd.DataFrame:
    """
    Search SARIMA parameter combinations using AIC and BIC.

    The assignment requires all p, d and q combinations:
    p = 0,...,6
    d = 0,...,2
    q = 0,...,6

    Seasonal candidates are restricted to a small set for
    computational feasibility.
    """

    if train.empty:
        raise ValueError("Training series is empty.")

    if seasonal_candidates is None:
        seasonal_candidates = [
            (0, 1, 0),
            (1, 1, 0),
            (0, 1, 1),
            (1, 1, 1),
        ]

    results = []

    total_models = (
        len(list(p_values))
        * len(list(d_values))
        * len(list(q_values))
        * len(seasonal_candidates)
    )

    counter = 0

    for P, D, Q in seasonal_candidates:
        for p, d, q in itertools.product(
            p_values,
            d_values,
            q_values,
        ):
            counter += 1

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore",
                        ConvergenceWarning,
                    )
                    warnings.simplefilter(
                        "ignore",
                        UserWarning,
                    )
                    warnings.simplefilter(
                        "ignore",
                        RuntimeWarning,
                    )

                    model = SARIMAX(
                        train,
                        order=(p, d, q),
                        seasonal_order=(
                            P,
                            D,
                            Q,
                            seasonal_period,
                        ),
                        trend="n",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )

                    fitted = model.fit(
                        disp=False,
                        maxiter=maxiter,
                        method="lbfgs",
                    )

                results.append(
                    {
                        "p": p,
                        "d": d,
                        "q": q,
                        "P": P,
                        "D": D,
                        "Q": Q,
                        "seasonal_period": seasonal_period,
                        "AIC": fitted.aic,
                        "BIC": fitted.bic,
                        "converged": fitted.mle_retvals.get(
                            "converged",
                            False,
                        ),
                    }
                )

            except Exception:
                continue

            if counter % 25 == 0:
                print(
                    f"Completed {counter}/{total_models} models"
                )

    if not results:
        raise RuntimeError(
            "No SARIMA models were fitted successfully."
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    results_df = results_df.dropna(
        subset=["AIC"]
    )

    results_df = results_df.sort_values(
        "AIC"
    ).reset_index(drop=True)

    return results_df


def select_best_converged_order(
    search_results: pd.DataFrame,
) -> tuple[tuple, tuple]:
    """
    Select the lowest-AIC converged SARIMA model.
    """

    required_columns = {
        "p",
        "d",
        "q",
        "P",
        "D",
        "Q",
        "seasonal_period",
        "AIC",
        "converged",
    }

    missing_columns = (
        required_columns
        - set(search_results.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    converged = search_results[
        search_results["converged"] == True
    ].copy()

    if converged.empty:
        raise RuntimeError(
            "No converged SARIMA model was found."
        )

    best = converged.sort_values(
        "AIC"
    ).iloc[0]

    order = (
        int(best["p"]),
        int(best["d"]),
        int(best["q"]),
    )

    seasonal_order = (
        int(best["P"]),
        int(best["D"]),
        int(best["Q"]),
        int(best["seasonal_period"]),
    )

    return order, seasonal_order


def fit_sarima(
    train: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    maxiter: int = 300,
):
    """
    Fit a SARIMA model using the selected orders.
    """

    if train.empty:
        raise ValueError(
            "Training series is empty."
        )

    model = SARIMAX(
        train,
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


def forecast_sarima(
    fitted_model,
    horizon: int,
    index: pd.Index,
    alpha: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Generate SARIMA forecasts and confidence intervals.
    """

    forecast_result = fitted_model.get_forecast(
        steps=horizon
    )

    prediction = forecast_result.predicted_mean
    prediction.index = index
    prediction.name = "sarima"

    confidence_interval = forecast_result.conf_int(
        alpha=alpha
    )
    confidence_interval.index = index

    return prediction, confidence_interval