import json

import pandas as pd

from .config import (
    FIGURE_DIR,
    FORECAST_DIR,
    METRICS_DIR,
    MODEL_DIR,
    SEASONAL_PERIOD,
    TEST_WEEKS,
)
from .data import load_processed_data
from .evaluation import evaluate_forecast
from .models.benchmarks import (
    drift_forecast,
    mean_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)
from .models.feature_models import (
    fit_gradient_boosting,
    fit_random_forest,
    predict_feature_model,
)
from .models.sarimax import (
    fit_sarimax,
    forecast_sarimax,
)
from .plotting import plot_forecasts


def load_selected_sarima_orders() -> tuple[tuple, tuple]:
    """
    Load selected SARIMA orders from a JSON file.

    If the file does not exist, a default order is used.
    Replace the default later with the final selected model.
    """

    order_file = MODEL_DIR / "selected_sarima_order.json"

    if order_file.exists():
        with open(order_file, "r", encoding="utf-8") as file:
            saved = json.load(file)

        order = tuple(saved["order"])
        seasonal_order = tuple(saved["seasonal_order"])

        return order, seasonal_order

    print(
        "Warning: selected_sarima_order.json was not found. "
        "Using a temporary default SARIMAX specification."
    )

    return (1, 1, 1), (1, 1, 1, 52)


def run_pipeline(
    test_weeks: int = TEST_WEEKS,
    include_sarimax: bool = True,
    include_feature_models: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the weekly German electricity-demand forecasting workflow.

    Steps
    -----
    1. Load processed weekly data.
    2. Split into training and test periods.
    3. Generate benchmark forecasts.
    4. Fit SARIMAX using temperature variables.
    5. Fit Random Forest and Gradient Boosting models.
    6. Evaluate all model forecasts.
    7. Save forecasts, metrics and the final comparison plot.
    """

    data = load_processed_data()

    if "load_gw" not in data.columns:
        raise ValueError(
            "Processed data must contain a 'load_gw' column."
        )

    if len(data) <= test_weeks:
        raise ValueError(
            "The processed dataset is too short for the requested test horizon."
        )

    y = data["load_gw"]

    train = y.iloc[:-test_weeks]
    test = y.iloc[-test_weeks:]

    horizon = len(test)

    forecasts = {}
    metrics = []

    # -----------------------------------------------------
    # Benchmark models
    # -----------------------------------------------------

    forecasts["mean"] = mean_forecast(
        train=train,
        horizon=horizon,
        index=test.index,
    )

    forecasts["naive"] = naive_forecast(
        train=train,
        horizon=horizon,
        index=test.index,
    )

    forecasts["seasonal_naive"] = seasonal_naive_forecast(
        train=train,
        horizon=horizon,
        seasonality=SEASONAL_PERIOD,
        index=test.index,
    )

    forecasts["drift"] = drift_forecast(
        train=train,
        horizon=horizon,
        index=test.index,
    )

    # -----------------------------------------------------
    # Selected SARIMA orders
    # -----------------------------------------------------

    best_order, best_seasonal_order = load_selected_sarima_orders()

    print("SARIMA order:", best_order)
    print("Seasonal order:", best_seasonal_order)

    # -----------------------------------------------------
    # SARIMAX conditional forecast
    # -----------------------------------------------------

    if include_sarimax:
        exogenous_columns = [
            column
            for column in [
                "temp_mean",
                "heating_degree_days",
                "cooling_degree_days",
            ]
            if column in data.columns
        ]

        if exogenous_columns:
            X_exog = data[exogenous_columns]

            X_train_exog = X_exog.iloc[:-test_weeks]
            X_test_exog = X_exog.iloc[-test_weeks:]

            sarimax_fit = fit_sarimax(
                y_train=train,
                X_train=X_train_exog,
                order=best_order,
                seasonal_order=best_seasonal_order,
            )

            sarimax_pred, sarimax_ci = forecast_sarimax(
                fitted_model=sarimax_fit,
                X_test=X_test_exog,
                index=test.index,
            )

            forecasts["sarimax"] = sarimax_pred

            sarimax_ci.to_csv(
                FORECAST_DIR / "sarimax_confidence_intervals.csv"
            )

        else:
            print(
                "SARIMAX skipped because no temperature columns were found."
            )

    # -----------------------------------------------------
    # Feature-based models
    # -----------------------------------------------------

    if include_feature_models:
        feature_columns = [
            column
            for column in data.columns
            if column != "load_gw"
        ]

        if feature_columns:
            X = data[feature_columns]

            X_train = X.iloc[:-test_weeks]
            X_test = X.iloc[-test_weeks:]

            rf_model = fit_random_forest(
                X_train=X_train,
                y_train=train,
            )

            gb_model = fit_gradient_boosting(
                X_train=X_train,
                y_train=train,
            )

            forecasts["random_forest"] = predict_feature_model(
                model=rf_model,
                X_test=X_test,
                index=test.index,
                name="random_forest",
            )

            forecasts["gradient_boosting"] = predict_feature_model(
                model=gb_model,
                X_test=X_test,
                index=test.index,
                name="gradient_boosting",
            )

        else:
            print(
                "Feature models skipped because no feature columns were found."
            )

    # -----------------------------------------------------
    # Evaluate models
    # -----------------------------------------------------

    for model_name, prediction in forecasts.items():
        model_metrics = evaluate_forecast(
            name=model_name,
            y_true=test,
            y_pred=prediction,
            y_train=train,
            seasonality=SEASONAL_PERIOD,
        )

        metrics.append(model_metrics)

    metrics_df = (
        pd.DataFrame(metrics)
        .sort_values("MASE")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Build forecast table
    # -----------------------------------------------------

    forecast_df = pd.DataFrame(
        {
            "actual": test,
        }
    )

    for model_name, prediction in forecasts.items():
        forecast_df[model_name] = prediction.reindex(
            test.index
        )

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_df.to_csv(
        FORECAST_DIR / "all_forecasts.csv"
    )

    metrics_df.to_csv(
        METRICS_DIR / "model_comparison.csv",
        index=False,
    )

    plot_forecasts(
        train=train,
        test=test,
        forecasts=forecasts,
        output_path=(
            FIGURE_DIR / "forecast_comparison.png"
        ),
    )

    return metrics_df, forecast_df