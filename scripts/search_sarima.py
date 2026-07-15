import json

import pandas as pd

from electricity_demand.config import (
    MODEL_DIR,
    SEASONAL_PERIOD,
    TEST_WEEKS,
)
from electricity_demand.data import load_processed_data
from electricity_demand.models.sarima import (
    search_sarima_orders,
    select_best_converged_order,
)


def main():
    """
    Run the SARIMA AIC grid search and save the selected order.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_file = (
        MODEL_DIR / "sarima_grid_results.csv"
    )

    order_file = (
        MODEL_DIR / "selected_sarima_order.json"
    )

    data = load_processed_data()

    y = data["load_gw"]

    train = y.iloc[:-TEST_WEEKS]

    print("Training observations:", len(train))

    if results_file.exists():
        print(
            "Existing SARIMA grid results found."
        )
        print(
            "Loading:",
            results_file,
        )

        search_results = pd.read_csv(
            results_file
        )

    else:
        print(
            "No saved grid results found."
        )
        print(
            "Starting SARIMA grid search."
        )
        print(
            "This may take several hours."
        )

        search_results = search_sarima_orders(
            train=train,
            p_values=range(0, 7),
            d_values=range(0, 3),
            q_values=range(0, 7),
            seasonal_candidates=[
                (0, 1, 0),
                (1, 1, 0),
                (0, 1, 1),
                (1, 1, 1),
            ],
            seasonal_period=SEASONAL_PERIOD,
            maxiter=80,
        )

        search_results.to_csv(
            results_file,
            index=False,
        )

        print(
            "Saved SARIMA grid results to:",
            results_file,
        )

    print("\nTop 10 models by AIC:")
    print(
        search_results
        .sort_values("AIC")
        .head(10)
        .to_string(index=False)
    )

    best_order, best_seasonal_order = (
        select_best_converged_order(
            search_results
        )
    )

    selected = {
        "order": list(best_order),
        "seasonal_order": list(
            best_seasonal_order
        ),
    }

    with open(
        order_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected,
            file,
            indent=4,
        )

    print("\nSelected SARIMA order:")
    print(best_order)

    print("Selected seasonal order:")
    print(best_seasonal_order)

    print("\nSaved selected order to:")
    print(order_file)


if __name__ == "__main__":
    main()