from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecasts(
    train: pd.Series,
    test: pd.Series,
    forecasts: dict[str, pd.Series],
    output_path: Path | None = None,
):
    """
    Plot training data, actual test values and model forecasts.
    """

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    ax.plot(
        train.index,
        train,
        label="Training data",
        linewidth=1,
    )

    ax.plot(
        test.index,
        test,
        label="Actual test data",
        linewidth=2,
    )

    for model_name, forecast in forecasts.items():
        aligned_forecast = forecast.reindex(
            test.index
        )

        ax.plot(
            aligned_forecast.index,
            aligned_forecast,
            label=model_name,
        )

    ax.set_title(
        "German Weekly Electricity Demand Forecast Comparison"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Average load, GW")
    ax.legend(
        ncol=2,
        fontsize=9,
    )

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig