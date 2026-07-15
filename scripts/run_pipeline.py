from electricity_demand.pipeline import run_pipeline


def main():
    """
    Run the complete weekly forecasting pipeline.
    """

    metrics, forecasts = run_pipeline()

    print("\nModel comparison:")
    print(metrics.to_string(index=False))

    print("\nForecast output preview:")
    print(forecasts.head())

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()