from electricity_demand.config import START_DATE
from electricity_demand.data import (
    aggregate_load,
    load_hourly_load,
    save_processed_weekly_data,
)
from electricity_demand.features import (
    get_berlin_temperature,
    make_ml_features,
    make_weekly_temperature_features,
)


def main():
    print("Loading hourly electricity data...")

    hourly = load_hourly_load()

    print(f"Hourly observations: {len(hourly)}")

    daily, weekly = aggregate_load(hourly)

    print(f"Daily observations: {len(daily)}")
    print(f"Weekly observations: {len(weekly)}")

    print("Downloading Berlin temperature data...")

    temperature_daily = get_berlin_temperature(
        start_date=START_DATE,
        end_date=str(weekly.index.max().date()),
    )

    print(f"Temperature observations: {len(temperature_daily)}")

    temperature_weekly = make_weekly_temperature_features(
        temperature_daily
    )

    weekly_data = weekly.to_frame()

    weekly_data = weekly_data.join(
        temperature_weekly,
        how="left",
    )

    weekly_data = weekly_data.interpolate(method="time")
    weekly_data = weekly_data.dropna()

    print("Combined weekly data shape:", weekly_data.shape)

    modelling_data = make_ml_features(weekly_data)

    print("Modelling data shape:", modelling_data.shape)

    output_path = save_processed_weekly_data(modelling_data)

    print("\nProcessed dataset saved to:")
    print(output_path)

    print("\nFirst five rows:")
    print(modelling_data.head())


if __name__ == "__main__":
    main()