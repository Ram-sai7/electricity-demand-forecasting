import numpy as np
import pandas as pd
import requests


def get_berlin_temperature(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download daily mean temperature for Berlin from Open-Meteo.

    Parameters
    ----------
    start_date:
        Start date in YYYY-MM-DD format.

    end_date:
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Daily Berlin temperature data with a timezone-naive index.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin",
    }

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    daily_data = response.json()["daily"]

    temperature = pd.DataFrame(
        {
            "date": pd.to_datetime(daily_data["time"]),
            "temp_mean_daily": daily_data["temperature_2m_mean"],
        }
    )

    temperature = temperature.set_index("date").sort_index()

    # Ensure index is timezone-naive so it can be joined
    # with the electricity-load index.
    if temperature.index.tz is not None:
        temperature.index = temperature.index.tz_convert(None)

    return temperature


def make_weekly_temperature_features(
    daily_temperature: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert daily temperature into weekly temperature features.

    Features created:
    - weekly mean temperature
    - weekly minimum temperature
    - weekly maximum temperature
    - heating degree days
    - cooling degree days
    """

    if daily_temperature.empty:
        raise ValueError(
            "Daily temperature data is empty."
        )

    if "temp_mean_daily" not in daily_temperature.columns:
        raise ValueError(
            "Expected column 'temp_mean_daily' was not found."
        )

    temperature = daily_temperature["temp_mean_daily"].copy()

    weekly = pd.DataFrame()

    weekly["temp_mean"] = (
        temperature
        .resample("W-SUN")
        .mean()
    )

    weekly["temp_min"] = (
        temperature
        .resample("W-SUN")
        .min()
    )

    weekly["temp_max"] = (
        temperature
        .resample("W-SUN")
        .max()
    )

    heating_base = 15.5
    cooling_base = 22.0

    heating_degree_daily = np.maximum(
        heating_base - temperature,
        0,
    )

    cooling_degree_daily = np.maximum(
        temperature - cooling_base,
        0,
    )

    weekly["heating_degree_days"] = (
        heating_degree_daily
        .resample("W-SUN")
        .sum()
    )

    weekly["cooling_degree_days"] = (
        cooling_degree_daily
        .resample("W-SUN")
        .sum()
    )

    if weekly.index.tz is not None:
        weekly.index = weekly.index.tz_convert(None)

    return weekly


def make_ml_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create supervised-learning features for weekly load forecasting.

    Lag and rolling features use only past load values.
    """

    if data.empty:
        raise ValueError(
            "Input data is empty."
        )

    if "load_gw" not in data.columns:
        raise ValueError(
            "Input data must contain a 'load_gw' column."
        )

    output = data.copy()

    if output.index.tz is not None:
        output.index = output.index.tz_convert(None)

    # Load lag features
    load_lags = [1, 2, 4, 8, 13, 26, 52]

    for lag in load_lags:
        output[f"load_lag_{lag}"] = (
            output["load_gw"].shift(lag)
        )

    # Rolling load features
    shifted_load = output["load_gw"].shift(1)

    rolling_windows = [4, 13, 26, 52]

    for window in rolling_windows:
        output[f"load_roll_mean_{window}"] = (
            shifted_load
            .rolling(window)
            .mean()
        )

        output[f"load_roll_std_{window}"] = (
            shifted_load
            .rolling(window)
            .std()
        )

    # Temperature lag features
    if "temp_mean" in output.columns:
        for lag in [1, 2, 4]:
            output[f"temp_lag_{lag}"] = (
                output["temp_mean"].shift(lag)
            )

    # Calendar features
    week_number = (
        output.index
        .isocalendar()
        .week
        .astype(int)
    )

    output["week_number"] = week_number
    output["month"] = output.index.month
    output["year"] = output.index.year

    # Fourier terms for annual seasonality
    for harmonic in range(1, 5):
        output[f"week_sin_{harmonic}"] = np.sin(
            2
            * np.pi
            * harmonic
            * week_number
            / 52
        )

        output[f"week_cos_{harmonic}"] = np.cos(
            2
            * np.pi
            * harmonic
            * week_number
            / 52
        )

    output = output.dropna()

    return output