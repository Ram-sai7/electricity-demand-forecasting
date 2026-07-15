from pathlib import Path

import pandas as pd

from .config import (
    LOAD_COLUMN,
    POWER_DATA_FILE,
    POWER_DATA_URL,
    PROCESSED_WEEKLY_FILE,
    START_DATE,
)


def download_power_data(
    url: str = POWER_DATA_URL,
    output_path: Path = POWER_DATA_FILE,
) -> Path:
    """
    Download the OPSD hourly electricity-demand dataset.

    If the file already exists, it is reused instead of downloaded again.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        print(f"Raw data already exists: {output_path}")
        return output_path

    print("Downloading electricity demand data...")

    data = pd.read_csv(url)

    data.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved raw data to: {output_path}")

    return output_path


def load_hourly_load(
    file_path: Path = POWER_DATA_FILE,
    start_date: str = START_DATE,
) -> pd.Series:
    """
    Load and clean German hourly electricity demand.

    Parameters
    ----------
    file_path:
        Path to the OPSD hourly CSV file.

    start_date:
        First date to retain, in YYYY-MM-DD format.

    Returns
    -------
    pd.Series
        Hourly German electricity load in MW.
    """

    if not file_path.exists():
        download_power_data(
            output_path=file_path,
        )

    data = pd.read_csv(
        file_path,
        usecols=[
            "utc_timestamp",
            LOAD_COLUMN,
        ],
        parse_dates=["utc_timestamp"],
    )

    data = data.rename(
        columns={
            "utc_timestamp": "datetime",
            LOAD_COLUMN: "load_mw",
        }
    )

    data = data.set_index("datetime").sort_index()

    # Convert timezone-aware UTC timestamps to timezone-naive timestamps.
    # This allows the load data to be joined with timezone-naive weather data.
    if data.index.tz is not None:
        data.index = data.index.tz_convert(None)

    data = data.loc[start_date:]

    data["load_mw"] = pd.to_numeric(
        data["load_mw"],
        errors="coerce",
    )

    missing_before = data["load_mw"].isna().sum()

    data["load_mw"] = data["load_mw"].interpolate(
        method="time",
    )

    missing_after = data["load_mw"].isna().sum()

    if missing_before > 0:
        print(
            f"Interpolated missing load values: "
            f"{missing_before - missing_after}"
        )

    hourly = data["load_mw"].dropna().copy()
    hourly.name = "load_mw"

    if hourly.empty:
        raise ValueError(
            "The cleaned hourly electricity load series is empty."
        )

    return hourly


def aggregate_load(
    hourly: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Aggregate hourly electricity load to daily and weekly mean demand.

    The original load values are in MW. Aggregated series are converted to GW.

    Parameters
    ----------
    hourly:
        Hourly electricity load in MW.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Daily and weekly average load in GW.
    """

    if hourly.empty:
        raise ValueError(
            "Cannot aggregate an empty hourly series."
        )

    daily = hourly.resample("D").mean() / 1000
    weekly = hourly.resample("W-SUN").mean() / 1000

    daily.name = "load_gw"
    weekly.name = "load_gw"

    return daily.dropna(), weekly.dropna()


def save_processed_weekly_data(
    weekly_data: pd.DataFrame,
    output_path: Path = PROCESSED_WEEKLY_FILE,
) -> Path:
    """
    Save the processed weekly modelling dataset to CSV.

    Parameters
    ----------
    weekly_data:
        Processed weekly modelling table.

    output_path:
        Location where the CSV file will be saved.

    Returns
    -------
    Path
        Path to the saved file.
    """

    if weekly_data.empty:
        raise ValueError(
            "Cannot save an empty processed dataset."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly_data.to_csv(
        output_path,
        index=True,
    )

    print(
        f"Saved processed weekly data to: "
        f"{output_path}"
    )

    return output_path


def load_processed_data(
    file_path: Path = PROCESSED_WEEKLY_FILE,
) -> pd.DataFrame:
    """
    Load the processed weekly modelling dataset.

    Parameters
    ----------
    file_path:
        Path to the processed weekly CSV file.

    Returns
    -------
    pd.DataFrame
        Processed weekly modelling data.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {file_path}. "
            "Run scripts/make_features.py first."
        )

    data = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True,
    )

    data = data.sort_index()

    if data.index.tz is not None:
        data.index = data.index.tz_convert(None)

    if "load_gw" not in data.columns:
        raise ValueError(
            "Processed dataset must contain a 'load_gw' column."
        )

    if data["load_gw"].isna().any():
        raise ValueError(
            "Processed dataset contains missing target values."
        )

    return data