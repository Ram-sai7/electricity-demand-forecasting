from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "model_objects"

POWER_DATA_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)

POWER_DATA_FILE = (
    RAW_DATA_DIR / "time_series_60min_singleindex.csv"
)

LOAD_COLUMN = "DE_load_actual_entsoe_transparency"

PROCESSED_WEEKLY_FILE = (
    PROCESSED_DATA_DIR / "weekly_electricity_data.csv"
)

START_DATE = "2015-01-01"
TEST_WEEKS = 104
SEASONAL_PERIOD = 52
RANDOM_STATE = 42