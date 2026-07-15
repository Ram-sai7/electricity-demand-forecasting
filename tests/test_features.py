import pandas as pd

from electricity_demand.features import make_ml_features


def test_lag_feature_uses_past_value():
    index = pd.date_range(
        "2020-01-05",
        periods=60,
        freq="W-SUN",
    )

    data = pd.DataFrame(
        {
            "load_gw": range(60),
            "temp_mean": range(60),
        },
        index=index,
    )

    result = make_ml_features(data)

    first_valid_date = result.index[0]
    original_position = data.index.get_loc(
        first_valid_date
    )

    expected = data.iloc[
        original_position - 1
    ]["load_gw"]

    actual = result.loc[
        first_valid_date,
        "load_lag_1",
    ]

    assert actual == expected