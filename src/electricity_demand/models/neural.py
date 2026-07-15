import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    LSTM,
)
from tensorflow.keras.optimizers import Adam


def make_windows(
    values: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create one-step-ahead input and target windows for an LSTM.

    Parameters
    ----------
    values:
        Scaled time-series values. Expected shape is either
        (n_observations,) or (n_observations, 1).

    lookback:
        Number of previous hourly observations supplied to the model.

    Returns
    -------
    X:
        LSTM input array with shape
        (n_samples, lookback, 1).

    y:
        Target array with shape
        (n_samples,).
    """

    values = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1)

    if lookback <= 0:
        raise ValueError(
            "lookback must be greater than zero."
        )

    if len(values) <= lookback:
        raise ValueError(
            "The series must be longer than the lookback window."
        )

    X = []
    y = []

    for position in range(
        lookback,
        len(values),
    ):
        X.append(
            values[position - lookback:position]
        )

        y.append(
            values[position]
        )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1,
    )

    return X, y


def build_lstm(
    lookback: int,
    units_1: int = 32,
    units_2: int = 16,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> Sequential:
    """
    Build and compile a two-layer LSTM model.

    Parameters
    ----------
    lookback:
        Number of hourly observations in each input sequence.

    units_1:
        Number of units in the first LSTM layer.

    units_2:
        Number of units in the second LSTM layer.

    dropout:
        Dropout rate applied after each LSTM layer.

    learning_rate:
        Adam optimiser learning rate.
    """

    if lookback <= 0:
        raise ValueError(
            "lookback must be greater than zero."
        )

    if units_1 <= 0 or units_2 <= 0:
        raise ValueError(
            "LSTM layer sizes must be greater than zero."
        )

    if not 0 <= dropout < 1:
        raise ValueError(
            "dropout must be between zero and one."
        )

    model = Sequential(
        [
            Input(
                shape=(lookback, 1),
            ),
            LSTM(
                units_1,
                return_sequences=True,
            ),
            Dropout(
                dropout,
            ),
            LSTM(
                units_2,
                return_sequences=False,
            ),
            Dropout(
                dropout,
            ),
            Dense(
                16,
                activation="relu",
            ),
            Dense(
                1,
            ),
        ]
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate,
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def iterative_forecast(
    model,
    scaled_history: np.ndarray,
    horizon: int,
    lookback: int,
) -> np.ndarray:
    """
    Produce a recursive multi-step forecast from a one-step LSTM.

    Each prediction is appended to the model history and then used
    as an input when forecasting the following hour.

    This method can accumulate error over long forecast horizons.
    """

    if horizon <= 0:
        raise ValueError(
            "horizon must be greater than zero."
        )

    history = list(
        np.asarray(
            scaled_history,
            dtype=np.float32,
        ).reshape(-1)
    )

    if len(history) < lookback:
        raise ValueError(
            "History must contain at least lookback observations."
        )

    predictions = []

    for _ in range(horizon):
        model_input = np.asarray(
            history[-lookback:],
            dtype=np.float32,
        ).reshape(
            1,
            lookback,
            1,
        )

        prediction = float(
            model.predict(
                model_input,
                verbose=0,
            )[0, 0]
        )

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )

    return np.asarray(
        predictions,
        dtype=np.float32,
    )