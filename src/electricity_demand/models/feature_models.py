import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Fit a Random Forest regression model.

    Parameters
    ----------
    X_train:
        Training feature matrix.

    y_train:
        Training target series.

    random_state:
        Seed used for reproducibility.

    Returns
    -------
    RandomForestRegressor
        Fitted Random Forest model.
    """

    if X_train.empty:
        raise ValueError(
            "Random Forest training features are empty."
        )

    if y_train.empty:
        raise ValueError(
            "Random Forest training target is empty."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training features and target must have the same length."
        )

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def fit_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    """
    Fit a Histogram Gradient Boosting regression model.
    """

    if X_train.empty:
        raise ValueError(
            "Gradient Boosting training features are empty."
        )

    if y_train.empty:
        raise ValueError(
            "Gradient Boosting training target is empty."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training features and target must have the same length."
        )

    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def predict_feature_model(
    model,
    X_test: pd.DataFrame,
    index: pd.Index,
    name: str,
) -> pd.Series:
    """
    Generate predictions from a fitted feature-based model.
    """

    if X_test.empty:
        raise ValueError(
            "Test feature matrix is empty."
        )

    predictions = model.predict(
        X_test
    )

    return pd.Series(
        predictions,
        index=index,
        name=name,
    )