import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier


def load_data(ticker: str = "CL=F", period: str = "10y") -> pd.DataFrame:
    """Download historical data for the specified ticker.

    yfinance returns a MultiIndex column structure when a single ticker is
    requested. This helper flattens the columns for easier downstream use.
    """
    data = yf.download(ticker, period=period, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.dropna(inplace=True)
    return data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    for window in [5, 10, 20]:
        df[f"ma_{window}"] = df["Close"].rolling(window).mean()
        df[f"std_{window}"] = df["Close"].rolling(window).std()
    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = df["Close"].shift(lag)
    df.dropna(inplace=True)
    return df


def create_labels(df: pd.DataFrame) -> pd.Series:
    return (df["Close"].shift(-1) > df["Close"]).astype(int)


def prepare_dataset(ticker: str = "CL=F", period: str = "10y"):
    df = load_data(ticker, period)
    df = add_features(df)
    labels = create_labels(df)
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])
    df.drop(df.tail(1).index, inplace=True)  # drop last row without label
    labels = labels.loc[df.index]
    return df, labels


def train_model(X: pd.DataFrame, y: pd.Series):
    param_dist = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


if __name__ == "__main__":
    X, y = prepare_dataset()
    clf = train_model(X, y)
    print("Best parameters:", clf.best_params_)
    preds = clf.predict(X)
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds))
