# This script downloads front-month crude oil futures data and trains an XGBoost binary classifier
# to predict next-day direction. It uses RandomizedSearchCV with TimeSeriesSplit cross validation.

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier


# --- Fetch daily crude oil futures data from Yahoo Finance ---
def fetch_crude_oil(start="2010-01-01", end=None):
    ticker = "CL=F"  # front month crude oil futures symbol on Yahoo Finance
    df = yf.download(ticker, start=start, end=end)
    df = df.dropna()
    return df


# --- Feature engineering ---
def make_features(df):
    df = df.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["rolling_std_5"] = df["Close"].rolling(5).std()
    df["rolling_std_10"] = df["Close"].rolling(10).std()
    df.dropna(inplace=True)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    features = df.drop(columns=["target"])
    target = df["target"]
    return features, target


# --- Train model with RandomizedSearchCV and TimeSeriesSplit ---
def train_xgb_classifier(X, y):
    param_dist = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "n_estimators": [100, 200, 300],
    }
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,
        cv=tscv,
        scoring="roc_auc",
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


if __name__ == "__main__":
    df = fetch_crude_oil()
    X, y = make_features(df)
    search = train_xgb_classifier(X, y)
    print("Best parameters:\n", search.best_params_)
    preds = search.predict(X)
    print("Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds))
