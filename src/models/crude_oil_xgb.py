import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score


def download_data(start="2010-01-01", end=None):
    """Download front month Crude Oil futures prices from Yahoo Finance."""
    ticker = "CL=F"
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    return df


def prepare_features(df):
    """Create features and binary target based on next day price movement."""
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["hl_range"] = df["High"] - df["Low"]
    df["oc_diff"] = df["Close"] - df["Open"]
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    features = ["return", "hl_range", "oc_diff", "ma_5", "ma_10"]
    X = df[features]
    y = df["target"]
    return X, y


def train_model(X, y):
    """Train XGBClassifier using TimeSeriesSplit and RandomizedSearchCV."""
    tscv = TimeSeriesSplit(n_splits=5)
    clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss")
    param_grid = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "n_estimators": [100, 200, 300, 400]
    }
    search = RandomizedSearchCV(
        clf,
        param_grid,
        n_iter=20,
        scoring="accuracy",
        cv=tscv,
        verbose=0,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def split_train_test(X, y, test_size=0.2):
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    return X_train, X_test, y_train, y_test


def main():
    df = download_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model, params, cv_score = train_model(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Best Params:", params)
    print(f"CV Best Score: {cv_score:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
