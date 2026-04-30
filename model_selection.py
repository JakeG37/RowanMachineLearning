import json
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


OUTPUT_DIR = Path("model_selection_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "volume_change",
    "close_to_open",
    "high_low_spread",
    "return_streak",
    "ma_ratio_5",
    "ma_ratio_10",
    "volatility_5",
    "volatility_10",
    "rsi_14",
]

TARGET_COLUMN = "target_next_day_up"

SECTOR_TICKERS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AMD", "ORCL", "CRM", "ADBE"],
    "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH", "TMO"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "BKNG"],
    "Communication Services": ["GOOGL", "META", "NFLX", "TMUS", "DIS", "CMCSA", "T"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
    "Industrials": ["CAT", "GE", "BA", "HON", "UPS", "UNP", "DE"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MDLZ"],
}


def sanitize_name(value):
    return value.lower().replace(" ", "_").replace("/", "_")


def normalize_tickers(tickers_text):
    return [ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip()]


def add_technical_indicators(price_df):
    df = price_df.copy()

    df["daily_return"] = df["Close"].pct_change()
    df["volume_change"] = df["Volume"].pct_change()
    df["close_to_open"] = (df["Close"] - df["Open"]) / df["Open"]
    df["high_low_spread"] = (df["High"] - df["Low"]) / df["Open"]
    direction = pd.Series(0, index=df.index, dtype="float64")
    direction.loc[df["daily_return"] > 0] = 1.0
    direction.loc[df["daily_return"] < 0] = -1.0

    streak_change = direction.ne(direction.shift(fill_value=0))
    streak_groups = streak_change.cumsum()
    streak_lengths = direction.groupby(streak_groups).cumcount() + 1
    df["return_streak"] = streak_lengths.where(direction != 0, 0).astype(float) * direction
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()
    df["ma_ratio_5"] = df["Close"] / df["ma_5"]
    df["ma_ratio_10"] = df["Close"] / df["ma_10"]
    df["volatility_5"] = df["daily_return"].rolling(window=5).std()
    df["volatility_10"] = df["daily_return"].rolling(window=10).std()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["next_open"] = df["Open"].shift(-1)
    df[TARGET_COLUMN] = (df["next_open"] > df["Close"]).astype(int)

    return df


def flatten_history_columns(history, ticker):
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = [
            column[0] if isinstance(column, tuple) and column[1] == ticker else column[0]
            for column in history.columns
        ]
    return history


def download_ticker_history(ticker, years=5):
    history = yf.download(
        ticker,
        period=f"{years}y",
        progress=False,
        auto_adjust=False,
    )

    if history.empty:
        return pd.DataFrame()

    history = history.reset_index()
    history = flatten_history_columns(history, ticker)
    history["Date"] = pd.to_datetime(history["Date"])
    history["date"] = history["Date"].dt.normalize()
    history["ticker"] = ticker

    history = add_technical_indicators(history)
    history = history.dropna(subset=FEATURE_COLUMNS + ["next_open"]).copy()
    return history


def build_sector_dataset(sector_name, tickers, years=5):
    frames = []

    for ticker in tickers:
        ticker_df = download_ticker_history(ticker, years=years)
        if ticker_df.empty:
            print(f"Skipping {ticker}: no usable price history returned.")
            continue

        ticker_df["sector"] = sector_name
        frames.append(ticker_df)
        print(f"Added {ticker}: {len(ticker_df)} rows")

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)
    return dataset


def build_model_candidates():
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=4,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_leaf=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "adaboost": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", AdaBoostClassifier(random_state=42)),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15)),
            ]
        ),
        "svc_rbf": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
        "gaussian_nb": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GaussianNB()),
            ]
        ),
        "lda": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearDiscriminantAnalysis()),
            ]
        ),
    }


def score_predictions(y_true, predictions, probabilities):
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "predicted_up_rate": float(pd.Series(predictions).mean()),
    }

    if probabilities is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, probabilities)
    else:
        metrics["roc_auc"] = None

    return metrics


def evaluate_model_with_time_series_cv(model_name, pipeline, x, y, n_splits=5):
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold_number, (train_index, test_index) in enumerate(splitter.split(x), start=1):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(x_test)[:, 1]
        else:
            probabilities = None

        metrics = score_predictions(y_test, predictions, probabilities)
        metrics["fold"] = fold_number
        fold_metrics.append(metrics)

    fold_df = pd.DataFrame(fold_metrics)
    summary = {
        "model_name": model_name,
        "accuracy_mean": fold_df["accuracy"].mean(),
        "accuracy_std": fold_df["accuracy"].std(),
        "precision_mean": fold_df["precision"].mean(),
        "recall_mean": fold_df["recall"].mean(),
        "f1_mean": fold_df["f1"].mean(),
        "f1_std": fold_df["f1"].std(),
        "roc_auc_mean": fold_df["roc_auc"].dropna().mean() if fold_df["roc_auc"].notna().any() else None,
        "predicted_up_rate_mean": fold_df["predicted_up_rate"].mean(),
    }

    return summary, fold_df


def fit_best_model(dataset, model_name):
    candidates = build_model_candidates()
    pipeline = candidates[model_name]
    pipeline.fit(dataset[FEATURE_COLUMNS], dataset[TARGET_COLUMN])
    return pipeline


def save_outputs(sector_name, tickers, dataset, summary_df, fold_df, best_model_name, best_model):
    sector_slug = sanitize_name(sector_name)

    summary_path = OUTPUT_DIR / f"{sector_slug}_summary.csv"
    folds_path = OUTPUT_DIR / f"{sector_slug}_folds.csv"
    metadata_path = OUTPUT_DIR / f"{sector_slug}_best_model.json"
    model_path = OUTPUT_DIR / f"{sector_slug}_best_model.joblib"

    summary_df.to_csv(summary_path, index=False)
    fold_df.to_csv(folds_path, index=False)
    joblib.dump(best_model, model_path)

    metadata = {
        "sector": sector_name,
        "tickers": tickers,
        "rows": int(len(dataset)),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "best_model_name": best_model_name,
        "model_path": str(model_path),
        "summary_path": str(summary_path),
        "folds_path": str(folds_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return summary_path, folds_path, metadata_path, model_path


def select_models_for_sector(sector_name, tickers, years=5):
    dataset = build_sector_dataset(sector_name, tickers, years=years)
    if dataset.empty:
        raise ValueError(f"No data was available for sector {sector_name}.")

    x = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    summary_rows = []
    fold_frames = []

    for model_name, pipeline in build_model_candidates().items():
        print(f"Evaluating {model_name}...")
        summary, fold_df = evaluate_model_with_time_series_cv(model_name, pipeline, x, y)
        summary_rows.append(summary)

        fold_df["model_name"] = model_name
        fold_frames.append(fold_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["f1_mean", "accuracy_mean", "roc_auc_mean"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    fold_df = pd.concat(fold_frames, ignore_index=True)
    best_model_name = summary_df.iloc[0]["model_name"]
    best_model = fit_best_model(dataset, best_model_name)

    paths = save_outputs(
        sector_name=sector_name,
        tickers=tickers,
        dataset=dataset,
        summary_df=summary_df,
        fold_df=fold_df,
        best_model_name=best_model_name,
        best_model=best_model,
    )

    return {
        "dataset_rows": len(dataset),
        "summary_df": summary_df,
        "best_model_name": best_model_name,
        "summary_path": paths[0],
        "folds_path": paths[1],
        "metadata_path": paths[2],
        "model_path": paths[3],
    }


def main():
    selected_sector = input(
        "Enter a sector to evaluate, or press Enter to list available sectors: "
    ).strip()

    if not selected_sector:
        print("\nAvailable sectors:")
        for sector_name in SECTOR_TICKERS:
            print(f"- {sector_name}")
        return

    default_tickers = SECTOR_TICKERS.get(selected_sector)
    if not default_tickers:
        raise ValueError("Sector not found in SECTOR_TICKERS. Add it to the dictionary first.")

    custom_tickers_text = input(
        "Optional custom ticker list, comma-separated. Press Enter to use defaults: "
    ).strip()
    tickers = normalize_tickers(custom_tickers_text) if custom_tickers_text else default_tickers

    print(f"\nRunning model selection for {selected_sector}")
    print(f"Tickers: {', '.join(tickers)}")

    result = select_models_for_sector(selected_sector, tickers, years=5)

    print(f"\nDataset rows: {result['dataset_rows']}")
    print("\nModel ranking:")
    print(result["summary_df"].to_string(index=False))
    print(f"\nBest model: {result['best_model_name']}")
    print(f"Saved summary to: {result['summary_path']}")
    print(f"Saved fold metrics to: {result['folds_path']}")
    print(f"Saved best model metadata to: {result['metadata_path']}")
    print(f"Saved best fitted model to: {result['model_path']}")


if __name__ == "__main__":
    main()
