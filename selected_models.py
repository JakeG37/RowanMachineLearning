import json
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

from model_selection import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_technical_indicators,
    build_model_candidates,
    build_sector_dataset,
    flatten_history_columns,
    sanitize_name,
)


DEPLOY_DIR = Path("selected_model_outputs")
DEPLOY_DIR.mkdir(exist_ok=True)

# This is the deployment registry for the project.
# Add more sectors here as you finalize model choices.
SELECTED_MODELS_BY_SECTOR = {
    "Technology": "knn",
}

SECTOR_TICKERS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AMD", "ORCL", "CRM", "ADBE"],
}


def train_selected_model_for_sector(sector_name, model_name, tickers, years=5):
    candidates = build_model_candidates()
    if model_name not in candidates:
        raise ValueError(f"Model '{model_name}' is not defined in model_selection.py.")

    dataset = build_sector_dataset(sector_name, tickers, years=years)
    if dataset.empty:
        raise ValueError(f"No data was available to train the selected model for {sector_name}.")

    pipeline = candidates[model_name]
    pipeline.fit(dataset[FEATURE_COLUMNS], dataset[TARGET_COLUMN])

    sector_slug = sanitize_name(sector_name)
    model_path = DEPLOY_DIR / f"{sector_slug}_{model_name}.joblib"
    metadata_path = DEPLOY_DIR / f"{sector_slug}_{model_name}.json"

    joblib.dump(pipeline, model_path)

    metadata = {
        "sector": sector_name,
        "selected_model_name": model_name,
        "tickers": tickers,
        "years": years,
        "rows": int(len(dataset)),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "model_path": str(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return model_path, metadata_path, metadata


def load_selected_model_for_sector(sector_name):
    model_name = SELECTED_MODELS_BY_SECTOR.get(sector_name)
    if not model_name:
        raise ValueError(f"No selected model configured for sector {sector_name}.")

    sector_slug = sanitize_name(sector_name)
    model_path = DEPLOY_DIR / f"{sector_slug}_{model_name}.joblib"
    metadata_path = DEPLOY_DIR / f"{sector_slug}_{model_name}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected model file not found for {sector_name}. Run selected_models.py first."
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return model, metadata


def build_prediction_feature_row(ticker, years=2):
    history = yf.download(
        ticker,
        period=f"{years}y",
        progress=False,
        auto_adjust=False,
    )

    if history.empty:
        raise ValueError(f"No recent market history was returned for ticker {ticker}.")

    history = history.reset_index()
    history = flatten_history_columns(history, ticker)
    history["Date"] = pd.to_datetime(history["Date"])
    history["date"] = history["Date"].dt.normalize()
    history["ticker"] = ticker
    history = add_technical_indicators(history)
    history = history.dropna(subset=FEATURE_COLUMNS).copy()

    if history.empty:
        raise ValueError(f"Not enough price history to compute features for ticker {ticker}.")

    latest_row = history.sort_values("date").iloc[-1]
    feature_frame = pd.DataFrame([latest_row[FEATURE_COLUMNS]])
    return feature_frame, latest_row


def predict_with_selected_model(ticker, sector_name):
    model, metadata = load_selected_model_for_sector(sector_name)
    feature_frame, latest_row = build_prediction_feature_row(ticker)

    prediction = int(model.predict(feature_frame)[0])
    probability_up = None
    if hasattr(model, "predict_proba"):
        probability_up = float(model.predict_proba(feature_frame)[0][1])

    return {
        "prediction": prediction,
        "prediction_label": "UP" if prediction == 1 else "DOWN",
        "probability_up": probability_up,
        "latest_close": float(latest_row["Close"]),
        "latest_date": latest_row["date"],
        "model_metadata": metadata,
        "features": latest_row[FEATURE_COLUMNS].to_dict(),
    }


def main():
    for sector_name, model_name in SELECTED_MODELS_BY_SECTOR.items():
        tickers = SECTOR_TICKERS.get(sector_name)
        if not tickers:
            raise ValueError(f"No ticker list configured for sector {sector_name}.")

        print(f"\nTraining selected model for {sector_name}")
        print(f"Chosen model: {model_name}")
        print(f"Tickers: {', '.join(tickers)}")

        model_path, metadata_path, metadata = train_selected_model_for_sector(
            sector_name=sector_name,
            model_name=model_name,
            tickers=tickers,
            years=5,
        )

        print(f"Rows used: {metadata['rows']}")
        print(f"Saved selected model to: {model_path}")
        print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
