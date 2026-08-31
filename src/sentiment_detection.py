#!/usr/bin/env python3
"""
Script intermedio: aggiunge il sentiment ai messaggi Telegram raccolti.

Dipendenze:
    pip install pandas transformers torch

Esempio:
    python telegram_fake_news_sentiment.py \
        --input-csv telegram_fakenews_analysis.csv \
        --output-csv telegram_fakenews_analysis_with_sentiment.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from transformers import pipeline

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esegue sentiment analysis sui messaggi Telegram e aggiorna il dataframe"
    )
    parser.add_argument(
        "--input-csv",
        default="telegram_fakenews_analysis.csv",
        help="CSV in input prodotto dallo script di datacollection",
    )
    parser.add_argument(
        "--output-csv",
        default="telegram_fakenews_analysis_with_sentiment.csv",
        help="CSV in output con colonne sentiment aggiuntive",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Nome colonna contenente il testo del messaggio",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Modello Hugging Face per sentiment analysis",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per inferenza",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device per transformers pipeline (-1=CPU, 0+ GPU)",
    )
    return parser.parse_args()


def load_dataframe(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV input non trovato: {csv_path}")
    return pd.read_csv(path)


def normalize_label(raw_label: str) -> str:
    label = raw_label.strip().lower()

    if "neg" in label or label.endswith("0"):
        return "negative"
    if "neu" in label or label.endswith("1"):
        return "neutral"
    if "pos" in label or label.endswith("2"):
        return "positive"

    return "neutral"


def label_to_score(label: str) -> int:
    if label == "positive":
        return 1
    if label == "negative":
        return -1
    return 0


def infer_sentiment(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: int,
) -> pd.DataFrame:
    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        truncation=True,
        max_length=512,
    )

    sentiment_labels: List[str] = []
    sentiment_scores: List[int] = []
    sentiment_confidences: List[float] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        predictions = classifier(batch)

        for prediction in predictions:
            label = normalize_label(str(prediction["label"]))
            confidence = float(prediction["score"])

            sentiment_labels.append(label)
            sentiment_scores.append(label_to_score(label))
            sentiment_confidences.append(confidence)

    return pd.DataFrame(
        {
            "sentiment_label": sentiment_labels,
            "sentiment_score": sentiment_scores,
            "sentiment_confidence": sentiment_confidences,
        }
    )


def enrich_with_sentiment(
    df: pd.DataFrame,
    text_column: str,
    model_name: str,
    batch_size: int,
    device: int,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Colonna testo non trovata nel CSV: {text_column}")

    enriched_df = df.copy()
    texts = enriched_df[text_column].fillna("").astype(str).tolist()

    if len(texts) == 0:
        enriched_df["sentiment_label"] = []
        enriched_df["sentiment_score"] = []
        enriched_df["sentiment_confidence"] = []
        return enriched_df

    sentiment_df = infer_sentiment(
        texts=texts,
        model_name=model_name,
        batch_size=max(1, batch_size),
        device=device,
    )

    enriched_df["sentiment_label"] = sentiment_df["sentiment_label"]
    enriched_df["sentiment_score"] = sentiment_df["sentiment_score"]
    enriched_df["sentiment_confidence"] = sentiment_df["sentiment_confidence"]

    return enriched_df


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Dataframe vuoto: nessun record analizzato.")
        return

    counts = df["sentiment_label"].value_counts(dropna=False).to_dict()
    print("\n===== RIEPILOGO SENTIMENT =====")
    print(f"Messaggi analizzati: {len(df)}")
    print("Distribuzione sentiment:")
    for key in ("positive", "neutral", "negative"):
        print(f"  - {key}: {int(counts.get(key, 0))}")


def main() -> None:
    args = parse_args()
    input_df = load_dataframe(args.input_csv)

    output_df = enrich_with_sentiment(
        df=input_df,
        text_column=args.text_column,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )

    output_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"CSV arricchito salvato in: {args.output_csv}")
    print_summary(output_df)


if __name__ == "__main__":
    main()
