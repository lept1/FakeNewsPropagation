#!/usr/bin/env python3
"""
Analisi completa dei nomi canale estratti dai CSV snowball.

Copre i punti richiesti:
1) Word count / term frequency
2) Lunghezza e struttura sintattica
3) Campi semantici dominanti
4) Sentiment e tonalita (euristico lessicale)
5) Distribuzione delle lunghezze
6) Legge di Zipf
7) N-grammi
8) Centralita su grafo di co-occorrenza parole

Esempio:
    python src/channel_name_analysis.py
"""

from __future__ import annotations

import glob
import importlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

try:
    from .config import (
        CONFIG_FILE,
        DATA_DIR,
        PROJECT_ROOT,
        SNOWBALL_CHANNELS_OUTPUT_DIR,
        load_project_config,
        validate_log_level,
    )
except ImportError:
    from config import (
        CONFIG_FILE,
        DATA_DIR,
        PROJECT_ROOT,
        SNOWBALL_CHANNELS_OUTPUT_DIR,
        load_project_config,
        validate_log_level,
    )


logger = logging.getLogger(__name__)


TOKEN_RE = re.compile(r"[A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ]+", flags=re.UNICODE)
ARTICLE_PREP_IT = {
    "il", "lo", "la", "i", "gli", "le",
    "un", "uno", "una",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra", "del", "della", "dello", "dei", "degli", "delle",
    "al", "allo", "alla", "ai", "agli", "alle",
    "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "nel", "nello", "nella", "nei", "negli", "nelle",
    "sul", "sullo", "sulla", "sui", "sugli", "sulle",
}


SEMANTIC_FIELDS = {
    "geografia": {
        "italia", "europe", "europa", "world", "mondo", "global", "russia", "mosca", "roma", "eu", "usa",
        "nord", "sud", "est", "ovest", "mediterraneo", "sahel",
    },
    "tempo_velocita": {
        "news", "live", "flash", "breaking", "today", "daily", "now", "instant", "tempo", "ora", "veloce", "rapido", "24",
    },
    "verita_informazione": {
        "verita", "truth", "facts", "fact", "news", "notizie", "info", "informazione", "report", "monitor", "osint", "media",
    },
    "economia": {
        "economia", "economy", "mercato", "market", "finanza", "finance", "business", "borsa", "trading", "capital", "capitale",
    },
}


TONE_LEXICON = {
    "autorevolezza": {
        "official", "ufficiale", "report", "monitor", "analysis", "analisi", "istituto", "centro", "osservatorio", "media", "news",
    },
    "urgenza": {
        "urgent", "allerta", "alert", "flash", "breaking", "now", "subito", "live", "emergency", "ultimo", "ultima",
    },
    "nazionalismo": {
        "italia", "italiano", "italiani", "patria", "nation", "nazione", "russia", "russo", "europe", "europa", "usa",
    },
    "innovazione": {
        "tech", "innovation", "innovazione", "digital", "ai", "data", "lab", "future", "futuro", "quantum", "startup",
    },
}


@dataclass(frozen=True)
class AnalysisConfig:
    input_glob: str
    output_dir: Path
    top_n: int
    min_token_len: int
    log_level: str


def configure_logging(log_level: str) -> None:
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _to_abs_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _safe_int(value: Any, default_value: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default_value
    return max(minimum, parsed)


def load_analysis_config(config_file: str = str(CONFIG_FILE)) -> AnalysisConfig:
    # Validate the main project configuration using the central config system.
    load_project_config(config_file)

    with Path(config_file).open("r", encoding="utf-8") as handle:
        raw_cfg = json.load(handle)

    if not isinstance(raw_cfg, dict):
        raise ValueError("Il file config deve contenere un oggetto JSON")

    snowball_cfg = raw_cfg.get("snowball", {})
    if not isinstance(snowball_cfg, dict):
        snowball_cfg = {}

    channel_cfg = raw_cfg.get("channel_name_analysis", {})
    if not isinstance(channel_cfg, dict):
        channel_cfg = {}

    channels_output_dir = snowball_cfg.get("channels_output_dir")
    if channels_output_dir:
        default_input_glob = str(_to_abs_path(str(channels_output_dir).strip()) / "*.csv")
    else:
        default_input_glob = str(SNOWBALL_CHANNELS_OUTPUT_DIR / "*.csv")

    input_glob = str(channel_cfg.get("channels_input_glob", default_input_glob)).strip() or default_input_glob

    output_dir_value = str(
        channel_cfg.get("output_dir", DATA_DIR / "channel_name_analysis")
    ).strip()
    if not output_dir_value:
        output_dir_value = str(DATA_DIR / "channel_name_analysis")

    output_dir = _to_abs_path(output_dir_value)
    top_n = _safe_int(channel_cfg.get("top_n", 40), default_value=40, minimum=1)
    min_token_len = _safe_int(channel_cfg.get("min_token_len", 2), default_value=2, minimum=1)
    log_level = str(channel_cfg.get("log_level", "INFO")).strip() or "INFO"
    effective_log_level = validate_log_level(
        log_level,
        section_name="channel_name_analysis.log_level",
    )

    return AnalysisConfig(
        input_glob=input_glob,
        output_dir=output_dir,
        top_n=top_n,
        min_token_len=min_token_len,
        log_level=effective_log_level,
    )


def tokenize(text: str, min_len: int) -> List[str]:
    raw_tokens = TOKEN_RE.findall(text.lower())
    return [tok for tok in raw_tokens if len(tok) >= min_len]


def iter_ngrams(tokens: Sequence[str], n: int) -> Iterable[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return (tuple(tokens[idx: idx + n]) for idx in range(len(tokens) - n + 1))


def load_merged_channels(input_glob: str) -> pd.DataFrame:
    pattern = input_glob
    if not Path(input_glob).is_absolute():
        pattern = str((PROJECT_ROOT / input_glob).resolve())
    logger.info("Ricerca file CSV con pattern: %s", pattern)
    csv_paths = [Path(path) for path in sorted(glob.glob(pattern))]

    if not csv_paths:
        raise FileNotFoundError(f"Nessun CSV trovato con glob: {input_glob}")
    logger.info("CSV trovati: %s", len(csv_paths))

    dataframes = []
    for csv_path in csv_paths:
        logger.debug("Lettura CSV: %s", csv_path)
        df = pd.read_csv(csv_path)
        missing = {"username", "title"} - set(df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"File {csv_path} non valido: colonne mancanti -> {missing_str}")
        df["source_file"] = str(csv_path)
        dataframes.append(df)

    merged = pd.concat(dataframes, ignore_index=True)
    return merged


def build_name_table(merged_df: pd.DataFrame) -> pd.DataFrame:
    username_df = merged_df[["channel_id", "username", "source_file"]].copy()
    username_df["name_kind"] = "username"
    username_df = username_df.rename(columns={"username": "name"})

    title_df = merged_df[["channel_id", "title", "source_file"]].copy()
    title_df["name_kind"] = "title"
    title_df = title_df.rename(columns={"title": "name"})

    names_df = pd.concat([username_df, title_df], ignore_index=True)
    names_df["name"] = names_df["name"].fillna("").astype(str).str.strip()
    names_df = names_df[names_df["name"] != ""].reset_index(drop=True)
    return names_df


def add_text_features(names_df: pd.DataFrame, min_token_len: int) -> pd.DataFrame:
    features_df = names_df.copy()
    features_df["tokens"] = features_df["name"].map(lambda value: tokenize(value, min_token_len))
    features_df["word_count"] = features_df["tokens"].map(len)
    features_df["char_count"] = features_df["name"].map(len)
    features_df["article_prep_count"] = features_df["tokens"].map(
        lambda toks: sum(1 for tok in toks if tok in ARTICLE_PREP_IT)
    )
    features_df["contains_article_prep"] = features_df["article_prep_count"] > 0
    return features_df


def compute_word_frequency(features_df: pd.DataFrame) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for tokens in features_df["tokens"]:
        counter.update(tokens)

    freq_df = pd.DataFrame(
        [{"token": token, "frequency": freq} for token, freq in counter.items()]
    )
    if freq_df.empty:
        return pd.DataFrame(columns=["token", "frequency", "relative_frequency"])

    total = int(freq_df["frequency"].sum())
    freq_df["relative_frequency"] = freq_df["frequency"] / total
    freq_df = freq_df.sort_values(by=["frequency", "token"], ascending=[False, True]).reset_index(drop=True)
    return freq_df


def compute_length_stats(features_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "names_total", "value": float(len(features_df))},
        {"metric": "avg_words_per_name", "value": float(features_df["word_count"].mean())},
        {"metric": "median_words_per_name", "value": float(features_df["word_count"].median())},
        {"metric": "avg_chars_per_name", "value": float(features_df["char_count"].mean())},
        {"metric": "median_chars_per_name", "value": float(features_df["char_count"].median())},
        {
            "metric": "pct_names_with_article_or_preposition",
            "value": float(features_df["contains_article_prep"].mean() * 100.0),
        },
    ]
    return pd.DataFrame(rows)


def compute_semantic_scores(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_name_rows = []
    field_counter: Counter[str] = Counter()

    for idx, row in features_df.iterrows():
        tokens = row["tokens"]
        scores = {
            field: sum(1 for tok in tokens if tok in lexicon)
            for field, lexicon in SEMANTIC_FIELDS.items()
        }

        dominant_field = "none"
        if any(scores.values()):
            dominant_field = max(scores, key=scores.get)
            field_counter[dominant_field] += 1

        per_name_rows.append(
            {
                "row_index": int(idx),
                "name": row["name"],
                "name_kind": row["name_kind"],
                **scores,
                "dominant_semantic_field": dominant_field,
            }
        )

    per_name_df = pd.DataFrame(per_name_rows)

    summary_df = pd.DataFrame(
        [{"semantic_field": field, "count": field_counter.get(field, 0)} for field in SEMANTIC_FIELDS]
    )
    if not summary_df.empty:
        total = max(1, int(summary_df["count"].sum()))
        summary_df["pct"] = summary_df["count"] / total * 100.0
        summary_df = summary_df.sort_values(by=["count", "semantic_field"], ascending=[False, True]).reset_index(drop=True)

    return per_name_df, summary_df


def compute_tone_scores(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_name_rows = []
    tone_counter: Counter[str] = Counter()

    for idx, row in features_df.iterrows():
        tokens = row["tokens"]
        scores = {
            tone: sum(1 for tok in tokens if tok in lexicon)
            for tone, lexicon in TONE_LEXICON.items()
        }

        dominant_tone = "none"
        if any(scores.values()):
            dominant_tone = max(scores, key=scores.get)
            tone_counter[dominant_tone] += 1

        per_name_rows.append(
            {
                "row_index": int(idx),
                "name": row["name"],
                "name_kind": row["name_kind"],
                **scores,
                "dominant_tone": dominant_tone,
            }
        )

    per_name_df = pd.DataFrame(per_name_rows)
    summary_df = pd.DataFrame(
        [{"tone": tone, "count": tone_counter.get(tone, 0)} for tone in TONE_LEXICON]
    )
    if not summary_df.empty:
        total = max(1, int(summary_df["count"].sum()))
        summary_df["pct"] = summary_df["count"] / total * 100.0
        summary_df = summary_df.sort_values(by=["count", "tone"], ascending=[False, True]).reset_index(drop=True)

    return per_name_df, summary_df


def compute_zipf(freq_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    zipf_df = freq_df[["token", "frequency"]].copy()
    if zipf_df.empty:
        return zipf_df, {"slope": 0.0, "intercept": 0.0, "zipf_exponent": 0.0, "r2": 0.0}

    zipf_df["rank"] = np.arange(1, len(zipf_df) + 1)
    zipf_df["log_rank"] = np.log(zipf_df["rank"])
    zipf_df["log_frequency"] = np.log(zipf_df["frequency"])

    slope, intercept = np.polyfit(zipf_df["log_rank"], zipf_df["log_frequency"], 1)
    predicted = slope * zipf_df["log_rank"] + intercept
    residual_sum = float(np.sum((zipf_df["log_frequency"] - predicted) ** 2))
    total_sum = float(np.sum((zipf_df["log_frequency"] - zipf_df["log_frequency"].mean()) ** 2))
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0

    metrics = {
        "slope": float(slope),
        "intercept": float(intercept),
        "zipf_exponent": float(-slope),
        "r2": float(r2),
    }
    return zipf_df, metrics


def compute_ngrams(features_df: pd.DataFrame, n: int) -> pd.DataFrame:
    counter: Counter[Tuple[str, ...]] = Counter()
    for tokens in features_df["tokens"]:
        counter.update(iter_ngrams(tokens, n))

    rows = []
    for gram, freq in counter.items():
        rows.append({"ngram": " ".join(gram), "frequency": freq})

    ngram_df = pd.DataFrame(rows)
    if ngram_df.empty:
        return pd.DataFrame(columns=["ngram", "frequency"])

    ngram_df = ngram_df.sort_values(by=["frequency", "ngram"], ascending=[False, True]).reset_index(drop=True)
    return ngram_df


def build_word_cooccurrence_graph(features_df: pd.DataFrame):
    graph = nx.Graph()

    for tokens in features_df["tokens"]:
        unique_tokens = sorted(set(tokens))
        for tok in unique_tokens:
            if tok not in graph:
                graph.add_node(tok)

        for a, b in combinations(unique_tokens, 2):
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += 1
            else:
                graph.add_edge(a, b, weight=1)

    return graph


def compute_graph_centrality(graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["token", "degree_centrality", "betweenness_centrality", "weighted_degree"])

    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight="weight")
    weighted_degree = dict(graph.degree(weight="weight"))

    rows = []
    for token in graph.nodes:
        rows.append(
            {
                "token": token,
                "degree_centrality": float(degree.get(token, 0.0)),
                "betweenness_centrality": float(betweenness.get(token, 0.0)),
                "weighted_degree": float(weighted_degree.get(token, 0.0)),
            }
        )

    centrality_df = pd.DataFrame(rows)
    centrality_df = centrality_df.sort_values(
        by=["betweenness_centrality", "degree_centrality", "weighted_degree", "token"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return centrality_df


def save_plots(
    output_dir: Path,
    freq_df: pd.DataFrame,
    features_df: pd.DataFrame,
    zipf_df: pd.DataFrame,
    zipf_metrics: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not freq_df.empty:
        top = freq_df.head(25)
        plt.figure(figsize=(14, 6))
        plt.bar(top["token"], top["frequency"])
        plt.title("Top parole nei nomi canale")
        plt.xlabel("Token")
        plt.ylabel("Frequenza")
        plt.xticks(rotation=65, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "top_tokens.png", dpi=180)
        plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(features_df["word_count"], bins=20)
    plt.title("Distribuzione lunghezze (numero parole per nome)")
    plt.xlabel("Parole per nome")
    plt.ylabel("Numero nomi")
    plt.tight_layout()
    plt.savefig(output_dir / "word_count_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(features_df["char_count"], bins=20)
    plt.title("Distribuzione lunghezze (numero caratteri per nome)")
    plt.xlabel("Caratteri per nome")
    plt.ylabel("Numero nomi")
    plt.tight_layout()
    plt.savefig(output_dir / "char_count_distribution.png", dpi=180)
    plt.close()

    if not zipf_df.empty:
        plt.figure(figsize=(8, 6))
        plt.scatter(zipf_df["log_rank"], zipf_df["log_frequency"], s=12, alpha=0.7)
        fit_y = zipf_metrics["slope"] * zipf_df["log_rank"] + zipf_metrics["intercept"]
        plt.plot(zipf_df["log_rank"], fit_y, color="red", linewidth=2)
        plt.title("Zipf: log(rank) vs log(frequenza)")
        plt.xlabel("log(rank)")
        plt.ylabel("log(frequency)")
        plt.tight_layout()
        plt.savefig(output_dir / "zipf_loglog.png", dpi=180)
        plt.close()


def write_summary(
    output_dir: Path,
    merged_df: pd.DataFrame,
    names_df: pd.DataFrame,
    length_stats_df: pd.DataFrame,
    zipf_metrics: Dict[str, float],
    semantic_summary_df: pd.DataFrame,
    tone_summary_df: pd.DataFrame,
) -> None:
    stats_map = {row["metric"]: row["value"] for _, row in length_stats_df.iterrows()}
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "merged_csv_rows": int(len(merged_df)),
        "names_analyzed": int(len(names_df)),
        "avg_words_per_name": float(stats_map.get("avg_words_per_name", 0.0)),
        "avg_chars_per_name": float(stats_map.get("avg_chars_per_name", 0.0)),
        "pct_names_with_article_or_preposition": float(
            stats_map.get("pct_names_with_article_or_preposition", 0.0)
        ),
        "zipf": zipf_metrics,
        "dominant_semantic_field_counts": {
            row["semantic_field"]: int(row["count"])
            for _, row in semantic_summary_df.iterrows()
        },
        "dominant_tone_counts": {
            row["tone"]: int(row["count"])
            for _, row in tone_summary_df.iterrows()
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = load_analysis_config()
    configure_logging(cfg.log_level)
    logger.info("Logger configurato con livello: %s", cfg.log_level)
    logger.info("Configurazione analisi: input_glob=%s", cfg.input_glob)
    logger.info("Configurazione analisi: output_dir=%s", cfg.output_dir)
    logger.info("Configurazione analisi: top_n=%s min_token_len=%s", cfg.top_n, cfg.min_token_len)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Avvio merge dataset canali")
    merged_df = load_merged_channels(cfg.input_glob)
    names_df = build_name_table(merged_df)
    features_df = add_text_features(names_df, cfg.min_token_len)
    logger.info("Nomi validi estratti (username + title): %s", len(names_df))

    logger.info("Calcolo metriche testuali e statistiche")
    freq_df = compute_word_frequency(features_df)
    length_stats_df = compute_length_stats(features_df)
    semantic_per_name_df, semantic_summary_df = compute_semantic_scores(features_df)
    tone_per_name_df, tone_summary_df = compute_tone_scores(features_df)
    zipf_df, zipf_metrics = compute_zipf(freq_df)
    bigrams_df = compute_ngrams(features_df, n=2)
    trigrams_df = compute_ngrams(features_df, n=3)

    logger.info("Costruzione grafo co-occorrenza e centralita")
    graph = build_word_cooccurrence_graph(features_df)
    centrality_df = compute_graph_centrality(graph)

    logger.info("Scrittura output CSV in corso")
    merged_df.to_csv(cfg.output_dir / "merged_snowball_channels.csv", index=False, encoding="utf-8")
    names_df.to_csv(cfg.output_dir / "name_rows.csv", index=False, encoding="utf-8")

    export_features_df = features_df.copy()
    export_features_df["tokens"] = export_features_df["tokens"].map(lambda toks: "|".join(toks))
    export_features_df.to_csv(cfg.output_dir / "name_features.csv", index=False, encoding="utf-8")

    freq_df.to_csv(cfg.output_dir / "word_frequency.csv", index=False, encoding="utf-8")
    length_stats_df.to_csv(cfg.output_dir / "length_stats.csv", index=False, encoding="utf-8")
    semantic_per_name_df.to_csv(cfg.output_dir / "semantic_scores_per_name.csv", index=False, encoding="utf-8")
    semantic_summary_df.to_csv(cfg.output_dir / "semantic_summary.csv", index=False, encoding="utf-8")
    tone_per_name_df.to_csv(cfg.output_dir / "tone_scores_per_name.csv", index=False, encoding="utf-8")
    tone_summary_df.to_csv(cfg.output_dir / "tone_summary.csv", index=False, encoding="utf-8")
    zipf_df.to_csv(cfg.output_dir / "zipf_table.csv", index=False, encoding="utf-8")
    bigrams_df.to_csv(cfg.output_dir / "bigrams_frequency.csv", index=False, encoding="utf-8")
    trigrams_df.to_csv(cfg.output_dir / "trigrams_frequency.csv", index=False, encoding="utf-8")

    nx.write_gexf(graph, cfg.output_dir / "word_cooccurrence_graph.gexf")
    centrality_df.to_csv(cfg.output_dir / "word_graph_centrality.csv", index=False, encoding="utf-8")

    logger.info("Generazione grafici e summary")
    save_plots(cfg.output_dir, freq_df.head(cfg.top_n), features_df, zipf_df, zipf_metrics)
    write_summary(
        cfg.output_dir,
        merged_df,
        names_df,
        length_stats_df,
        zipf_metrics,
        semantic_summary_df,
        tone_summary_df,
    )
    logger.info("Analisi completata con successo")

    print("Analisi completata.")
    print(f"Output directory: {cfg.output_dir}")
    print(f"CSV uniti: {len(merged_df)} righe")
    print(f"Nomi analizzati (username + title): {len(names_df)}")
    print(f"Nodi grafo parole: {graph.number_of_nodes()} | Archi: {graph.number_of_edges()}")
    print(
        "Zipf -> "
        f"esponente={zipf_metrics['zipf_exponent']:.4f}, "
        f"slope={zipf_metrics['slope']:.4f}, "
        f"R2={zipf_metrics['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
