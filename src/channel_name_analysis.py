#!/usr/bin/env python3
"""
Analisi completa dei nomi canale estratti dai CSV snowball.

Copre i punti richiesti:
1) Word count / term frequency
2) Centralita su grafo di co-occorrenza parole

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
import nltk
nltk.download("stopwords")
nltk.download('punkt_tab')

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
    # username_df = merged_df[["channel_id", "username", "source_file"]].copy()
    # username_df["name_kind"] = "username"
    # username_df = username_df.rename(columns={"username": "name"})

    title_df = merged_df[["channel_id", "title", "source_file"]].copy()
    title_df["name_kind"] = "title"
    title_df = title_df.rename(columns={"title": "name"})
    # dropna sui titoli vuoti
    title_df = title_df.dropna(subset=["name"])
    title_df = title_df[title_df["name"].str.strip() != ""].reset_index(drop=True)

    # names_df = pd.concat([username_df, title_df], ignore_index=True)
    # names_df["name"] = names_df["name"].fillna("").astype(str).str.strip()
    # names_df = names_df[names_df["name"] != ""].reset_index(drop=True)
    # return names_df
    return title_df


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
    # semantic_summary_df: pd.DataFrame,
    # tone_summary_df: pd.DataFrame,
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
        # "dominant_semantic_field_counts": {
        #     row["semantic_field"]: int(row["count"])
        #     for _, row in semantic_summary_df.iterrows()
        # },
        # "dominant_tone_counts": {
        #     row["tone"]: int(row["count"])
        #     for _, row in tone_summary_df.iterrows()
        # },
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


    stopwords_it = nltk.corpus.stopwords.words("italian")
    stopwords_en = nltk.corpus.stopwords.words("english")
    stopwords_ru = nltk.corpus.stopwords.words("russian")
    stopwords_de = nltk.corpus.stopwords.words("german")
    stopwords_es = nltk.corpus.stopwords.words("spanish")
    stopwords_fr = nltk.corpus.stopwords.words("french")
    all_stopwords = set(stopwords_it + stopwords_en + stopwords_ru + stopwords_de + stopwords_es + stopwords_fr )

    from sklearn.feature_extraction.text import CountVectorizer
    count_model = CountVectorizer(ngram_range=(1,3), stop_words=all_stopwords) # default unigram model
    docs = names_df["name"].tolist()
    X = count_model.fit_transform(docs)
    # X[X > 0] = 1 # run this line if you don't want extra within-text cooccurence (see below)
    Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
    Xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
    print(Xc.todense()) # print out matrix in dense format 

    # extract most frequent words
    word_freq = X.sum(axis=0).A1
    words = count_model.get_feature_names_out()
    word_freq_df = pd.DataFrame({"word": words, "frequency": word_freq})
    word_freq_df = word_freq_df.sort_values(by="frequency", ascending=False)
    logger.info("Most frequent words extracted")
    logger.info("Writing word frequency DataFrame")
    word_freq_df.to_csv(cfg.output_dir / "word_frequency.csv", index=False, encoding="utf-8")

    # create a graph from the co-occurrence matrix
    logger.info("Costruzione grafo co-occorrenza e centralita")
    import networkx as nx
    G = nx.from_scipy_sparse_matrix(Xc)
    mapping = {i: word for i, word in enumerate(words)}
    G = nx.relabel_nodes(G, mapping)
    # visualize the graph
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    nx.draw(G, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold")
    plt.show()
    # save plot 
    logger.info('Saving graph in gexf format')
    nx.write_gexf(G, cfg.output_dir / "word_cooccurrence_graph.gexf")
    logger.info('Graph saved successfully')

    logger.info("Calcolo centralita del grafo")
    centrality_df = compute_graph_centrality(G)
    logger.info("Writing graph centrality DataFrame")
    centrality_df.to_csv(cfg.output_dir / "word_graph_centrality.csv", index=False, encoding="utf-8")
    logger.info("Graph centrality DataFrame written successfully")


if __name__ == "__main__":
    main()
