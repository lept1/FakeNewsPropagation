#!/usr/bin/env python3
"""
Genera un grafo di diffusione da dati Telegram raccolti in CSV.

Dipendenze:
    pip install pandas networkx matplotlib

Esempio:
    python telegram_fake_news_graph.py \
        --input-csv telegram_fakenews_analysis.csv \
        --output-prefix telegram_diffusion_graph

Output generati:
- <output-prefix>.gexf  (grafo per Gephi/Cytoscape)
- <output-prefix>.png   (visualizzazione rapida)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import networkx as nx
import pandas as pd


REQUIRED_COLUMNS = {
    "message_id",
    "channel_username",
    "views",
    "forwards",
    "is_forwarded",
    "forward_from_chat",
}

SENTIMENT_LABEL_MAP = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Costruisce il grafo di diffusione dai dati della datacollection Telegram"
    )
    parser.add_argument(
        "--input-csv",
        default="telegram_fakenews_analysis.csv",
        help="Path al CSV prodotto dallo script di datacollection",
    )
    parser.add_argument(
        "--output-prefix",
        default="telegram_diffusion_graph",
        help="Prefisso dei file output (.gexf e .png)",
    )
    parser.add_argument(
        "--min-weight",
        type=int,
        default=1,
        help="Soglia minima del peso arco per essere visualizzato/salvato",
    )
    parser.add_argument(
        "--include-unknown-sources",
        action="store_true",
        help="Include nodi forward con sorgente non identificata",
    )
    return parser.parse_args()


def load_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"CSV non valido: colonne mancanti -> {missing_str}")

    return df


def is_unknown_source(source: str) -> bool:
    source_l = source.strip().lower()
    return source_l in {"", "none", "nan", "unknown_forward_source"}


def extract_row_sentiment_score(row: pd.Series) -> float | None:
    if "sentiment_score" in row.index and pd.notna(row["sentiment_score"]):
        try:
            return float(row["sentiment_score"])
        except (TypeError, ValueError):
            pass

    if "sentiment_label" in row.index and pd.notna(row["sentiment_label"]):
        label = str(row["sentiment_label"]).strip().lower()
        if label in SENTIMENT_LABEL_MAP:
            return SENTIMENT_LABEL_MAP[label]

    return None


def build_diffusion_graph(
    df: pd.DataFrame,
    min_weight: int,
    include_unknown_sources: bool,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    forwarded_df = df[df["is_forwarded"].astype(bool)].copy()
    if forwarded_df.empty:
        return graph

    edge_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    has_sentiment = False

    for _, row in forwarded_df.iterrows():
        source = str(row["forward_from_chat"]).strip()
        target = str(row["channel_username"]).strip()

        if not source or not target:
            continue
        if is_unknown_source(source) and not include_unknown_sources:
            continue

        edge_key = (source, target)
        if edge_key not in edge_stats:
            edge_stats[edge_key] = {
                "messages": 0,
                "sum_views": 0,
                "sum_forwards": 0,
                "sum_sentiment": 0.0,
                "sentiment_count": 0,
            }

        edge_stats[edge_key]["messages"] += 1
        edge_stats[edge_key]["sum_views"] += int(row.get("views", 0) or 0)
        edge_stats[edge_key]["sum_forwards"] += int(row.get("forwards", 0) or 0)

        sentiment_score = extract_row_sentiment_score(row)
        if sentiment_score is not None:
            edge_stats[edge_key]["sum_sentiment"] += sentiment_score
            edge_stats[edge_key]["sentiment_count"] += 1
            has_sentiment = True

    for (source, target), stats in edge_stats.items():
        if stats["messages"] < min_weight:
            continue

        graph.add_node(source, node_type="source")
        graph.add_node(target, node_type="channel")

        sentiment_count = int(stats["sentiment_count"])
        avg_sentiment = (
            float(stats["sum_sentiment"]) / sentiment_count if sentiment_count > 0 else 0.0
        )

        graph.add_edge(
            source,
            target,
            weight=int(stats["messages"]),
            views=int(stats["sum_views"]),
            forwards=int(stats["sum_forwards"]),
            sentiment_count=sentiment_count,
            avg_sentiment=avg_sentiment,
        )

    graph.graph["has_sentiment"] = has_sentiment

    return graph


def save_graph(graph: nx.DiGraph, output_prefix: str) -> Tuple[Path, Path]:
    gexf_path = Path(f"{output_prefix}.gexf")
    png_path = Path(f"{output_prefix}.png")

    nx.write_gexf(graph, gexf_path)

    plt.figure(figsize=(14, 10))

    if graph.number_of_nodes() == 0:
        plt.title("Grafo vuoto: nessun forward trovato")
        plt.axis("off")
    else:
        pos = nx.spring_layout(graph, seed=42, k=1.2)
        edge_weights = [max(1.0, float(data.get("weight", 1))) for _, _, data in graph.edges(data=True)]
        has_sentiment = bool(graph.graph.get("has_sentiment", False))

        if has_sentiment:
            norm = colors.Normalize(vmin=-1.0, vmax=1.0)
            edge_colors = [
                cm.RdYlGn(norm(float(data.get("avg_sentiment", 0.0))))
                for _, _, data in graph.edges(data=True)
            ]
        else:
            edge_colors = "#5a5a5a"

        node_colors = [
            "#1f77b4" if graph.nodes[n].get("node_type") == "channel" else "#ff7f0e"
            for n in graph.nodes
        ]

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=900,
            node_color=node_colors,
            alpha=0.9,
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            width=edge_weights,
            alpha=0.55,
            arrows=True,
            arrowsize=14,
            edge_color=edge_colors,
        )
        nx.draw_networkx_labels(graph, pos, font_size=8)

        plt.title("Rete di diffusione Telegram (forward source -> canale)")
        plt.axis("off")

        if has_sentiment:
            sm = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.03, pad=0.02)
            cbar.set_label("Sentiment medio arco (-1 negativo, +1 positivo)")

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return gexf_path, png_path


def print_graph_summary(graph: nx.DiGraph) -> None:
    print("\n===== RIEPILOGO GRAFO =====")
    print(f"Nodi: {graph.number_of_nodes()}")
    print(f"Archi: {graph.number_of_edges()}")

    if graph.number_of_edges() == 0:
        print("Nessun arco presente: verifica i forward nel CSV input.")
        return

    top_edges = sorted(
        graph.edges(data=True),
        key=lambda x: int(x[2].get("weight", 0)),
        reverse=True,
    )[:5]

    has_sentiment = bool(graph.graph.get("has_sentiment", False))
    if has_sentiment:
        weighted_sentiment_sum = 0.0
        weighted_sentiment_count = 0
        for _, _, data in graph.edges(data=True):
            count = int(data.get("sentiment_count", 0))
            weighted_sentiment_sum += float(data.get("avg_sentiment", 0.0)) * count
            weighted_sentiment_count += count

        if weighted_sentiment_count > 0:
            overall_avg_sentiment = weighted_sentiment_sum / weighted_sentiment_count
            print(f"Sentiment medio globale (solo forward con sentiment): {overall_avg_sentiment:.3f}")

    print("Top 5 archi per numero di messaggi inoltrati:")
    for source, target, data in top_edges:
        base = (
            f"  - {source} -> {target}: "
            f"messaggi={data.get('weight', 0)}, "
            f"views={data.get('views', 0)}, "
            f"forwards={data.get('forwards', 0)}"
        )
        if has_sentiment and int(data.get("sentiment_count", 0)) > 0:
            base += f", avg_sentiment={float(data.get('avg_sentiment', 0.0)):.3f}"
        print(base)


def main() -> None:
    args = parse_args()

    df = load_dataset(args.input_csv)
    graph = build_diffusion_graph(
        df=df,
        min_weight=max(1, args.min_weight),
        include_unknown_sources=args.include_unknown_sources,
    )

    gexf_path, png_path = save_graph(graph, args.output_prefix)
    print_graph_summary(graph)
    print(f"\nGrafo GEXF salvato in: {gexf_path}")
    print(f"Immagine PNG salvata in: {png_path}")


if __name__ == "__main__":
    main()
