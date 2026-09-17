#!/usr/bin/env python3
"""
Costruisce una matrice di relazione e un grafo dai forward del snowball sampling.

Dipendenze:
    pip install pandas networkx matplotlib

Esempio:
    python src/graph_analysis.py

Output generati:
- matrice relazioni CSV (source -> target)
- grafo .gexf (Gephi/Cytoscape)
- grafo .png (visualizzazione rapida)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

try:
    from .config import DEFAULT_CONFIG_FILE, GraphAnalysisConfig, load_project_config
except ImportError:
    from config import DEFAULT_CONFIG_FILE, GraphAnalysisConfig, load_project_config


REQUIRED_COLUMNS = {
    "channel_to_id",
    "channel_to_username",
    "channel_to_title",
    "channel_from_id",
    "channel_from_username",
    "channel_from_title",
    "from_name_fallback",
    "target_message_id",
}


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


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"", "none", "nan"}:
        return ""
    return text


def resolve_source_label(row: pd.Series) -> str:
    source_username = normalize_cell(row.get("channel_from_username", ""))
    source_title = normalize_cell(row.get("channel_from_title", ""))
    source_fallback = normalize_cell(row.get("from_name_fallback", ""))
    source_id = normalize_cell(row.get("channel_from_id", ""))

    if source_username:
        return source_username
    if source_title:
        return source_title
    if source_fallback:
        return source_fallback
    if source_id:
        return f"channel_id:{source_id}"

    return ""


def resolve_target_label(row: pd.Series) -> str:
    target_username = normalize_cell(row.get("channel_to_username", ""))
    target_title = normalize_cell(row.get("channel_to_title", ""))
    target_id = normalize_cell(row.get("channel_to_id", ""))

    if target_username:
        return target_username
    if target_title:
        return target_title
    if target_id:
        return f"channel_id:{target_id}"

    return ""


def is_unresolved_source(row: pd.Series) -> bool:
    return not normalize_cell(row.get("channel_from_id", ""))


def build_edges_dataframe(
    df: pd.DataFrame,
    include_unresolved_sources: bool,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        if is_unresolved_source(row) and not include_unresolved_sources:
            continue

        source = resolve_source_label(row)
        target = resolve_target_label(row)
        if not source or not target:
            continue

        rows.append({"source": source, "target": target, "weight": 1})

    if not rows:
        return pd.DataFrame(columns=["source", "target", "weight"])

    edges_df = pd.DataFrame(rows)
    edges_df = (
        edges_df.groupby(["source", "target"], as_index=False)["weight"]
        .sum()
        .sort_values(by=["weight", "source", "target"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return edges_df


def build_relation_matrix(edges_df: pd.DataFrame) -> pd.DataFrame:
    if edges_df.empty:
        return pd.DataFrame()

    matrix = pd.pivot_table(
        edges_df,
        index="source",
        columns="target",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )
    matrix = matrix.sort_index().reindex(sorted(matrix.columns), axis=1)
    return matrix


def save_relation_matrix(matrix_df: pd.DataFrame, output_csv: str) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix_df.to_csv(output_path, encoding="utf-8")
    return output_path


def build_diffusion_graph(
    edges_df: pd.DataFrame,
    min_weight: int,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    if edges_df.empty:
        return graph

    for _, row in edges_df.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        weight = int(row["weight"])

        if weight < min_weight:
            continue

        graph.add_node(source, node_type="source")
        graph.add_node(target, node_type="channel")

        graph.add_edge(
            source,
            target,
            weight=weight,
        )

    return graph


def save_graph(graph: nx.DiGraph, output_prefix: str) -> Tuple[Path, Path]:
    gexf_path = Path(f"{output_prefix}.gexf")
    png_path = Path(f"{output_prefix}.png")

    gexf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    nx.write_gexf(graph, gexf_path)

    plt.figure(figsize=(14, 10))

    if graph.number_of_nodes() == 0:
        plt.title("Grafo vuoto: nessun forward trovato")
        plt.axis("off")
    else:
        pos = nx.spring_layout(graph, seed=42, k=1.2)
        edge_weights = [max(1.0, float(data.get("weight", 1))) for _, _, data in graph.edges(data=True)]

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
            edge_color="#5a5a5a",
        )
        nx.draw_networkx_labels(graph, pos, font_size=8)

        plt.title("Rete di diffusione Telegram (snowball relations)")
        plt.axis("off")

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

    print("Top 5 archi per numero di messaggi inoltrati:")
    for source, target, data in top_edges:
        print(f"  - {source} -> {target}: messaggi={data.get('weight', 0)}")


def print_matrix_summary(matrix_df: pd.DataFrame) -> None:
    print("\n===== RIEPILOGO MATRICE =====")
    print(f"Sorgenti (righe): {len(matrix_df.index)}")
    print(f"Destinazioni (colonne): {len(matrix_df.columns)}")

    if matrix_df.empty:
        print("Matrice vuota: nessuna relazione disponibile.")
        return

    non_zero = int((matrix_df > 0).sum().sum())
    print(f"Celle con valore > 0: {non_zero}")


def main() -> None:
    cfg: GraphAnalysisConfig = load_project_config(str(DEFAULT_CONFIG_FILE)).graph_analysis
    df = load_dataset(cfg.relations_input_csv)

    edges_df = build_edges_dataframe(
        df=df,
        include_unresolved_sources=cfg.include_unresolved_sources,
    )

    matrix_df = build_relation_matrix(edges_df)
    matrix_path = save_relation_matrix(matrix_df, cfg.relation_matrix_output_csv)

    graph = build_diffusion_graph(
        edges_df=edges_df,
        min_weight=cfg.min_weight,
    )

    gexf_path, png_path = save_graph(graph, cfg.graph_output_prefix)
    print_matrix_summary(matrix_df)
    print_graph_summary(graph)
    print(f"\nMatrice relazioni salvata in: {matrix_path}")
    print(f"\nGrafo GEXF salvato in: {gexf_path}")
    print(f"Immagine PNG salvata in: {png_path}")


if __name__ == "__main__":
    main()
