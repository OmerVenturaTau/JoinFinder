#!/usr/bin/env python3
"""
Build a manuscript graph and export only its clique subgraph.

This mirrors `results_analysis/clusters/geniza_manuscript_graph.py` workflow:
- Build manuscript graph from SQL neighbors tables
- Filter graph to nodes/edges that participate in cliques
- Export the same interactive vis-network HTML graph
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from itertools import combinations

import networkx as nx

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from system import CLUSTERING_DB_CONFIG_PATH
from results_analysis.clusters.geniza_graphs_sql import (
    build_manuscript_graph_from_sql,
    export_manuscript_graph_interactive,
)


def _build_clique_only_subgraph(
    G_ms: nx.Graph,
    cliques: list[list[str]],
) -> nx.Graph:
    """
    Keep only nodes/edges that participate in at least one retained clique.
    """
    clique_nodes: set[str] = set()
    clique_edges: set[tuple[str, str]] = set()
    for clique in cliques:
        members = [str(m) for m in clique]
        clique_nodes.update(members)
        for a, b in combinations(sorted(members), 2):
            clique_edges.add((a, b))

    H = nx.Graph()
    for n in clique_nodes:
        if n in G_ms:
            H.add_node(n, **G_ms.nodes[n])

    for u, v in clique_edges:
        if G_ms.has_edge(u, v):
            H.add_edge(u, v, **G_ms.edges[u, v])

    return H


def _extract_nodes_and_edges_from_interactive_html(html_text: str) -> tuple[list[dict], list[dict]]:
    nodes_match = re.search(
        r"var nodes = new vis\.DataSet\((\[.*?\])\);\s*",
        html_text,
        flags=re.DOTALL,
    )
    edges_match = re.search(
        r"var allEdges = (\[.*?\]);\s*",
        html_text,
        flags=re.DOTALL,
    )
    if nodes_match is None or edges_match is None:
        raise ValueError(
            "Could not parse nodes/edges from input HTML. "
            "Use an HTML file produced by geniza_manuscript_graph export."
        )
    return json.loads(nodes_match.group(1)), json.loads(edges_match.group(1))


def _build_unattributed_graph_from_nodes_edges(nodes: list[dict], edges: list[dict]) -> nx.Graph:
    G = nx.Graph()
    for n in nodes:
        G.add_node(str(n["id"]))
    for e in edges:
        G.add_edge(str(e["from"]), str(e["to"]))
    return G


def _filter_nodes_edges_to_cliques(
    nodes: list[dict],
    edges: list[dict],
    cliques: list[list[str]],
) -> tuple[list[dict], list[dict]]:
    clique_nodes: set[str] = set()
    clique_edges: set[tuple[str, str]] = set()
    for clique in cliques:
        members = [str(m) for m in clique]
        clique_nodes.update(members)
        for a, b in combinations(sorted(members), 2):
            clique_edges.add((a, b))

    filtered_nodes = [n for n in nodes if str(n.get("id")) in clique_nodes]
    filtered_edges = []
    for e in edges:
        u = str(e.get("from"))
        v = str(e.get("to"))
        if tuple(sorted((u, v))) in clique_edges:
            filtered_edges.append(e)
    return filtered_nodes, filtered_edges


def _rewrite_html_with_filtered_nodes_edges(
    html_text: str,
    filtered_nodes: list[dict],
    filtered_edges: list[dict],
) -> str:
    nodes_json = json.dumps(filtered_nodes)
    edges_json = json.dumps(filtered_edges)

    out = re.sub(
        r"var nodes = new vis\.DataSet\(\[.*?\]\);\s*",
        lambda _: f"var nodes = new vis.DataSet({nodes_json});\n    ",
        html_text,
        count=1,
        flags=re.DOTALL,
    )
    out = re.sub(
        r"var allEdges = \[.*?\];\s*",
        lambda _: f"var allEdges = {edges_json};\n    ",
        out,
        count=1,
        flags=re.DOTALL,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and export manuscript graph filtered to clique-only nodes/edges.",
    )
    parser.add_argument(
        "--db-config",
        type=str,
        default=CLUSTERING_DB_CONFIG_PATH,
        help="Path to DB config file (same as used for KNN generation).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["sql", "overall_sql"],
        default="sql",
        help=(
            "Source table: 'sql' = geniza_knn_results (excludes same-manuscript), "
            "'overall_sql' = geniza_knn_results_including_intermanuscripts "
            "(includes same-manuscript)."
        ),
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=0.0,
        help="Minimum similarity_score to include a row from the KNN table.",
    )
    parser.add_argument(
        "--min_ms_edge_weight",
        type=int,
        default=1,
        help="Minimum number of image connections required to include an edge in the manuscript graph.",
    )
    parser.add_argument(
        "--min_clique_size",
        type=int,
        default=3,
        help="Keep only maximal cliques with at least this many manuscripts.",
    )
    parser.add_argument(
        "--min_patches",
        type=int,
        default=None,
        help="Min query/neighbor num_visual_patches to include.",
    )
    parser.add_argument(
        "--min_glyphs",
        type=int,
        default=None,
        help="Min query/neighbor num_glyphs to include.",
    )
    parser.add_argument(
        "--interactive_ms_html",
        type=str,
        required=True,
        help="Output path for interactive HTML manuscript graph (clique-only).",
    )
    parser.add_argument(
        "--input_ms_html",
        type=str,
        default="",
        help="Optional existing manuscript graph HTML to filter directly (fast mode).",
    )
    args = parser.parse_args()

    if args.min_clique_size < 2:
        raise ValueError("--min_clique_size must be >= 2.")

    if args.input_ms_html:
        print(f"Loading existing interactive manuscript graph from {args.input_ms_html}...")
        with open(args.input_ms_html, "r", encoding="utf-8") as f:
            html_text = f.read()
        nodes, edges = _extract_nodes_and_edges_from_interactive_html(html_text)
        G_existing = _build_unattributed_graph_from_nodes_edges(nodes, edges)
        print(
            f"Existing graph: {len(G_existing.nodes()):,} nodes, {len(G_existing.edges()):,} edges."
        )

        print("Finding maximal cliques...")
        cliques = [
            [str(n) for n in clique]
            for clique in nx.find_cliques(G_existing)
            if len(clique) >= args.min_clique_size
        ]
        largest = max((len(c) for c in cliques), default=0)
        print(
            f"Found {len(cliques):,} maximal cliques (size >= {args.min_clique_size}); "
            f"largest size = {largest}."
        )

        filtered_nodes, filtered_edges = _filter_nodes_edges_to_cliques(nodes, edges, cliques)
        print(
            f"Clique-only graph: {len(filtered_nodes):,} nodes, {len(filtered_edges):,} edges."
        )
        output_html = _rewrite_html_with_filtered_nodes_edges(
            html_text=html_text,
            filtered_nodes=filtered_nodes,
            filtered_edges=filtered_edges,
        )
        with open(args.interactive_ms_html, "w", encoding="utf-8") as f:
            f.write(output_html)
        print(f"Done! Open {args.interactive_ms_html} in your browser.")
        return

    use_overall_table = args.source == "overall_sql"
    print(
        f"Building manuscript graph for min_similarity={args.min_similarity}, "
        f"min_ms_edge_weight={args.min_ms_edge_weight}..."
    )
    G_ms, _ = build_manuscript_graph_from_sql(
        db_config_path=args.db_config,
        min_similarity=args.min_similarity,
        min_edge_weight=args.min_ms_edge_weight,
        use_overall_table=use_overall_table,
        min_patches=args.min_patches,
        min_glyphs=args.min_glyphs,
    )
    print(
        f"Full manuscript graph: {len(G_ms.nodes()):,} nodes, {len(G_ms.edges()):,} edges."
    )

    print("Finding maximal cliques...")
    cliques = [
        [str(n) for n in clique]
        for clique in nx.find_cliques(G_ms)
        if len(clique) >= args.min_clique_size
    ]
    largest = max((len(c) for c in cliques), default=0)
    print(
        f"Found {len(cliques):,} maximal cliques (size >= {args.min_clique_size}); "
        f"largest size = {largest}."
    )

    clique_graph = _build_clique_only_subgraph(G_ms, cliques)
    print(
        f"Clique-only graph: {len(clique_graph.nodes()):,} nodes, "
        f"{len(clique_graph.edges()):,} edges."
    )
    print(f"Exporting interactive manuscript HTML to {args.interactive_ms_html}...")
    export_manuscript_graph_interactive(clique_graph, args.interactive_ms_html)
    print(f"Done! Open {args.interactive_ms_html} in your browser.")


if __name__ == "__main__":
    main()
