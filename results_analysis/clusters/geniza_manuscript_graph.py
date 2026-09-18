#!/usr/bin/env python3
"""
Manuscript-level connectivity graph visualization.

Builds and exports an interactive HTML visualization of manuscript relationships,
where nodes are manuscripts and edges represent image connections between them.

Usage example (full manuscript graph):

    python -m results_analysis.clusters.geniza_manuscript_graph \
        --min_similarity 0.7 \
        --min_ms_edge_weight 5 \
        --interactive_ms_html manuscript_graph.html

Usage example (manuscript ego-subgraph around a specific manuscript):

    python -m results_analysis.clusters.geniza_manuscript_graph \
        --min_similarity 0.7 \
        --ms_ego_id 960055 \
        --ms_ego_radius 2 \
        --interactive_ms_html ms_960055_ego.html
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on path when running as
# `python results_analysis/clusters/geniza_manuscript_graph.py`
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import networkx as nx

from system import CLUSTERING_DB_CONFIG_PATH
from results_analysis.clusters.geniza_graphs_sql import (
    build_manuscript_graph_from_sql,
    export_manuscript_graph_interactive,
    make_manuscript_ego_subgraph,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and visualize manuscript-level connectivity graphs from SQL KNN results.",
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
        default="overall_sql",
        help="Source table: 'sql' = geniza_knn_results (excludes same-manuscript), 'overall_sql' = geniza_knn_results_including_intermanuscripts (includes same-manuscript).",
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
        "--ms_ego_id",
        type=str,
        default="",
        help="If set, extract an ego-subgraph around this manuscript id (otherwise builds full graph).",
    )
    parser.add_argument(
        "--ms_ego_radius",
        type=int,
        default=1,
        help="Radius (in hops) for the manuscript ego-subgraph.",
    )
    parser.add_argument(
        "--max_ms_nodes",
        type=int,
        default=2000,
        help="Maximum number of nodes to include in the manuscript ego-subgraph.",
    )
    parser.add_argument(
        "--interactive_ms_html",
        type=str,
        required=True,
        help="Output path for interactive HTML manuscript graph.",
    )
    parser.add_argument(
        "--min_patches",
        type=int,
        default=None,
        help="Min query/neighbor num_visual_patches to include (default: from system.RESULTS_ANALYSIS_MIN_PATCHES).",
    )
    parser.add_argument(
        "--min_glyphs",
        type=int,
        default=None,
        help="Min query/neighbor num_glyphs to include (default: from system.RESULTS_ANALYSIS_MIN_GLYPHS).",
    )

    args = parser.parse_args()

    # Build manuscript graph from SQL
    use_overall_table = args.source == "overall_sql"
    source_desc = "geniza_knn_results_including_intermanuscripts" if use_overall_table else "geniza_knn_results"
    print(f"Building manuscript graph from SQL ({source_desc})...")
    G_ms, ms_counts = build_manuscript_graph_from_sql(
        db_config_path=args.db_config,
        min_similarity=args.min_similarity,
        min_edge_weight=args.min_ms_edge_weight,
        use_overall_table=use_overall_table,
        min_patches=args.min_patches,
        min_glyphs=args.min_glyphs,
    )
    print(
        f"Manuscript graph: {len(G_ms.nodes()):,} nodes, "
        f"{len(G_ms.edges()):,} edges."
    )

    # Optional manuscript ego-subgraph around a specific manuscript id
    target_graph: nx.Graph = G_ms
    if args.ms_ego_id:
        print(
            f"Extracting manuscript ego-subgraph around {args.ms_ego_id!r} "
            f"(radius={args.ms_ego_radius}, max_nodes={args.max_ms_nodes})"
        )
        try:
            H_ms = make_manuscript_ego_subgraph(
                G_ms,
                center_ms_id=args.ms_ego_id,
                radius=args.ms_ego_radius,
                max_nodes=args.max_ms_nodes,
            )
            print(
                f"Manuscript ego-subgraph: {len(H_ms.nodes()):,} nodes, "
                f"{len(H_ms.edges()):,} edges."
            )
            target_graph = H_ms
        except ValueError as e:
            print(f"Error: {e}")
            print("Falling back to full manuscript graph.")

    # Export interactive HTML
    print(
        f"Exporting interactive manuscript HTML to {args.interactive_ms_html} "
        f"({len(target_graph.nodes())} nodes, {len(target_graph.edges())} edges)..."
    )
    export_manuscript_graph_interactive(target_graph, args.interactive_ms_html)
    print(f"Done! Open {args.interactive_ms_html} in your browser.")


if __name__ == "__main__":
    main()
