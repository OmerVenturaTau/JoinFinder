#!/usr/bin/env python3
"""Build an interactive manuscript graph from inter-library KNN links only.

An edge is aggregated only when both manuscripts have a non-empty
``normalized_library`` and those libraries differ.

Example:

    python -m results_analysis.clusters.geniza_inter_library_manuscript_graph \
        --min_similarity 0.7 \
        --min_ms_edge_weight 5 \
        --interactive_ms_html inter_library_manuscripts.html
"""

from __future__ import annotations

import argparse
import os
import sys

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
        description="Build a manuscript graph using only inter-library KNN results.",
    )
    parser.add_argument("--db-config", default=CLUSTERING_DB_CONFIG_PATH)
    parser.add_argument(
        "--source",
        choices=["sql", "overall_sql"],
        default="sql",
        help=(
            "KNN source: 'sql' uses geniza_knn_results; 'overall_sql' uses "
            "geniza_knn_results_including_intermanuscripts."
        ),
    )
    parser.add_argument("--min_similarity", type=float, default=0.0)
    parser.add_argument("--min_ms_edge_weight", type=int, default=1)
    parser.add_argument("--ms_ego_id", default="")
    parser.add_argument("--ms_ego_radius", type=int, default=1)
    parser.add_argument("--max_ms_nodes", type=int, default=2000)
    parser.add_argument("--min_patches", type=int, default=None)
    parser.add_argument("--min_glyphs", type=int, default=None)
    parser.add_argument("--interactive_ms_html", required=True)
    args = parser.parse_args()

    source_desc = (
        "geniza_knn_results_including_intermanuscripts"
        if args.source == "overall_sql"
        else "geniza_knn_results"
    )
    print(f"Building inter-library manuscript graph from {source_desc}...")
    graph, _ = build_manuscript_graph_from_sql(
        db_config_path=args.db_config,
        min_similarity=args.min_similarity,
        min_edge_weight=args.min_ms_edge_weight,
        use_overall_table=args.source == "overall_sql",
        min_patches=args.min_patches,
        min_glyphs=args.min_glyphs,
        inter_library_only=True,
    )
    print(f"Manuscript graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges.")

    target_graph: nx.Graph = graph
    if args.ms_ego_id:
        target_graph = make_manuscript_ego_subgraph(
            graph,
            center_ms_id=args.ms_ego_id,
            radius=args.ms_ego_radius,
            max_nodes=args.max_ms_nodes,
        )
        print(
            f"Manuscript ego-subgraph: {target_graph.number_of_nodes():,} nodes, "
            f"{target_graph.number_of_edges():,} edges."
        )

    print(f"Exporting interactive HTML to {args.interactive_ms_html}...")
    export_manuscript_graph_interactive(target_graph, args.interactive_ms_html)
    print(f"Done! Open {args.interactive_ms_html} in your browser.")


if __name__ == "__main__":
    main()
