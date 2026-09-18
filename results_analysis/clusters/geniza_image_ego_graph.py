#!/usr/bin/env python3
"""
Image-level graph visualization (with red/green edges).

Builds and exports an interactive HTML visualization of image relationships.
Green edges = same manuscript; red edges = different manuscript.

Usage example (ego graph around a specific image):

    python -m results_analysis.clusters.geniza_image_ego_graph \
        --image_ego_id IE52850139_P000002_FL52850143.jpg \
        --image_ego_radius 2 \
        --min_similarity 0.7 \
        --interactive_image_html image_ego.html

Usage example (full image graph - all images):

    python -m results_analysis.clusters.geniza_image_ego_graph \
        --min_similarity 0.7 \
        --interactive_image_html all_images.html

    (Omit --max_image_edges to include all edges from the KNN table.)
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on path when running as `python results_analysis/clusters/geniza_image_ego_graph.py`
# (__file__ is under results_analysis/clusters/, so go up two levels to repo root)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import networkx as nx

from system import CLUSTERING_DB_CONFIG_PATH

from results_analysis.clusters.geniza_graphs_sql import (
    build_image_graph_from_overall_sql,
    build_image_graph_from_sql,
    export_image_ego_interactive,
    make_image_ego_subgraph,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and visualize image-level graphs with red/green edge coloring. Can show ego graph around a specific image, or the full graph of all images.",
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
        help="Source table: 'sql' = geniza_knn_results (excludes same-manuscript), 'overall_sql' = geniza_knn_results_including_intermanuscripts (includes same-manuscript, recommended for ego graphs).",
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=0.0,
        help="Minimum similarity_score to include a row from the KNN table.",
    )
    parser.add_argument(
        "--max_image_edges",
        type=int,
        default=None,
        help="Maximum number of image-level edges to build from SQL (for global image graph, before ego extraction).",
    )
    parser.add_argument(
        "--image_ego_id",
        type=str,
        default="",
        help="If provided, extract an ego-subgraph around this image id (image_name). If not provided, shows the full graph of all images.",
    )
    parser.add_argument(
        "--image_ego_radius",
        type=int,
        default=1,
        help="Radius (in hops) for the image ego-subgraph (only used if --image_ego_id is provided).",
    )
    parser.add_argument(
        "--max_image_nodes",
        type=int,
        default=1000,
        help="Maximum number of nodes to include in the image ego-subgraph (only used if --image_ego_id is provided).",
    )
    parser.add_argument(
        "--interactive_image_html",
        type=str,
        required=True,
        help="Output path for interactive HTML image ego graph.",
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

    # Build global image graph from SQL
    if args.source == "sql":
        print("Building global image graph from SQL (geniza_knn_results, excludes same-manuscript)...")
        G_img = build_image_graph_from_sql(
            db_config_path=args.db_config,
            min_similarity=args.min_similarity,
            max_edges=args.max_image_edges,
            min_patches=args.min_patches,
            min_glyphs=args.min_glyphs,
        )
    else:
        print("Building global image graph from SQL (geniza_knn_results_including_intermanuscripts, includes same-manuscript)...")
        G_img = build_image_graph_from_overall_sql(
            db_config_path=args.db_config,
            min_similarity=args.min_similarity,
            max_edges=args.max_image_edges,
            min_patches=args.min_patches,
            min_glyphs=args.min_glyphs,
        )
    print(
        f"Image graph (global): {len(G_img.nodes()):,} nodes, "
        f"{len(G_img.edges()):,} edges."
    )

    # Extract ego-subgraph if a center image is specified, otherwise use full graph
    if args.image_ego_id:
        print(
            f"Extracting ego-subgraph around image {args.image_ego_id!r} "
            f"(radius={args.image_ego_radius}, max_nodes={args.max_image_nodes})"
        )
        try:
            H = make_image_ego_subgraph(
                G_img,
                center_image_id=args.image_ego_id,
                radius=args.image_ego_radius,
                max_nodes=args.max_image_nodes,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return

        print(
            f"Ego-subgraph: {len(H.nodes()):,} nodes, "
            f"{len(H.edges()):,} edges."
        )
        target_graph = H
        center_id = args.image_ego_id
    else:
        print("Using full image graph (no ego extraction).")
        target_graph = G_img
        center_id = ""  # No center: title will show "all images"

    # Export interactive HTML (with red/green edge coloring)
    print(f"Exporting interactive image graph HTML to {args.interactive_image_html}...")
    export_image_ego_interactive(
        target_graph,
        center_image_id=center_id,
        html_path=args.interactive_image_html,
    )
    print(f"Done! Open {args.interactive_image_html} in your browser.")


if __name__ == "__main__":
    main()
