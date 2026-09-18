#!/usr/bin/env python3
"""
Interactive Geniza graphs built directly from the SQL KNN results table
(`GENIZA_KNN_RESULTS_TABLE`), without going through the Excel export.

Two graph types:
  1. Image-level graph:
     - Nodes: images (identified by `query_image_name` / `neighbor_image_name`).
     - Edges: similarity links with `similarity_score` as weight.
     - Intended for *local* exploration (ego graphs around selected images),
       not for plotting the full 100K-node / 1M-edge graph at once.

  2. Manuscript-level graph:
     - Nodes: manuscript IDs.
     - Edges: aggregated image connections between manuscripts.
       Edge weight = number of image pairs connecting the two manuscripts.
     - This can realistically be built for the *entire* dataset.

This script focuses on:
  - Building graphs from SQL with streaming and progress bars (via tqdm).
  - Providing an interactive HTML for the *manuscript* graph that shows the
    whole picture, with an edge-weight slider and community-aware layout.
  - Allowing ego-subgraph extraction for image-level exploration.

Usage example (full manuscript graph + interactive HTML):

    python -m results_analysis.clusters.geniza_graphs_sql \
        --graph manuscripts \
        --min_similarity 0.7 \
        --min_ms_edge_weight 5 \
        --interactive_ms_html manuscript_graph.html

Usage example (image ego graph for a specific image, static plot only):

    python -m results_analysis.clusters.geniza_graphs_sql \
        --graph images \
        --min_similarity 0.7 \
        --image_ego_id IE52850139_P000002_FL52850143.jpg \
        --image_ego_radius 1 \
        --max_image_nodes 1000
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
from collections import Counter, defaultdict
import math
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from tqdm.auto import tqdm

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from system import (
    CLUSTERING_DB_CONFIG_PATH,
    GENIZA_KNN_RESULTS_TABLE,
    GENIZA_MANUSCRIPT_SHELFMARK_TABLE,
    GENIZA_OVERALL_NEIGHBORS_TABLE,
    RESULTS_ANALYSIS_MIN_GLYPHS,
    RESULTS_ANALYSIS_MIN_PATCHES,
)


def get_db_connection(db_config_path: str):
    """Open the graph database without importing the GPU neighbor pipeline."""
    import psycopg2

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = db_config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)
    section = "postgresql" if "postgresql" in config else "database"
    database = config[section]
    return psycopg2.connect(
        host=database["host"],
        database=database["database"],
        user=database["user"],
        password=database["password"],
        port=database.get("port", 5432),
    )


def load_manuscript_metadata(conn) -> Dict[str, Dict[str, str]]:
    """Load the library and indexed shelfmark displayed by graph exporters."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                manuscript_id,
                normalized_library,
                shelfmark_root,
                shelfmark_running_index
            FROM {GENIZA_MANUSCRIPT_SHELFMARK_TABLE}
            """
        )
        return {
            str(manuscript_id): {
                "library": str(library or "").strip(),
                "shelfmark": str(shelfmark or "").strip(),
                "shelfmark_running_index": str(running_index or "").strip(),
            }
            for manuscript_id, library, shelfmark, running_index in cur.fetchall()
        }


def _manuscript_metadata_attrs(
    manuscript_id: str,
    metadata: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    values = metadata.get(str(manuscript_id), {})
    return {
        "library": values.get("library", ""),
        "shelfmark": values.get("shelfmark", ""),
        "shelfmark_running_index": values.get("shelfmark_running_index", ""),
    }


def _format_indexed_shelfmark(shelfmark: object, running_index: object) -> str:
    """Format a shelfmark root with its stored per-root running index."""
    shelfmark_text = str(shelfmark or "").strip()
    running_index_text = str(running_index or "").strip()
    if shelfmark_text and running_index_text:
        return f"{shelfmark_text} ({running_index_text})"
    return shelfmark_text


def _add_manuscript_metadata_to_image_nodes(
    graph: nx.Graph,
    metadata: Dict[str, Dict[str, str]],
) -> None:
    for _, data in graph.nodes(data=True):
        manuscript_id = data.get("manuscript_id")
        if manuscript_id is None:
            data.update(
                {
                    "library": "",
                    "shelfmark": "",
                    "shelfmark_running_index": "",
                }
            )
        else:
            data.update(_manuscript_metadata_attrs(str(manuscript_id), metadata))


def build_manuscript_graph_from_sql(
    db_config_path: str,
    min_similarity: float = 0.0,
    min_edge_weight: int = 1,
    use_overall_table: bool = False,
    min_patches: int | None = None,
    min_glyphs: int | None = None,
    inter_library_only: bool = False,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Build the manuscript-level graph directly from the SQL KNN results table.

    - Node attribute: total_images (max of total_images_in_query_manuscripts for that manuscript).
    - Edge weight: number of image pairs connecting the two manuscripts.
    
    Args:
        use_overall_table: If True, use geniza_knn_results_including_intermanuscripts
                          instead of geniza_knn_results.
        inter_library_only: Keep only edges whose endpoint libraries are both
            known and different.
    """
    min_p = min_patches if min_patches is not None else RESULTS_ANALYSIS_MIN_PATCHES
    min_g = min_glyphs if min_glyphs is not None else RESULTS_ANALYSIS_MIN_GLYPHS

    table_name = GENIZA_OVERALL_NEIGHBORS_TABLE if use_overall_table else GENIZA_KNN_RESULTS_TABLE
    conn = get_db_connection(db_config_path)

    where_extra = (
        " AND COALESCE(query_num_visual_patches, 0) >= %s"
        " AND COALESCE(query_num_glyphs, 0) >= %s"
        " AND COALESCE(neighbor_num_visual_patches, 0) >= %s"
        " AND COALESCE(neighbor_num_glyphs, 0) >= %s"
    )
    where_params = (min_similarity, min_p, min_g, min_p, min_g)

    # ── Step 1: total rows for progress bar ──────────────────────────────────
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM {table_name} "
            f"WHERE similarity_score >= %s{where_extra}",
            where_params,
        )
        total_rows = int(cur.fetchone()[0])

    ms_pair_counts: Counter[Tuple[str, str]] = Counter()
    # Track maximum similarity observed for each manuscript pair so we can
    # expose a similarity-based slider in the interactive view.
    ms_pair_max_sim: Dict[Tuple[str, str], float] = {}
    # Track example image pairs for each manuscript pair (up to 100 per pair):
    # list of (query_image_name, neighbor_image_name, similarity_score)
    ms_pair_examples: Dict[
        Tuple[str, str], List[Tuple[str | None, str | None, float]]
    ] = {}
    ms_total_images: Dict[str, int] = {}

    # ── Step 2: stream through KNN table and aggregate ───────────────────────
    with conn.cursor(name="ms_edges") as cur:
        cur.itersize = 10_000
        cur.execute(
            f"""
            SELECT
                query_manuscript_id,
                neighbor_manuscript_id,
                total_images_in_query_manuscripts,
                similarity_score,
                query_image_name,
                neighbor_image_name
            FROM {table_name}
            WHERE similarity_score >= %s{where_extra}
            """,
            where_params,
        )

        pbar = tqdm(total=total_rows, desc="Scanning KNN rows for manuscript graph")
        while True:
            rows = cur.fetchmany(10_000)
            if not rows:
                break
            for q_ms, n_ms, total_imgs, sim, q_img, n_img in rows:
                if q_ms is None or n_ms is None:
                    continue
                q_ms = str(q_ms)
                n_ms = str(n_ms)
                # Update total images (take max seen)
                if total_imgs is not None:
                    cur_max = ms_total_images.get(q_ms, 0)
                    if total_imgs > cur_max:
                        ms_total_images[q_ms] = int(total_imgs)
                # Aggregate undirected manuscript pairs
                if q_ms == n_ms:
                    continue
                pair = tuple(sorted((q_ms, n_ms)))
                ms_pair_counts[pair] += 1
                # Update maximum similarity seen for this pair and collect
                # example image pairs (capped at 100 per manuscript pair,
                # keeping the highest-similarity ones).
                if sim is not None:
                    sim_f = float(sim)
                    cur_max = ms_pair_max_sim.get(pair, 0.0)
                    if sim_f > cur_max:
                        ms_pair_max_sim[pair] = sim_f
                if sim is not None and (q_img is not None or n_img is not None):
                    examples = ms_pair_examples.setdefault(pair, [])
                    examples.append(
                        (
                            str(q_img) if q_img is not None else None,
                            str(n_img) if n_img is not None else None,
                            float(sim),
                        )
                    )
                    # Keep only the top-100 by similarity for this pair.
                    if len(examples) > 100:
                        examples.sort(key=lambda x: x[2], reverse=True)
                        del examples[100:]
            pbar.update(len(rows))
        pbar.close()

    manuscript_metadata = load_manuscript_metadata(conn)
    conn.close()

    # ── Step 3: build NetworkX graph ────────────────────────────────────────
    G = nx.Graph()
    for (ms1, ms2), w in ms_pair_counts.items():
        if w < min_edge_weight:
            continue
        ms1_metadata = _manuscript_metadata_attrs(ms1, manuscript_metadata)
        ms2_metadata = _manuscript_metadata_attrs(ms2, manuscript_metadata)
        if inter_library_only and (
            not ms1_metadata["library"]
            or not ms2_metadata["library"]
            or ms1_metadata["library"] == ms2_metadata["library"]
        ):
            continue
        pair = (ms1, ms2)
        max_sim = float(ms_pair_max_sim.get(pair, 0.0))
        examples = ms_pair_examples.get(pair, [])
        if examples:
            # examples are kept sorted by similarity descending when >100,
            # but for safety re-pick the best here.
            best_q_img, best_n_img, best_sim = max(examples, key=lambda x: x[2])
        else:
            best_q_img = best_n_img = None
            best_sim = 0.0
        G.add_node(
            ms1,
            total_images=ms_total_images.get(ms1),
            **ms1_metadata,
        )
        G.add_node(
            ms2,
            total_images=ms_total_images.get(ms2),
            **ms2_metadata,
        )
        G.add_edge(
            ms1,
            ms2,
            weight=int(w),
            max_similarity=max_sim,
            best_q_image=best_q_img,
            best_n_image=best_n_img,
            best_similarity=float(best_sim),
            examples=examples,
        )

    return G, ms_total_images


def build_image_graph_from_sql(
    db_config_path: str,
    min_similarity: float = 0.0,
    max_edges: int | None = None,
    min_patches: int | None = None,
    min_glyphs: int | None = None,
) -> nx.Graph:
    """
    Build the *global* image graph from SQL.

    NOTE: This can be heavy (100k+ nodes, ~1M edges), but we:
      - Stream rows from Postgres with a server-side cursor.
      - Show a progress bar.

    You generally will not want to plot this entire graph at once; instead use
    ego-subgraph extraction (see make_image_ego_subgraph).
    """
    min_p = min_patches if min_patches is not None else RESULTS_ANALYSIS_MIN_PATCHES
    min_g = min_glyphs if min_glyphs is not None else RESULTS_ANALYSIS_MIN_GLYPHS

    conn = get_db_connection(db_config_path)

    where_extra = (
        " AND COALESCE(query_num_visual_patches, 0) >= %s"
        " AND COALESCE(query_num_glyphs, 0) >= %s"
        " AND COALESCE(neighbor_num_visual_patches, 0) >= %s"
        " AND COALESCE(neighbor_num_glyphs, 0) >= %s"
    )
    where_params = (min_similarity, min_p, min_g, min_p, min_g)

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM {GENIZA_KNN_RESULTS_TABLE} "
            f"WHERE similarity_score >= %s{where_extra}",
            where_params,
        )
        total_rows = int(cur.fetchone()[0])

    G = nx.Graph()
    n_edges_added = 0

    with conn.cursor(name="img_edges") as cur:
        cur.itersize = 10_000
        cur.execute(
            f"""
            SELECT
                query_image_name,
                query_image_path,
                query_manuscript_id,
                neighbor_image_name,
                neighbor_image_path,
                neighbor_manuscript_id,
                similarity_score,
                distance_score,
                same_manuscript
            FROM {GENIZA_KNN_RESULTS_TABLE}
            WHERE similarity_score >= %s{where_extra}
            ORDER BY similarity_score DESC
            """,
            where_params,
        )

        pbar = tqdm(total=total_rows, desc="Scanning KNN rows for image graph")
        while True:
            rows = cur.fetchmany(10_000)
            if not rows:
                break
            for (
                q_name,
                q_path,
                q_ms,
                n_name,
                n_path,
                n_ms,
                sim,
                dist,
                _,
            ) in rows:
                if q_name is None or n_name is None:
                    continue
                q_name = str(q_name)
                n_name = str(n_name)

                # Nodes
                G.add_node(
                    q_name,
                    manuscript_id=str(q_ms) if q_ms is not None else None,
                    image_id=q_name,
                    image_path=q_path,
                )
                G.add_node(
                    n_name,
                    manuscript_id=str(n_ms) if n_ms is not None else None,
                    image_id=n_name,
                    image_path=n_path,
                )

                sim_f = float(sim) if sim is not None else 0.0
                dist_f = float(dist) if dist is not None else 0.0
                # Define same manuscript strictly by manuscript_id (9900... number)
                same_flag = (
                    q_ms is not None
                    and n_ms is not None
                    and str(q_ms).strip() == str(n_ms).strip()
                )

                if G.has_edge(q_name, n_name):
                    if sim_f > G[q_name][n_name]["similarity_score"]:
                        G[q_name][n_name].update(
                            {
                                "weight": sim_f,
                                "similarity_score": sim_f,
                                "distance_score": dist_f,
                                "same_manuscript": same_flag,
                            }
                        )
                else:
                    G.add_edge(
                        q_name,
                        n_name,
                        weight=sim_f,
                        similarity_score=sim_f,
                        distance_score=dist_f,
                        same_manuscript=same_flag,
                    )
                    n_edges_added += 1
                    if max_edges is not None and n_edges_added >= max_edges:
                        break
            pbar.update(len(rows))
            if max_edges is not None and n_edges_added >= max_edges:
                break
        pbar.close()

    manuscript_metadata = load_manuscript_metadata(conn)
    conn.close()
    _add_manuscript_metadata_to_image_nodes(G, manuscript_metadata)
    return G


def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """Detect communities (clusters) using greedy modularity."""
    if len(G) == 0:
        return {}
    from networkx.algorithms.community import greedy_modularity_communities

    communities = greedy_modularity_communities(G)
    node_to_comm: Dict[str, int] = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            node_to_comm[str(n)] = cid
    return node_to_comm


def build_image_graph_from_overall_sql(
    db_config_path: str,
    min_similarity: float = 0.0,
    max_edges: int | None = None,
    min_patches: int | None = None,
    min_glyphs: int | None = None,
) -> nx.Graph:
    """
    Build a global image graph from the geniza_knn_results_including_intermanuscripts SQL table.
    
    This table includes inter-manuscript connections (same_manuscript can be True),
    unlike the geniza_knn_results table which excludes same-manuscript neighbors.
    """
    min_p = min_patches if min_patches is not None else RESULTS_ANALYSIS_MIN_PATCHES
    min_g = min_glyphs if min_glyphs is not None else RESULTS_ANALYSIS_MIN_GLYPHS

    conn = get_db_connection(db_config_path)

    where_extra = (
        " AND COALESCE(query_num_visual_patches, 0) >= %s"
        " AND COALESCE(query_num_glyphs, 0) >= %s"
        " AND COALESCE(neighbor_num_visual_patches, 0) >= %s"
        " AND COALESCE(neighbor_num_glyphs, 0) >= %s"
    )
    where_params = (min_similarity, min_p, min_g, min_p, min_g)

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM {GENIZA_OVERALL_NEIGHBORS_TABLE} "
            f"WHERE similarity_score >= %s{where_extra}",
            where_params,
        )
        total_rows = int(cur.fetchone()[0])

    G = nx.Graph()
    n_edges_added = 0

    with conn.cursor(name="overall_img_edges") as cur:
        cur.itersize = 10_000
        cur.execute(
            f"""
            SELECT
                query_image_name,
                query_image_path,
                query_manuscript_id,
                neighbor_image_name,
                neighbor_image_path,
                neighbor_manuscript_id,
                similarity_score,
                distance_score,
                same_manuscript
            FROM {GENIZA_OVERALL_NEIGHBORS_TABLE}
            WHERE similarity_score >= %s{where_extra}
            ORDER BY similarity_score DESC
            """,
            where_params,
        )

        pbar = tqdm(total=total_rows, desc="Scanning overall neighbors table for image graph")
        while True:
            rows = cur.fetchmany(10_000)
            if not rows:
                break
            for (
                q_name,
                q_path,
                q_ms,
                n_name,
                n_path,
                n_ms,
                sim,
                dist,
                _,
            ) in rows:
                if q_name is None or n_name is None:
                    continue
                q_name = str(q_name)
                n_name = str(n_name)

                # Nodes
                G.add_node(
                    q_name,
                    manuscript_id=str(q_ms) if q_ms is not None else None,
                    image_id=q_name,
                    image_path=q_path,
                )
                G.add_node(
                    n_name,
                    manuscript_id=str(n_ms) if n_ms is not None else None,
                    image_id=n_name,
                    image_path=n_path,
                )

                sim_f = float(sim) if sim is not None else 0.0
                dist_f = float(dist) if dist is not None else 0.0
                # Define same manuscript strictly by manuscript_id (9900... number)
                same_flag = (
                    q_ms is not None
                    and n_ms is not None
                    and str(q_ms).strip() == str(n_ms).strip()
                )

                if G.has_edge(q_name, n_name):
                    if sim_f > G[q_name][n_name]["similarity_score"]:
                        G[q_name][n_name].update(
                            {
                                "weight": sim_f,
                                "similarity_score": sim_f,
                                "distance_score": dist_f,
                                "same_manuscript": same_flag,
                            }
                        )
                else:
                    G.add_edge(
                        q_name,
                        n_name,
                        weight=sim_f,
                        similarity_score=sim_f,
                        distance_score=dist_f,
                        same_manuscript=same_flag,
                    )
                    n_edges_added += 1
                    if max_edges is not None and n_edges_added >= max_edges:
                        break
            pbar.update(len(rows))
            if max_edges is not None and n_edges_added >= max_edges:
                break
        pbar.close()

    manuscript_metadata = load_manuscript_metadata(conn)
    conn.close()
    _add_manuscript_metadata_to_image_nodes(G, manuscript_metadata)
    return G


def build_image_graph_from_excel(
    excel_path: str,
    min_similarity: float = 0.0,
    max_edges: int | None = None,
    db_config_path: str | None = None,
) -> nx.Graph:
    """
    Build a global image graph from an Excel neighbors file with columns:
      query_manuscript_id, query_image_path, query_image_name,
      neighbor_manuscript_id, neighbor_image_path, neighbor_image_name,
      similarity_score, distance_score, same_manuscript.

    This is useful when the Excel export includes inter-manuscript connections
    that may not exist in the SQL KNN table.
    """
    required_cols = [
        "query_manuscript_id",
        "query_image_path",
        "query_image_name",
        "neighbor_manuscript_id",
        "neighbor_image_path",
        "neighbor_image_name",
        "similarity_score",
        "distance_score",
        "same_manuscript",
    ]

    print(f"Loading Excel neighbors from {excel_path} ...")
    df = pd.read_excel(excel_path, engine="openpyxl", usecols=required_cols)
    print(f"Loaded {len(df):,} rows from Excel.")

    # Filter by similarity
    df = df[df["similarity_score"] >= min_similarity].copy()
    if max_edges is not None and len(df) > max_edges:
        df = df.nlargest(max_edges, "similarity_score")
    print(f"Using {len(df):,} rows after similarity / max_edges filtering.")

    G = nx.Graph()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building image graph from Excel"):
        q_name = str(row["query_image_name"])
        n_name = str(row["neighbor_image_name"])

        q_ms = row["query_manuscript_id"]
        n_ms = row["neighbor_manuscript_id"]

        # Nodes
        G.add_node(
            q_name,
            manuscript_id=str(q_ms) if pd.notna(q_ms) else None,
            image_id=q_name,
            image_path=row["query_image_path"],
        )
        G.add_node(
            n_name,
            manuscript_id=str(n_ms) if pd.notna(n_ms) else None,
            image_id=n_name,
            image_path=row["neighbor_image_path"],
        )

        sim = float(row["similarity_score"])
        dist = float(row["distance_score"])
        # Define same manuscript strictly by manuscript_id (9900... number)
        same_ms = (
            pd.notna(q_ms)
            and pd.notna(n_ms)
            and str(q_ms).strip() == str(n_ms).strip()
        )

        if G.has_edge(q_name, n_name):
            if sim > G[q_name][n_name]["similarity_score"]:
                G[q_name][n_name].update(
                    {
                        "weight": sim,
                        "similarity_score": sim,
                        "distance_score": dist,
                        "same_manuscript": same_ms,
                    }
                )
        else:
            G.add_edge(
                q_name,
                n_name,
                weight=sim,
                similarity_score=sim,
                distance_score=dist,
                same_manuscript=same_ms,
            )

    if db_config_path:
        conn = get_db_connection(db_config_path)
        try:
            manuscript_metadata = load_manuscript_metadata(conn)
        finally:
            conn.close()
        _add_manuscript_metadata_to_image_nodes(G, manuscript_metadata)

    return G


def community_layout(G: nx.Graph, node_to_comm: Dict[str, int] | None = None) -> Dict:
    """
    Compute a community-aware layout:
    - First, compute sub-layouts per community.
    - Then arrange communities on a grid so they are visually separated.
    """
    if len(G) == 0:
        return {}

    if node_to_comm is None:
        node_to_comm = detect_communities(G)
    if not node_to_comm:
        # Fallback: single spring layout
        return nx.spring_layout(G, seed=42, weight="weight")

    # Group nodes by community id
    comm_to_nodes: Dict[int, List] = defaultdict(list)
    for n in G.nodes():
        cid = int(node_to_comm.get(str(n), 0))
        comm_to_nodes[cid].append(n)

    # Place each community on its own circle, laid out side by side on X-axis.
    layout: Dict = {}
    current_x = 0.0
    padding = 25.0  # horizontal gap between communities (smaller = more compact)
    node_radius_px = 10.0  # approximate node radius in pixels

    for cid, nodes in sorted(comm_to_nodes.items(), key=lambda x: x[0]):
        n_nodes = max(len(nodes), 1)
        if n_nodes == 1:
            center_x = current_x + padding
            center_y = 0.0
            only = nodes[0]
            layout[only] = (center_x, center_y)
            current_x = center_x + padding
            continue

        # Choose circle radius; allow a bit of overlap so layout stays dense.
        # 2πR ≈ n_nodes * (2 * node_radius_px)  =>  R ≈ n_nodes * node_radius_px / π
        R = max(60.0, (n_nodes * node_radius_px) / (2.0 * math.pi))
        center_x = current_x + R + padding
        center_y = 0.0

        for idx, n in enumerate(nodes):
            angle = 2.0 * math.pi * idx / n_nodes
            x = center_x + R * math.cos(angle)
            y = center_y + R * math.sin(angle)
            layout[n] = (x, y)

        # Advance current_x so next community circle does not overlap this one
        current_x = center_x + R + padding

    return layout


def _graph_page_css(width: str, height: str) -> str:
    """Shared CSS for both manuscript and image-ego interactive graph pages."""
    return f"""
    :root {{
      --font-ui: "Manrope", "Poppins", sans-serif;
      --font-display: "Newsreader", Georgia, serif;
      --page-bg: #f5efe4;
      --panel-bg: rgba(252, 250, 246, 0.96);
      --panel-bg-strong: #fffdf8;
      --border-color: #ded2bd;
      --border-strong: #cdbda2;
      --accent-color: #2f5f4f;
      --accent-strong: #24483d;
      --accent-soft: rgba(47, 95, 79, 0.12);
      --secondary-color: #8b5e34;
      --text-main: #1f2937;
      --text-muted: #5f6b63;
      --shadow-card: 0 22px 48px rgba(80, 61, 34, 0.12);
      --shadow-soft: 0 12px 28px rgba(80, 61, 34, 0.08);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      padding: 20px 24px 28px;
      font-family: var(--font-ui);
      background:
        radial-gradient(circle at top left, rgba(202, 173, 126, 0.18), transparent 28%),
        linear-gradient(180deg, #f5efe4 0%, #f7f4ee 22%, #f4f0e8 100%);
      color: var(--text-main);
    }}

    .page-header {{
      margin-bottom: 18px;
      padding: 24px 28px;
      border-radius: 28px;
      border: 1px solid var(--border-color);
      background:
        radial-gradient(circle at top right, rgba(139, 94, 52, 0.12), transparent 34%),
        linear-gradient(180deg, rgba(255, 252, 247, 0.96), rgba(245, 238, 227, 0.96));
      box-shadow: var(--shadow-card);
      text-align: center;
    }}

    .graph-menu-fab {{
      position: fixed;
      top: 44px;
      left: 32px;
      z-index: 9999;
      width: 40px;
      height: 40px;
      border-radius: 999px;
      border: none;
      background: linear-gradient(135deg, var(--accent-color), var(--accent-strong));
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0;
      font-size: 1.1rem;
      line-height: 1;
      cursor: pointer;
      box-shadow: 0 18px 36px rgba(47, 95, 79, 0.32);
      transition: transform 0.12s ease, box-shadow 0.12s ease;
    }}

    .graph-menu-fab:hover {{
      transform: translateY(-1px);
      box-shadow: 0 22px 44px rgba(47, 95, 79, 0.4);
    }}

    .graph-sidebar {{
      position: fixed;
      top: 0;
      left: 0;
      width: 260px;
      height: 100vh;
      padding: 24px 0;
      background:
        linear-gradient(180deg, rgba(255, 252, 247, 0.98), rgba(244, 235, 221, 0.98)),
        var(--page-bg);
      color: var(--text-main);
      box-shadow: 0 24px 44px rgba(61, 47, 30, 0.14);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      border-right: 1px solid var(--border-strong);
      border-radius: 0 24px 24px 0;
      transform: translateX(-260px);
      transition: transform 0.25s ease-in-out;
      z-index: 9998;
      pointer-events: none;
    }}

    .graph-sidebar-open {{
      transform: translateX(0);
      pointer-events: auto;
    }}

    .graph-sidebar-inner {{
      display: flex;
      flex-direction: column;
      flex: 1;
      gap: 20px;
      padding: 88px 16px 20px 16px;
      min-height: 0;
    }}

    .graph-sidebar-image-strip {{
      width: 100%;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid var(--border-color);
      box-shadow: 0 10px 22px rgba(89, 61, 24, 0.10);
    }}

    .graph-sidebar-image-strip img {{
      display: block;
      width: 100%;
      height: auto;
    }}

    .graph-nav {{
      display: flex;
      flex-direction: column;
      flex: 1;
      gap: 10px;
      padding: 14px;
      border-radius: 20px;
      background: rgba(255, 251, 244, 0.92);
      border: 1px solid var(--border-color);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
    }}

    .graph-nav a {{
      display: flex;
      justify-content: center;
      padding: 12px 14px;
      border-radius: 14px;
      color: #433221;
      background: #f6eee1;
      border: 1px solid #e8dac3;
      font-size: 0.95rem;
      font-weight: 600;
      letter-spacing: 0.01em;
      text-align: center;
      text-decoration: none;
      transition: background 0.16s ease, transform 0.08s ease, box-shadow 0.16s ease;
    }}

    .graph-nav a:hover {{
      background: #efe2cf;
      transform: translateY(-1px);
      box-shadow: 0 10px 20px rgba(89, 61, 24, 0.14);
    }}

    .graph-nav a[aria-current="page"] {{
      color: #f8f5ef;
      background: linear-gradient(135deg, var(--accent-color), var(--accent-strong));
      border-color: var(--accent-color);
      box-shadow: 0 14px 28px rgba(47, 95, 79, 0.24);
    }}

    .graph-nav a:last-child {{
      margin-top: auto;
    }}

    h3 {{
      margin: 0 0 8px 0;
      font-weight: 600;
      letter-spacing: -0.01em;
      color: var(--text-main);
      font-family: var(--font-display);
      font-size: clamp(2rem, 3vw, 3rem);
    }}

    .subtitle {{
      margin: 0;
      font-size: 15px;
      line-height: 1.6;
      color: var(--text-muted);
      max-width: 920px;
      margin-left: auto;
      margin-right: auto;
      text-align: center;
    }}

    #controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px 18px;
      align-items: center;
      padding: 18px 20px;
      margin-bottom: 14px;
      border-radius: 24px;
      background: var(--panel-bg);
      border: 1px solid var(--border-color);
      box-shadow: var(--shadow-card);
    }}

    .control-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--text-main);
      flex-wrap: wrap;
    }}

    .control-group label {{
      font-weight: 700;
    }}

    .control-hint {{
      font-size: 11px;
      color: var(--text-muted);
      margin-left: 4px;
    }}

    .weight-range-group {{
      flex-wrap: nowrap;
    }}

    .weight-range-control {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .weight-range-control input[type="number"] {{
      width: 72px;
      text-align: center;
    }}

    .dual-range {{
      --range-start: 0%;
      --range-end: 100%;
      position: relative;
      width: 240px;
      height: 34px;
      flex: 0 0 240px;
    }}

    .dual-range-track {{
      position: absolute;
      top: 50%;
      right: 0;
      left: 0;
      height: 5px;
      border-radius: 999px;
      background: linear-gradient(
        to right,
        var(--border-strong) 0 var(--range-start),
        var(--accent-color) var(--range-start) var(--range-end),
        var(--border-strong) var(--range-end) 100%
      );
      transform: translateY(-50%);
    }}

    .dual-range input[type="range"] {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 34px;
      margin: 0;
      appearance: none;
      -webkit-appearance: none;
      background: transparent;
      pointer-events: none;
    }}

    .dual-range input[type="range"]::-webkit-slider-runnable-track {{
      height: 5px;
      background: transparent;
    }}

    .dual-range input[type="range"]::-webkit-slider-thumb {{
      width: 20px;
      height: 20px;
      margin-top: -7.5px;
      border: 2px solid var(--panel-bg-strong);
      border-radius: 50%;
      background: var(--accent-color);
      box-shadow: 0 2px 8px rgba(47, 95, 79, 0.3);
      appearance: none;
      -webkit-appearance: none;
      cursor: grab;
      pointer-events: auto;
    }}

    .dual-range input[type="range"]::-moz-range-track {{
      height: 5px;
      background: transparent;
    }}

    .dual-range input[type="range"]::-moz-range-thumb {{
      width: 18px;
      height: 18px;
      border: 2px solid var(--panel-bg-strong);
      border-radius: 50%;
      background: var(--accent-color);
      box-shadow: 0 2px 8px rgba(47, 95, 79, 0.3);
      cursor: grab;
      pointer-events: auto;
    }}

    .dual-range .min-weight-range {{ z-index: 3; }}
    .dual-range .max-weight-range {{ z-index: 2; }}

    .library-filter-group {{
      align-items: center;
    }}

    .library-dropdown {{
      position: relative;
      width: 240px;
      min-width: 240px;
    }}

    .library-dropdown summary {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 35px;
      padding: 8px 10px;
      border: 1px solid var(--border-strong);
      border-radius: 12px;
      background: var(--panel-bg-strong);
      color: var(--text-main);
      cursor: pointer;
      list-style: none;
      user-select: none;
    }}

    .library-dropdown summary::-webkit-details-marker {{
      display: none;
    }}

    .library-dropdown summary::after {{
      content: "▾";
      color: var(--text-muted);
      transition: transform 0.12s ease;
    }}

    .library-dropdown[open] summary {{
      border-color: var(--accent-color);
      box-shadow: 0 0 0 4px var(--accent-soft);
    }}

    .library-dropdown[open] summary::after {{
      transform: rotate(180deg);
    }}

    .library-checkboxes {{
      display: flex;
      flex-direction: column;
      gap: 2px;
      position: absolute;
      z-index: 30;
      top: calc(100% + 8px);
      left: 0;
      width: max-content;
      min-width: 100%;
      max-width: min(560px, 82vw);
      max-height: 240px;
      overflow-y: auto;
      padding: 12px;
      border: 1px solid var(--border-strong);
      border-radius: 14px;
      background: var(--panel-bg-strong);
      box-shadow: var(--shadow-card);
    }}

    .library-checkboxes .checkbox-option {{
      display: flex;
      align-items: center;
      gap: 5px;
      width: 100%;
      padding: 6px 8px;
      border-radius: 8px;
      font-weight: 500;
      white-space: nowrap;
      cursor: pointer;
    }}

    .library-checkboxes .checkbox-option:hover {{
      background: var(--accent-soft);
    }}

    .library-checkboxes input[type="checkbox"] {{
      accent-color: var(--accent-color);
    }}

    .library-filter-group,
    .library-target-group {{
      flex-wrap: nowrap;
    }}

    .library-connection-count {{
      color: var(--text-muted);
      font-size: 12px;
      min-width: 104px;
      white-space: nowrap;
    }}

    .find-joins-status {{
      display: none;
    }}

    #findLibraryJoins {{
      flex: 0 0 auto;
      min-width: 94px;
      white-space: nowrap;
    }}

    #findLibraryJoins[aria-pressed="true"] {{
      background: linear-gradient(135deg, var(--secondary-color), #6f4828);
      box-shadow: 0 0 0 4px rgba(139, 94, 52, 0.16),
        0 12px 24px rgba(139, 94, 52, 0.24);
    }}

    .focus-result-controls {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }}

    #focusLibraryResult {{
      min-width: 104px;
      white-space: nowrap;
    }}

    #focusLibraryResult:disabled {{
      cursor: default;
      filter: grayscale(0.7);
      opacity: 0.5;
      transform: none;
    }}

    .focus-result-status {{
      min-width: 42px;
      color: var(--text-muted);
      font-size: 12px;
      text-align: center;
    }}

    #networkContainer {{ position: relative; }}
    #loadingOverlay {{
      position: absolute; top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(252, 250, 246, 0.92); display: flex; flex-direction: column;
      align-items: center; justify-content: center; z-index: 10;
      border-radius: 24px;
    }}
    .loading-spinner {{
      width: 40px; height: 40px; border: 3px solid var(--border-color);
      border-top-color: var(--accent-color); border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }}
    .loading-text {{ margin-top: 12px; font-size: 14px; color: var(--text-muted); }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

    input[type="text"],
    input[type="number"] {{
      padding: 8px 10px;
      border-radius: 12px;
      border: 1px solid var(--border-strong);
      font-size: 13px;
      background: var(--panel-bg-strong);
      color: var(--text-main);
    }}

    input[type="text"]:focus,
    input[type="number"]:focus {{
      outline: none;
      border-color: var(--accent-color);
      box-shadow: 0 0 0 4px var(--accent-soft);
      background: #ffffff;
    }}

    input[type="range"] {{
      accent-color: var(--accent-color);
    }}

    button {{
      padding: 8px 14px;
      border-radius: 999px;
      border: none;
      font-size: 13px;
      font-weight: 700;
      color: #ffffff;
      background: linear-gradient(135deg, var(--accent-color), var(--accent-strong));
      cursor: pointer;
      box-shadow: 0 12px 24px rgba(47, 95, 79, 0.18);
      transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
    }}

    button:hover {{
      filter: brightness(1.02);
      box-shadow: 0 16px 28px rgba(47, 95, 79, 0.24);
      transform: translateY(-1px);
    }}

    button:active {{
      box-shadow: 0 10px 18px rgba(47, 95, 79, 0.18);
      transform: translateY(0);
    }}

    .search-status {{
      font-size: 12px;
      color: var(--text-muted);
    }}

    #networkContainer {{
      display: flex;
      flex-direction: row;
      align-items: stretch;
      width: {width};
      height: {height};
      gap: 14px;
    }}

    #network {{
      flex: 3 1 0%;
      border-radius: 24px;
      border: 1px solid var(--border-color);
      background: var(--panel-bg);
      box-shadow: var(--shadow-card);
    }}

    #nodeInfo {{
      flex: 1 1 0%;
      padding: 16px 18px;
      border-radius: 24px;
      border: 1px solid var(--border-color);
      background: var(--panel-bg);
      font-family: var(--font-ui);
      font-size: 13px;
      overflow-y: auto;
      box-shadow: var(--shadow-card);
    }}

    #nodeInfo strong {{
      font-weight: 600;
    }}

    #nodeInfo code {{
      background: #f4ede2;
      border-radius: 8px;
      padding: 2px 5px;
      font-size: 12px;
    }}

    #nodeInfo a {{
      color: var(--accent-color);
      font-weight: 700;
      text-decoration: none;
    }}

    #nodeInfo a:hover {{
      text-decoration: underline;
    }}

    #nodeInfo table {{
      margin-top: 4px;
      width: 100%;
    }}

    #nodeInfo th {{
      background: #efe3d0;
      color: var(--text-main);
    }}

    #nodeInfo td,
    #nodeInfo th {{
      border: 1px solid var(--border-color);
      padding: 4px 6px;
      text-align: left;
    }}

    @media (max-width: 980px) {{
      body {{
        padding: 16px;
      }}

      #networkContainer {{
        flex-direction: column;
        height: auto;
      }}

      #network {{
        min-height: 560px;
      }}

      #nodeInfo {{
        min-height: 240px;
      }}
    }}
"""


def _build_manuscript_graph_html(
    nodes_json: str,
    edges_json: str,
    min_w: int,
    max_w: int,
    min_s: float,
    max_s: float,
    width: str,
    height: str,
) -> str:
    """Build the full HTML for the manuscript connectivity graph (same style as image ego)."""
    style_content = _graph_page_css(width, height)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Manuscript graph</title>
  <style type="text/css">
{style_content}
  </style>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
  <button
    type="button"
    class="graph-menu-fab"
    id="graphMenuToggle"
    aria-label="Toggle navigation menu"
    aria-controls="graphSidebar"
    aria-expanded="false"
  >
    ☰
  </button>
  <aside class="graph-sidebar" id="graphSidebar">
    <div class="graph-sidebar-inner">
      <div class="graph-sidebar-image-strip">
        <img src="https://transcriptus.org/images/midrash.png" alt="Midrash banner">
      </div>
      <nav class="graph-nav" aria-label="Site navigation">
        <a href="https://transcriptus.org/#/">Search Ms-id</a>
        <a href="https://transcriptus.org/#/shelfmarks">Shelfmarks</a>
        <a href="https://transcriptus.org/#/clusters">Clusters</a>
        <a href="https://transcriptus.org/manuscript_graph_filtered.html" aria-current="page">Graph View</a>
        <a href="https://transcriptus.org/#/theory">Theory</a>
        <a href="https://transcriptus.org/#/about">About</a>
      </nav>
    </div>
  </aside>
  <div class="page-header">
    <h3>Manuscript connectivity graph</h3>
    <div class="subtitle">
      Filter connections, then click a manuscript or edge for details.
    </div>
  </div>
  <div id="controls">
    <div class="control-group">
      <label for="searchMsId">Jump to manuscript:</label>
      <input type="text" id="searchMsId" style="width: 150px;" placeholder="e.g. 960055">
      <button id="searchMsBtn">Go</button>
      <span id="searchMsStatus" class="search-status"></span>
    </div>
    <div class="control-group weight-range-group">
      <label for="minWeight">Edge weight:</label>
      <span>From</span>
      <div class="weight-range-control">
        <input type="number" id="minWeightInput" aria-label="Minimum edge weight" min="{min_w}" max="{max_w}" value="{min_w}" step="1">
        <div id="weightRange" class="dual-range">
          <div class="dual-range-track"></div>
          <input class="min-weight-range" type="range" id="minWeight" name="minWeight" aria-label="Minimum edge weight slider" min="{min_w}" max="{max_w}" value="{min_w}" step="1">
          <input class="max-weight-range" type="range" id="maxWeight" name="maxWeight" aria-label="Maximum edge weight slider" min="{min_w}" max="{max_w}" value="{max_w}" step="1">
        </div>
        <input type="number" id="maxWeightInput" aria-label="Maximum edge weight" min="{min_w}" max="{max_w}" value="{max_w}" step="1">
      </div>
      <span>To</span>
    </div>
    <div class="control-group">
      <label for="minSim">Min similarity:</label>
      <input type="range" id="minSim" name="minSim" min="{min_s:.3f}" max="{max_s:.3f}" value="{min_s:.3f}" step="0.01">
      <input type="number" id="minSimInput" min="{min_s:.3f}" max="{max_s:.3f}" value="{min_s:.3f}" step="0.01" style="width: 80px;">
      <span id="minSimLabel">{min_s:.3f}</span>
    </div>
    <div class="control-group library-filter-group">
      <label>From libraries:</label>
      <details id="libraryDropdown" class="library-dropdown">
        <summary><span id="librarySummary">All libraries</span></summary>
        <div id="libraryFilters" class="library-checkboxes">
          <label class="checkbox-option"><input type="checkbox" id="allLibraries" checked> <span id="allLibrariesLabel">All libraries</span></label>
        </div>
      </details>
    </div>
    <div class="control-group library-target-group">
      <label>To libraries:</label>
      <details id="connectedLibraryDropdown" class="library-dropdown">
        <summary><span id="connectedLibrarySummary">All libraries</span></summary>
        <div id="connectedLibraryFilters" class="library-checkboxes">
          <label class="checkbox-option"><input type="checkbox" id="allConnectedLibraries" checked> <span id="allConnectedLibrariesLabel">All libraries</span></label>
        </div>
      </details>
      <button type="button" id="findLibraryJoins" aria-pressed="false" title="Toggle inter-library joins mode. It keeps the current library, similarity, and edge-weight filters, and limits their results to connections between different libraries.">Find joins</button>
      <span class="focus-result-controls">
        <button type="button" id="focusLibraryResult" disabled title="Center the next manuscript from the explicitly selected libraries without zooming out.">Focus result</button>
        <span id="focusResultStatus" class="focus-result-status">0 / 0</span>
      </span>
      <span id="libraryConnectionCount" class="library-connection-count"></span>
      <span id="findJoinsStatus" class="find-joins-status"></span>
    </div>
  </div>
  <div id="networkContainer">
    <div id="network"></div>
    <div id="nodeInfo">Click a manuscript node or edge to see details here.</div>
  </div>
  <script type="text/javascript">
    var nodes = new vis.DataSet({nodes_json});
    var allEdges = {edges_json};
    var edgesDS = new vis.DataSet(allEdges);

    var container = document.getElementById('network');
    var data = {{
      nodes: nodes,
      edges: edgesDS
    }};
    var options = {{
      physics: false,
      interaction: {{ hover: true }},
      nodes: {{
        shape: 'dot',
        size: 24,
        font: {{
          size: 16,
          color: '#000000',
          face: 'sans-serif',
          bold: {{ size: 16 }}
        }},
        borderWidth: 3,
        borderWidthSelected: 6
      }},
      edges: {{
        scaling: {{
          min: 3,
          max: 12
        }},
        smooth: false,
        hoverWidth: 4,
        selectionWidth: 5,
        font: {{ size: 14, strokeWidth: 4 }}
      }}
    }};

    var network = new vis.Network(container, data, options);

    var slider = document.getElementById('minWeight');
    var sliderInput = document.getElementById('minWeightInput');
    var maxWeightSlider = document.getElementById('maxWeight');
    var maxWeightInput = document.getElementById('maxWeightInput');
    var weightRange = document.getElementById('weightRange');
    var simSlider = document.getElementById('minSim');
    var simInput = document.getElementById('minSimInput');
    var simLabel = document.getElementById('minSimLabel');
    var libraryFilters = document.getElementById('libraryFilters');
    var libraryDropdown = document.getElementById('libraryDropdown');
    var allLibrariesCheckbox = document.getElementById('allLibraries');
    var allLibrariesLabel = document.getElementById('allLibrariesLabel');
    var librarySummary = document.getElementById('librarySummary');
    var connectedLibraryFilters = document.getElementById('connectedLibraryFilters');
    var connectedLibraryDropdown = document.getElementById('connectedLibraryDropdown');
    var allConnectedLibrariesCheckbox = document.getElementById('allConnectedLibraries');
    var allConnectedLibrariesLabel = document.getElementById('allConnectedLibrariesLabel');
    var connectedLibrarySummary = document.getElementById('connectedLibrarySummary');
    var findLibraryJoins = document.getElementById('findLibraryJoins');
    var libraryConnectionCount = document.getElementById('libraryConnectionCount');
    var findJoinsStatus = document.getElementById('findJoinsStatus');
    var focusLibraryResult = document.getElementById('focusLibraryResult');
    var focusResultStatus = document.getElementById('focusResultStatus');
    var infoDiv = document.getElementById('nodeInfo');
    var searchInput = document.getElementById('searchMsId');
    var searchBtn = document.getElementById('searchMsBtn');
    var searchStatus = document.getElementById('searchMsStatus');
    var interLibraryJoinsMode = false;
    var focusCandidateIds = [];
    var focusCandidateIndex = -1;
    var focusRequestId = 0;
    var graphMenuToggle = document.getElementById('graphMenuToggle');
    var graphSidebar = document.getElementById('graphSidebar');

    function escapeHtml(value) {{
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    function manuscriptSearchHref(manuscriptId) {{
      return 'https://transcriptus.org/#/?manuscript_id=' +
        encodeURIComponent(manuscriptId) + '&library=Genizah';
    }}

    function manuscriptSearchLink(manuscriptId) {{
      var id = String(manuscriptId || '');
      return '<a href="' + manuscriptSearchHref(id) +
        '" target="_blank" rel="noopener noreferrer"><code>' +
        escapeHtml(id) + '</code></a>';
    }}

    function nodeShelfmark(node) {{
      return node.shelfmarkLabel || node.shelfmark || 'Unknown shelfmark';
    }}

    function setGraphSidebarOpen(isOpen) {{
      graphSidebar.classList.toggle('graph-sidebar-open', isOpen);
      graphMenuToggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      graphMenuToggle.textContent = isOpen ? '✕' : '☰';
    }}

    graphMenuToggle.addEventListener('click', function(event) {{
      event.stopPropagation();
      setGraphSidebarOpen(!graphSidebar.classList.contains('graph-sidebar-open'));
    }});

    graphSidebar.addEventListener('click', function(event) {{
      event.stopPropagation();
    }});

    document.addEventListener('click', function() {{
      setGraphSidebarOpen(false);
    }});

    document.addEventListener('keydown', function(event) {{
      if (event.key === 'Escape') setGraphSidebarOpen(false);
    }});

    var libraryNodeCounts = {{}};
    nodes.get().forEach(function(node) {{
      var library = node.library || "";
      if (library) libraryNodeCounts[library] = (libraryNodeCounts[library] || 0) + 1;
    }});
    function formatNodeCount(count) {{
      return count + (count === 1 ? ' node' : ' nodes');
    }}
    allLibrariesLabel.textContent =
      'All libraries (' + formatNodeCount(nodes.getIds().length) + ')';
    allConnectedLibrariesLabel.textContent =
      'All libraries (' + formatNodeCount(nodes.getIds().length) + ')';

    Object.keys(libraryNodeCounts)
      .sort(function(a, b) {{ return a.localeCompare(b); }})
      .forEach(function(library) {{
        var label = document.createElement('label');
        label.className = 'checkbox-option';
        var checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'library-checkbox';
        checkbox.value = library;
        checkbox.checked = true;
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(
          ' ' + library + ' (' + formatNodeCount(libraryNodeCounts[library]) + ')'
        ));
        libraryFilters.appendChild(label);

        var targetLabel = label.cloneNode(true);
        var targetCheckbox = targetLabel.querySelector('input');
        targetCheckbox.className = 'connected-library-checkbox';
        targetCheckbox.checked = true;
        connectedLibraryFilters.appendChild(targetLabel);
      }});

    function getSelectedLibraries(filters, allCheckbox, checkboxClass) {{
      if (allCheckbox.checked) return null;
      return new Set(
        Array.from(filters.querySelectorAll('.' + checkboxClass + ':checked'))
          .map(function(checkbox) {{ return checkbox.value; }})
      );
    }}

    function updateDropdownSummary(filters, allCheckbox, checkboxClass, summaryElement) {{
      var checked = Array.from(
        filters.querySelectorAll('.' + checkboxClass + ':checked')
      );
      var summary;
      if (allCheckbox.checked) {{
        summary = 'All libraries';
      }} else if (checked.length === 0) {{
        summary = 'No libraries';
      }} else if (checked.length === 1) {{
        summary = checked[0].value;
      }} else {{
        summary = checked.length + ' libraries';
      }}
      summaryElement.textContent = summary;
    }}

    function updateLibrarySummaries() {{
      updateDropdownSummary(
        libraryFilters, allLibrariesCheckbox, 'library-checkbox', librarySummary
      );
      updateDropdownSummary(
        connectedLibraryFilters, allConnectedLibrariesCheckbox,
        'connected-library-checkbox', connectedLibrarySummary
      );
    }}

    document.addEventListener('pointerdown', function(event) {{
      if (libraryDropdown.open && !libraryDropdown.contains(event.target)) {{
        libraryDropdown.open = false;
      }}
      if (connectedLibraryDropdown.open &&
          !connectedLibraryDropdown.contains(event.target)) {{
        connectedLibraryDropdown.open = false;
      }}
    }});
    document.addEventListener('keydown', function(event) {{
      if (event.key === 'Escape') {{
        libraryDropdown.open = false;
        connectedLibraryDropdown.open = false;
      }}
    }});

    function updateEdges() {{
      focusRequestId += 1;
      var minW = parseInt(slider.value);
      if (isNaN(minW)) minW = {min_w};
      var maxW = parseInt(maxWeightSlider.value);
      if (isNaN(maxW)) maxW = {max_w};

      var minS = parseFloat(simSlider.value);
      if (isNaN(minS)) {{
        minS = {min_s:.3f};
      }}
      simLabel.textContent = minS.toFixed(3);
      simInput.value = minS.toFixed(3);
      var selectedLibraries = getSelectedLibraries(
        libraryFilters, allLibrariesCheckbox, 'library-checkbox'
      );
      var targetLibraries = getSelectedLibraries(
        connectedLibraryFilters, allConnectedLibrariesCheckbox,
        'connected-library-checkbox'
      );
      var libraryFiltered = interLibraryJoinsMode ||
        selectedLibraries !== null || targetLibraries !== null;

      var filtered = allEdges.filter(function(e) {{
        var weightOk = e.weight >= minW && e.weight <= maxW;
        var libraryOk;
        var sourceFromOk = selectedLibraries === null ||
          selectedLibraries.has(e.fromLibrary);
        var sourceToOk = selectedLibraries === null ||
          selectedLibraries.has(e.toLibrary);
        var targetFromOk = targetLibraries === null ||
          targetLibraries.has(e.fromLibrary);
        var targetToOk = targetLibraries === null ||
          targetLibraries.has(e.toLibrary);
        libraryOk = (sourceFromOk && targetToOk) || (sourceToOk && targetFromOk);
        if (interLibraryJoinsMode) {{
          libraryOk = libraryOk && !!e.fromLibrary && !!e.toLibrary &&
            e.fromLibrary !== e.toLibrary;
        }}
        return weightOk && e.similarity >= minS && libraryOk;
      }});
      edgesDS.clear();
      edgesDS.add(filtered);
      libraryConnectionCount.textContent =
        filtered.length + (filtered.length === 1 ? ' connection' : ' connections');
      if (interLibraryJoinsMode) {{
        findJoinsStatus.textContent = filtered.length +
          (filtered.length === 1 ? ' inter-library join shown.' : ' inter-library joins shown.');
      }}

      var visibleNodes = {{}};
      if (libraryFiltered) {{
        filtered.forEach(function(e) {{
          visibleNodes[e.from] = true;
          visibleNodes[e.to] = true;
        }});
        if (!interLibraryJoinsMode && selectedLibraries !== null &&
            targetLibraries === null) {{
          nodes.get().forEach(function(n) {{
            if (selectedLibraries.has(n.library)) visibleNodes[n.id] = true;
          }});
        }}
      }}
      var explicitLibraries = new Set();
      if (selectedLibraries !== null) {{
        selectedLibraries.forEach(function(library) {{ explicitLibraries.add(library); }});
      }}
      if (targetLibraries !== null) {{
        targetLibraries.forEach(function(library) {{ explicitLibraries.add(library); }});
      }}
      var previousFocusedId = focusCandidateIndex >= 0
        ? focusCandidateIds[focusCandidateIndex]
        : null;
      focusCandidateIds = nodes.get().filter(function(node) {{
        var isVisible = !libraryFiltered || !!visibleNodes[node.id];
        return isVisible && explicitLibraries.has(node.library);
      }}).sort(function(a, b) {{
        var aKey = [a.library || '', a.shelfmark || '',
          a.shelfmarkRunningIndex || '', String(a.id)].join('\\u0000');
        var bKey = [b.library || '', b.shelfmark || '',
          b.shelfmarkRunningIndex || '', String(b.id)].join('\\u0000');
        return aKey.localeCompare(bKey);
      }}).map(function(node) {{ return node.id; }});
      focusCandidateIndex = previousFocusedId === null
        ? -1
        : focusCandidateIds.indexOf(previousFocusedId);
      var focusCandidateSet = new Set(
        focusCandidateIds.map(function(id) {{ return String(id); }})
      );
      nodes.update(nodes.getIds().map(function(id) {{
        var highlighted = focusCandidateSet.has(String(id));
        return {{
          id: id,
          hidden: libraryFiltered && !visibleNodes[id],
          borderWidth: highlighted ? 6 : 3,
          shadow: highlighted
            ? {{ enabled: true, color: 'rgba(139, 94, 52, 0.65)', size: 18, x: 0, y: 0 }}
            : false
        }};
      }}));
      focusLibraryResult.disabled = focusCandidateIds.length === 0;
      focusResultStatus.textContent = focusCandidateIds.length === 0
        ? '0 / 0'
        : ((focusCandidateIndex >= 0 ? focusCandidateIndex + 1 : 0) +
           ' / ' + focusCandidateIds.length);
      if (libraryFiltered && Object.keys(visibleNodes).length === 0) {{
        infoDiv.textContent = 'No nodes match the current filters.';
      }} else if (infoDiv.textContent === 'No nodes match the current filters.') {{
        infoDiv.textContent = 'Click a manuscript node or edge to see details here.';
      }}
    }}

    function getExplicitLibraryCount() {{
      var selectedLibraries = getSelectedLibraries(
        libraryFilters, allLibrariesCheckbox, 'library-checkbox'
      );
      var targetLibraries = getSelectedLibraries(
        connectedLibraryFilters, allConnectedLibrariesCheckbox,
        'connected-library-checkbox'
      );
      var libraries = new Set();
      if (selectedLibraries !== null) {{
        selectedLibraries.forEach(function(library) {{ libraries.add(library); }});
      }}
      if (targetLibraries !== null) {{
        targetLibraries.forEach(function(library) {{ libraries.add(library); }});
      }}
      return libraries.size;
    }}

    function focusCandidate(index) {{
      if (focusCandidateIds.length === 0) return;
      focusCandidateIndex = ((index % focusCandidateIds.length) +
        focusCandidateIds.length) % focusCandidateIds.length;
      var nodeId = focusCandidateIds[focusCandidateIndex];
      var requestId = ++focusRequestId;
      var currentScale = network.getScale();
      network.redraw();
      focusResultStatus.textContent = (focusCandidateIndex + 1) +
        ' / ' + focusCandidateIds.length;
      window.requestAnimationFrame(function() {{
        if (requestId !== focusRequestId) return;
        network.redraw();
        window.requestAnimationFrame(function() {{
          if (requestId !== focusRequestId) return;
          network.focus(nodeId, {{
            scale: Math.max(currentScale, 0.85),
            animation: {{ duration: 350, easingFunction: 'easeInOutQuad' }}
          }});
        }});
      }});
    }}

    function applyWeightRange(changedBound) {{
      var minValue = parseInt(
        changedBound === 'min' ? sliderInput.value : slider.value
      );
      var maxValue = parseInt(
        changedBound === 'max' ? maxWeightInput.value : maxWeightSlider.value
      );
      if (isNaN(minValue)) minValue = {min_w};
      if (isNaN(maxValue)) maxValue = {max_w};
      minValue = Math.max({min_w}, Math.min({max_w}, minValue));
      maxValue = Math.max({min_w}, Math.min({max_w}, maxValue));
      if (minValue > maxValue) {{
        if (changedBound === 'min') maxValue = minValue;
        else minValue = maxValue;
      }}
      slider.value = minValue;
      sliderInput.value = minValue;
      maxWeightSlider.value = maxValue;
      maxWeightInput.value = maxValue;
      updateWeightRangeFill();
      updateEdges();
    }}

    function updateWeightRangeFill() {{
      var fullRange = {max_w} - {min_w};
      var startPercent = fullRange > 0
        ? ((parseInt(slider.value) - {min_w}) / fullRange) * 100
        : 0;
      var endPercent = fullRange > 0
        ? ((parseInt(maxWeightSlider.value) - {min_w}) / fullRange) * 100
        : 100;
      weightRange.style.setProperty('--range-start', startPercent + '%');
      weightRange.style.setProperty('--range-end', endPercent + '%');
    }}

    slider.addEventListener('input', function() {{
      slider.style.zIndex = 4;
      maxWeightSlider.style.zIndex = 3;
      sliderInput.value = slider.value;
      applyWeightRange('min');
    }});
    sliderInput.addEventListener('change', function() {{
      slider.style.zIndex = 4;
      maxWeightSlider.style.zIndex = 3;
      applyWeightRange('min');
    }});
    maxWeightSlider.addEventListener('input', function() {{
      slider.style.zIndex = 3;
      maxWeightSlider.style.zIndex = 4;
      maxWeightInput.value = maxWeightSlider.value;
      applyWeightRange('max');
    }});
    maxWeightInput.addEventListener('change', function() {{
      slider.style.zIndex = 3;
      maxWeightSlider.style.zIndex = 4;
      applyWeightRange('max');
    }});
    function handleLibraryCheckboxChange(event, filters, allCheckbox, checkboxClass) {{
      if (!event.target.matches('input[type="checkbox"]')) return;
      var libraryCheckboxes = Array.from(
        filters.querySelectorAll('.' + checkboxClass)
      );
      if (event.target === allCheckbox) {{
        libraryCheckboxes.forEach(function(checkbox) {{
          checkbox.checked = allCheckbox.checked;
        }});
        allCheckbox.indeterminate = false;
      }} else {{
        var checkedCount = libraryCheckboxes.filter(function(checkbox) {{
          return checkbox.checked;
        }}).length;
        allCheckbox.checked = checkedCount === libraryCheckboxes.length;
        allCheckbox.indeterminate = checkedCount > 0 &&
          checkedCount < libraryCheckboxes.length;
      }}
      updateLibrarySummaries();
      updateEdges();
      if (getExplicitLibraryCount() === 1 && focusCandidateIds.length > 0) {{
        focusCandidate(0);
      }}
    }}
    libraryFilters.addEventListener('change', function(event) {{
      handleLibraryCheckboxChange(
        event, libraryFilters, allLibrariesCheckbox, 'library-checkbox'
      );
    }});
    connectedLibraryFilters.addEventListener('change', function(event) {{
      handleLibraryCheckboxChange(
        event, connectedLibraryFilters, allConnectedLibrariesCheckbox,
        'connected-library-checkbox'
      );
    }});
    findLibraryJoins.addEventListener('click', function() {{
      interLibraryJoinsMode = !interLibraryJoinsMode;
      findLibraryJoins.setAttribute(
        'aria-pressed', interLibraryJoinsMode ? 'true' : 'false'
      );
      findJoinsStatus.textContent = '';
      updateEdges();
    }});
    focusLibraryResult.addEventListener('click', function() {{
      focusCandidate(focusCandidateIndex + 1);
    }});
    simSlider.addEventListener('input', updateEdges);
    simInput.addEventListener('change', function() {{
      var v = parseFloat(simInput.value);
      if (isNaN(v)) {{
        v = {min_s:.3f};
      }}
      v = Math.max({min_s:.3f}, Math.min({max_s:.3f}, v));
      simInput.value = v.toFixed(3);
      simSlider.value = v.toFixed(3);
      updateEdges();
    }});
    // Initial filter
    updateWeightRangeFill();
    updateEdges();

    // Node / edge click handler to show manuscript and connection details.
    network.on("click", function (params) {{
      // Node clicked
      if (params.nodes && params.nodes.length > 0) {{
        var nodeId = params.nodes[0];
        var node = nodes.get(nodeId);
        if (node) {{
          infoDiv.innerHTML =
            "<strong>Manuscript:</strong> " + manuscriptSearchLink(node.id) + "<br/>" +
            "<strong>Shelfmark:</strong> " + escapeHtml(nodeShelfmark(node)) + "<br/>" +
            "<strong>Library:</strong> " + escapeHtml(node.library || "Unknown") +
            (node.totalImages == null ? "" : "<br/><strong>Images:</strong> " + escapeHtml(node.totalImages));
        }}
        return;
      }}

      // Edge clicked
      if (params.edges && params.edges.length > 0) {{
        var edgeId = params.edges[0];
        var edge = edgesDS.get(edgeId);
        if (edge) {{
          var fromId = edge.from;
          var toId = edge.to;
          var fromNode = nodes.get(fromId) || {{}};
          var toNode = nodes.get(toId) || {{}};

          var examples = edge.examples || [];
          var examplesHtml = "";
          if (examples.length > 0) {{
            examplesHtml += "<br/><strong>Image pair examples (up to 100, sorted by similarity):</strong><br/>";
            examplesHtml += "<table style='border-collapse: collapse; font-size: 11px;'>";
            examplesHtml += "<tr>" +
              "<th style='border:1px solid #ccc; padding:2px 4px;'>#</th>" +
              "<th style='border:1px solid #ccc; padding:2px 4px;'>Query image</th>" +
              "<th style='border:1px solid #ccc; padding:2px 4px;'>Neighbor image</th>" +
              "<th style='border:1px solid #ccc; padding:2px 4px;'>Similarity</th>" +
              "</tr>";
            for (var i = 0; i < examples.length; i++) {{
              var ex = examples[i];
              var q = escapeHtml(ex[0] || "(none)");
              var n = escapeHtml(ex[1] || "(none)");
              var s = escapeHtml((typeof ex[2] === "number") ? ex[2].toFixed(3) : ex[2]);
              examplesHtml += "<tr>" +
                "<td style='border:1px solid #ccc; padding:2px 4px;'>" + (i + 1) + "</td>" +
                "<td style='border:1px solid #ccc; padding:2px 4px;'><code>" + q + "</code></td>" +
                "<td style='border:1px solid #ccc; padding:2px 4px;'><code>" + n + "</code></td>" +
                "<td style='border:1px solid #ccc; padding:2px 4px;'>" + s + "</td>" +
                "</tr>";
            }}
            examplesHtml += "</table>";
          }}

          infoDiv.innerHTML =
            "<strong>From:</strong> " + manuscriptSearchLink(fromId) + " - " +
              escapeHtml(nodeShelfmark(fromNode)) + ", " + escapeHtml(fromNode.library || "Unknown library") + "<br/>" +
            "<strong>To:</strong> " + manuscriptSearchLink(toId) + " - " +
              escapeHtml(nodeShelfmark(toNode)) + ", " + escapeHtml(toNode.library || "Unknown library") + "<br/>" +
            "<strong>Connections:</strong> " + escapeHtml(edge.weight) +
            " · <strong>Max similarity:</strong> " + escapeHtml(edge.similarity.toFixed(3)) +
            examplesHtml;
        }}
      }}
    }});

    // Focus on manuscript by id.
    function focusOnManuscript(msId) {{
      if (!msId) {{
        searchStatus.textContent = "Please enter an id.";
        return;
      }}
      var node = nodes.get(msId);
      if (!node) {{
        searchStatus.textContent = "Not found.";
        return;
      }}
      searchStatus.textContent = "";
      network.selectNodes([msId], false);
      network.focus(msId, {{
        scale: 1.2,
        animation: {{
          duration: 500,
          easingFunction: "easeInOutQuad"
        }}
      }});
    }}

    searchBtn.addEventListener("click", function() {{
      var msId = (searchInput.value || "").trim();
      focusOnManuscript(msId);
    }});

    searchInput.addEventListener("keyup", function(e) {{
      if (e.key === "Enter") {{
        var msId = (searchInput.value || "").trim();
        focusOnManuscript(msId);
      }}
    }});
  </script>
</body>
</html>
"""


def export_manuscript_graph_interactive(
    G: nx.Graph,
    html_path: str,
    height: str = "600px",
    width: str = "100%",
) -> None:
    """
    Export an interactive HTML visualization of the manuscript graph using vis-network.
    Nodes are positioned once with a Python-side *community-based* layout (same idea
    as the image ego graph: each community on its own circle, laid out along X),
    then fixed in the browser. Sliders filter edges by weight and similarity in real time.
    """
    if len(G) == 0:
        print("Manuscript graph is empty; nothing to export.")
        return

    print("  [HTML] Detecting communities for manuscript graph...")
    node_to_comm = detect_communities(G)
    n_comms = len(set(node_to_comm.values())) if node_to_comm else 0
    print(f"  [HTML] Detected {n_comms} communities.")

    print(
        f"  [HTML] Computing community-based layout for manuscript graph "
        f"({len(G.nodes())} nodes, {len(G.edges())} edges)..."
    )
    pos = community_layout(G, node_to_comm=node_to_comm)
    print("  [HTML] Layout computation done.")

    print("  [HTML] Building node list for serialization...")
    nodes = []
    for n, data in G.nodes(data=True):
        n_str = str(n)
        total = data.get("total_images")
        library = str(data.get("library") or "")
        shelfmark = str(data.get("shelfmark") or "")
        shelfmark_running_index = str(
            data.get("shelfmark_running_index") or ""
        )
        shelfmark_label = _format_indexed_shelfmark(
            shelfmark, shelfmark_running_index
        )
        comm = int(node_to_comm.get(n_str, 0))
        title_lines = [f"Manuscript: {n_str}"]
        if shelfmark_label:
            title_lines.append(f"Shelfmark: {shelfmark_label}")
        if library:
            title_lines.append(f"Library: {library}")
        label_parts = [n_str]
        if shelfmark_label:
            label_parts.append(shelfmark_label)
        if library:
            label_parts.append(library)
        nodes.append(
            {
                "id": n_str,
                "label": "\n".join(label_parts),
                "title": "<br>".join(title_lines),
                "group": comm,
                "library": library,
                "shelfmark": shelfmark,
                "shelfmarkRunningIndex": shelfmark_running_index,
                "shelfmarkLabel": shelfmark_label,
                "totalImages": total,
                # Scale positions to spread communities nicely on screen.
                "x": float(pos[n][0]) * 6.0,
                "y": float(pos[n][1]) * 6.0,
                "fixed": True,
                "physics": False,
            }
        )

    print("  [HTML] Building edge list for serialization (with labels and similarity)...")
    edges = []
    weights: List[int] = []
    sims: List[float] = []
    for u, v, data in G.edges(data=True):
        w = int(data.get("weight", 1))
        weights.append(w)
        max_sim = float(data.get("max_similarity", 0.0))
        sims.append(max_sim)
        best_q_img = data.get("best_q_image")
        best_n_img = data.get("best_n_image")
        best_pair_sim = float(data.get("best_similarity", max_sim))
        examples = data.get("examples") or []

        # Edge tooltip includes manuscripts and, if available, the representative image pair.
        title_parts = [
            f"{w} image connections; max similarity={max_sim:.3f}",
        ]
        if best_q_img or best_n_img:
            title_parts.append(
                f"Best example: {best_q_img or '∅'} → {best_n_img or '∅'} (sim={best_pair_sim:.3f})"
            )

        edges.append(
            {
                "from": str(u),
                "to": str(v),
                "value": w,
                "weight": w,
                "title": "<br>".join(title_parts),
                "similarity": max_sim,
                "label": str(w),
                "bestQImage": best_q_img,
                "bestNImage": best_n_img,
                "bestPairSim": best_pair_sim,
                "examples": examples,
                "fromLibrary": str(G.nodes[u].get("library") or ""),
                "toLibrary": str(G.nodes[v].get("library") or ""),
            }
        )

    if not weights:
        min_w = max_w = 1
    else:
        min_w = min(weights)
        max_w = max(weights)

    if not sims:
        min_s = 0.0
        max_s = 1.0
    else:
        min_s = float(min(sims))
        max_s = float(max(sims))
        if max_s == min_s:
            max_s = min_s + 0.01

    print("  [HTML] Serializing nodes and edges to JSON...")
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    html = _build_manuscript_graph_html(
        nodes_json=nodes_json,
        edges_json=edges_json,
        min_w=min_w,
        max_w=max_w,
        min_s=min_s,
        max_s=max_s,
        width=width,
        height=height,
    )

    print(f"  [HTML] Writing HTML to {html_path} ...")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [HTML] Done. Interactive manuscript graph with slider exported to {html_path}")


def _build_image_ego_graph_html(
    nodes_json: str,
    edges_json: str,
    min_s: float,
    max_s: float,
    page_title: str,
    width: str,
    height: str,
    data_base: str | None = None,
) -> str:
    """Build the full HTML for the image ego graph (same style as manuscript graph).
    If data_base is set, loads nodes/edges from external JSON files for faster initial load."""
    style_content = _graph_page_css(width, height)
    data_base_js = json.dumps(data_base) if data_base else "null"
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{page_title}</title>
  <style type="text/css">
{style_content}
  </style>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
  <div class="page-header">
    <h3>{page_title}</h3>
    <div class="subtitle">
      Green edges share a manuscript; red edges connect different manuscripts.
    </div>
  </div>
  <div id="controls">
    <div class="control-group">
      <label for="searchImageId">Jump to image:</label>
      <input type="text" id="searchImageId" style="width: 200px;" placeholder="e.g. IE52850139_P000002_FL52850143.jpg">
      <button id="searchImageBtn">Go</button>
      <span id="searchImageStatus" class="search-status"></span>
    </div>
    <div class="control-group">
      <label for="minSim">Min similarity:</label>
      <input type="range" id="minSim" name="minSim" min="{min_s:.3f}" max="{max_s:.3f}" value="{min_s:.3f}" step="0.01">
      <input type="number" id="minSimInput" min="{min_s:.3f}" max="{max_s:.3f}" value="{min_s:.3f}" step="0.01" style="width: 80px;">
      <span id="minSimLabel">{min_s:.3f}</span>
    </div>
    <div class="control-group library-filter-group">
      <label>From libraries:</label>
      <details id="libraryDropdown" class="library-dropdown">
        <summary><span id="librarySummary">All libraries</span></summary>
        <div id="libraryFilters" class="library-checkboxes">
          <label class="checkbox-option"><input type="checkbox" id="allLibraries" checked> <span id="allLibrariesLabel">All libraries</span></label>
        </div>
      </details>
    </div>
    <div class="control-group library-target-group">
      <label>To libraries:</label>
      <details id="connectedLibraryDropdown" class="library-dropdown">
        <summary><span id="connectedLibrarySummary">All libraries</span></summary>
        <div id="connectedLibraryFilters" class="library-checkboxes">
          <label class="checkbox-option"><input type="checkbox" id="allConnectedLibraries" checked> <span id="allConnectedLibrariesLabel">All libraries</span></label>
        </div>
      </details>
      <button type="button" id="findLibraryJoins" aria-pressed="false" title="Toggle inter-library joins mode. It keeps the current library and similarity filters, and limits their results to connections between different libraries.">Find joins</button>
      <span class="focus-result-controls">
        <button type="button" id="focusLibraryResult" disabled title="Center the next image from the explicitly selected libraries without zooming out.">Focus result</button>
        <span id="focusResultStatus" class="focus-result-status">0 / 0</span>
      </span>
      <span id="libraryConnectionCount" class="library-connection-count"></span>
      <span id="findJoinsStatus" class="find-joins-status"></span>
    </div>
    <div class="control-group">
      <label for="bridgeMode">
        <input type="checkbox" id="bridgeMode" name="bridgeMode">
        Bridge nodes only
      </label>
      <span class="control-hint" title="Show only nodes with both green (same ms) and red (different ms) edges">(both green & red)</span>
    </div>
    <div class="control-group">
      <label for="clusterMode">
        <input type="checkbox" id="clusterMode" name="clusterMode">
        Cluster by manuscript
      </label>
      <span class="control-hint" title="Group nodes by manuscript for faster rendering">(fewer nodes)</span>
    </div>
  </div>
  <div id="networkContainer">
    <div id="loadingOverlay"><div class="loading-spinner"></div><div class="loading-text">Loading graph data...</div></div>
    <div id="network"></div>
    <div id="nodeInfo">Click an image node or edge to see details here.</div>
  </div>
  <script type="text/javascript">
    var DATA_BASE = {data_base_js};
    var INLINE_NODES = {nodes_json};
    var INLINE_EDGES = {edges_json};

    function initNetwork(nodesData, edgesData) {{
      var nodes = new vis.DataSet(nodesData);
      var allEdges = edgesData;
      var edgesDS = new vis.DataSet(allEdges);

      var container = document.getElementById('network');
      var data = {{
        nodes: nodes,
        edges: edgesDS
      }};
      var options = {{
        physics: false,
        interaction: {{
          hideEdgesOnDrag: true,
          hideEdgesOnZoom: true,
          hover: true
        }},
        nodes: {{
          shape: 'dot',
          size: nodesData.length > 5000 ? 11 : 18,
          font: {{
            size: nodesData.length > 5000 ? 10 : 13,
            color: '#000000',
            face: 'sans-serif',
            bold: {{ size: nodesData.length > 5000 ? 10 : 13 }}
          }},
          borderWidth: 3,
          borderWidthSelected: 6
        }},
        edges: {{
          scaling: {{ min: 3, max: 7 }},
          smooth: false,
          hoverWidth: 4,
          selectionWidth: 5,
          font: {{ size: 13, strokeWidth: 4 }}
        }}
      }};

      var network = new vis.Network(container, data, options);

      var slider = document.getElementById('minSim');
      var simInput = document.getElementById('minSimInput');
      var simLabel = document.getElementById('minSimLabel');
      var libraryFilters = document.getElementById('libraryFilters');
      var libraryDropdown = document.getElementById('libraryDropdown');
      var allLibrariesCheckbox = document.getElementById('allLibraries');
      var allLibrariesLabel = document.getElementById('allLibrariesLabel');
      var librarySummary = document.getElementById('librarySummary');
      var connectedLibraryFilters = document.getElementById('connectedLibraryFilters');
      var connectedLibraryDropdown = document.getElementById('connectedLibraryDropdown');
      var allConnectedLibrariesCheckbox = document.getElementById('allConnectedLibraries');
      var allConnectedLibrariesLabel = document.getElementById('allConnectedLibrariesLabel');
      var connectedLibrarySummary = document.getElementById('connectedLibrarySummary');
      var findLibraryJoins = document.getElementById('findLibraryJoins');
      var libraryConnectionCount = document.getElementById('libraryConnectionCount');
      var findJoinsStatus = document.getElementById('findJoinsStatus');
      var focusLibraryResult = document.getElementById('focusLibraryResult');
      var focusResultStatus = document.getElementById('focusResultStatus');
      var bridgeModeCheckbox = document.getElementById('bridgeMode');
      var clusterCheckbox = document.getElementById('clusterMode');
      var infoDiv = document.getElementById('nodeInfo');
      var searchInput = document.getElementById('searchImageId');
      var searchBtn = document.getElementById('searchImageBtn');
      var searchStatus = document.getElementById('searchImageStatus');
      var interLibraryJoinsMode = false;
      var focusCandidateIds = [];
      var focusCandidateIndex = -1;
      var focusRequestId = 0;

      function nodeShelfmark(node) {{
        return node.shelfmarkLabel || node.shelfmark || 'Unknown shelfmark';
      }}

      var libraryNodeCounts = {{}};
      nodes.get().forEach(function(node) {{
        var library = node.library || "";
        if (library) libraryNodeCounts[library] = (libraryNodeCounts[library] || 0) + 1;
      }});
      function formatNodeCount(count) {{
        return count + (count === 1 ? ' node' : ' nodes');
      }}
      allLibrariesLabel.textContent =
        'All libraries (' + formatNodeCount(nodes.getIds().length) + ')';
      allConnectedLibrariesLabel.textContent =
        'All libraries (' + formatNodeCount(nodes.getIds().length) + ')';

      Object.keys(libraryNodeCounts)
        .sort(function(a, b) {{ return a.localeCompare(b); }})
        .forEach(function(library) {{
          var label = document.createElement('label');
          label.className = 'checkbox-option';
          var checkbox = document.createElement('input');
          checkbox.type = 'checkbox';
          checkbox.className = 'library-checkbox';
          checkbox.value = library;
          checkbox.checked = true;
          label.appendChild(checkbox);
          label.appendChild(document.createTextNode(
            ' ' + library + ' (' + formatNodeCount(libraryNodeCounts[library]) + ')'
          ));
          libraryFilters.appendChild(label);

          var targetLabel = label.cloneNode(true);
          var targetCheckbox = targetLabel.querySelector('input');
          targetCheckbox.className = 'connected-library-checkbox';
          targetCheckbox.checked = true;
          connectedLibraryFilters.appendChild(targetLabel);
        }});

      function getSelectedLibraries(filters, allCheckbox, checkboxClass) {{
        if (allCheckbox.checked) return null;
        return new Set(
          Array.from(filters.querySelectorAll('.' + checkboxClass + ':checked'))
            .map(function(checkbox) {{ return checkbox.value; }})
        );
      }}

      function updateDropdownSummary(filters, allCheckbox, checkboxClass, summaryElement) {{
        var checked = Array.from(
          filters.querySelectorAll('.' + checkboxClass + ':checked')
        );
        var summary;
        if (allCheckbox.checked) {{
          summary = 'All libraries';
        }} else if (checked.length === 0) {{
          summary = 'No libraries';
        }} else if (checked.length === 1) {{
          summary = checked[0].value;
        }} else {{
          summary = checked.length + ' libraries';
        }}
        summaryElement.textContent = summary;
      }}

      function updateLibrarySummaries() {{
        updateDropdownSummary(
          libraryFilters, allLibrariesCheckbox, 'library-checkbox', librarySummary
        );
        updateDropdownSummary(
          connectedLibraryFilters, allConnectedLibrariesCheckbox,
          'connected-library-checkbox', connectedLibrarySummary
        );
      }}

      document.addEventListener('pointerdown', function(event) {{
        if (libraryDropdown.open && !libraryDropdown.contains(event.target)) {{
          libraryDropdown.open = false;
        }}
        if (connectedLibraryDropdown.open &&
            !connectedLibraryDropdown.contains(event.target)) {{
          connectedLibraryDropdown.open = false;
        }}
      }});
      document.addEventListener('keydown', function(event) {{
        if (event.key === 'Escape') {{
          libraryDropdown.open = false;
          connectedLibraryDropdown.open = false;
        }}
      }});

      function getBridgeNodes(edges) {{
        var nodeHasGreen = {{}};
        var nodeHasRed = {{}};
        for (var i = 0; i < edges.length; i++) {{
          var e = edges[i];
          if (e.sameManuscript) {{
            nodeHasGreen[e.from] = true;
            nodeHasGreen[e.to] = true;
          }} else {{
            nodeHasRed[e.from] = true;
            nodeHasRed[e.to] = true;
          }}
        }}
        var bridge = {{}};
        for (var n in nodeHasGreen) {{
          if (nodeHasRed[n]) bridge[n] = true;
        }}
        return bridge;
      }}

      function updateEdges() {{
        focusRequestId += 1;
        var minS = parseFloat(slider.value);
        if (isNaN(minS)) minS = {min_s:.3f};
        simLabel.textContent = minS.toFixed(3);
        simInput.value = minS.toFixed(3);
        var selectedLibraries = getSelectedLibraries(
          libraryFilters, allLibrariesCheckbox, 'library-checkbox'
        );
        var targetLibraries = getSelectedLibraries(
          connectedLibraryFilters, allConnectedLibrariesCheckbox,
          'connected-library-checkbox'
        );
        var libraryFiltered = interLibraryJoinsMode ||
          selectedLibraries !== null || targetLibraries !== null;
        var filtered = allEdges.filter(function(e) {{
          var libraryOk;
          var sourceFromOk = selectedLibraries === null ||
            selectedLibraries.has(e.fromLibrary);
          var sourceToOk = selectedLibraries === null ||
            selectedLibraries.has(e.toLibrary);
          var targetFromOk = targetLibraries === null ||
            targetLibraries.has(e.fromLibrary);
          var targetToOk = targetLibraries === null ||
            targetLibraries.has(e.toLibrary);
          libraryOk = (sourceFromOk && targetToOk) || (sourceToOk && targetFromOk);
          if (interLibraryJoinsMode) {{
            libraryOk = libraryOk && !!e.fromLibrary && !!e.toLibrary &&
              e.fromLibrary !== e.toLibrary;
          }}
          return e.similarity >= minS && libraryOk;
        }});
        var useBridge = bridgeModeCheckbox && bridgeModeCheckbox.checked;
        var bridgeNodes = useBridge ? getBridgeNodes(filtered) : {{}};
        if (useBridge) {{
          filtered = filtered.filter(function(e) {{ return bridgeNodes[e.from] && bridgeNodes[e.to]; }});
        }}
        edgesDS.clear();
        edgesDS.add(filtered);
        libraryConnectionCount.textContent =
          filtered.length + (filtered.length === 1 ? ' connection' : ' connections');
        if (interLibraryJoinsMode) {{
          findJoinsStatus.textContent = filtered.length +
            (filtered.length === 1 ? ' inter-library join shown.' : ' inter-library joins shown.');
        }}
        var visibleNodes = {{}};
        if (libraryFiltered || useBridge) {{
          filtered.forEach(function(e) {{
            visibleNodes[e.from] = true;
            visibleNodes[e.to] = true;
          }});
          if (!interLibraryJoinsMode && selectedLibraries !== null &&
              targetLibraries === null && !useBridge) {{
            nodes.get().forEach(function(n) {{
              if (selectedLibraries.has(n.library)) visibleNodes[n.id] = true;
            }});
          }}
        }}
        var explicitLibraries = new Set();
        if (selectedLibraries !== null) {{
          selectedLibraries.forEach(function(library) {{ explicitLibraries.add(library); }});
        }}
        if (targetLibraries !== null) {{
          targetLibraries.forEach(function(library) {{ explicitLibraries.add(library); }});
        }}
        var previousFocusedId = focusCandidateIndex >= 0
          ? focusCandidateIds[focusCandidateIndex]
          : null;
        focusCandidateIds = nodes.get().filter(function(node) {{
          var isVisible = !(libraryFiltered || useBridge) || !!visibleNodes[node.id];
          return isVisible && explicitLibraries.has(node.library);
        }}).sort(function(a, b) {{
          var aKey = [a.library || '', a.shelfmark || '',
            String(a.ms_id || ''), String(a.img_id || a.id)].join('\\u0000');
          var bKey = [b.library || '', b.shelfmark || '',
            String(b.ms_id || ''), String(b.img_id || b.id)].join('\\u0000');
          return aKey.localeCompare(bKey);
        }}).map(function(node) {{ return node.id; }});
        focusCandidateIndex = previousFocusedId === null
          ? -1
          : focusCandidateIds.indexOf(previousFocusedId);
        var focusCandidateSet = new Set(
          focusCandidateIds.map(function(id) {{ return String(id); }})
        );
        var allNodeIds = nodes.getIds();
        var updates = [];
        for (var i = 0; i < allNodeIds.length; i++) {{
          var id = allNodeIds[i];
          var highlighted = focusCandidateSet.has(String(id));
          updates.push({{
            id: id,
            hidden: (libraryFiltered || useBridge) ? !visibleNodes[id] : false,
            borderWidth: highlighted ? 6 : 3,
            shadow: highlighted
              ? {{ enabled: true, color: 'rgba(139, 94, 52, 0.65)', size: 18, x: 0, y: 0 }}
              : false
          }});
        }}
        if (updates.length > 0) nodes.update(updates);
        focusLibraryResult.disabled = focusCandidateIds.length === 0;
        focusResultStatus.textContent = focusCandidateIds.length === 0
          ? '0 / 0'
          : ((focusCandidateIndex >= 0 ? focusCandidateIndex + 1 : 0) +
             ' / ' + focusCandidateIds.length);
        if ((libraryFiltered || useBridge) && Object.keys(visibleNodes).length === 0) {{
          infoDiv.textContent = 'No nodes match the current filters.';
        }} else if (infoDiv.textContent === 'No nodes match the current filters.') {{
          infoDiv.textContent = 'Click an image node or edge to see details here.';
        }}
      }}

      function getExplicitLibraryCount() {{
        var selectedLibraries = getSelectedLibraries(
          libraryFilters, allLibrariesCheckbox, 'library-checkbox'
        );
        var targetLibraries = getSelectedLibraries(
          connectedLibraryFilters, allConnectedLibrariesCheckbox,
          'connected-library-checkbox'
        );
        var libraries = new Set();
        if (selectedLibraries !== null) {{
          selectedLibraries.forEach(function(library) {{ libraries.add(library); }});
        }}
        if (targetLibraries !== null) {{
          targetLibraries.forEach(function(library) {{ libraries.add(library); }});
        }}
        return libraries.size;
      }}

      function focusCandidate(index) {{
        if (focusCandidateIds.length === 0) return;
        if (clusterCheckbox && clusterCheckbox.checked) {{
          clusterCheckbox.checked = false;
          network.clustering.unclusterAll();
        }}
        focusCandidateIndex = ((index % focusCandidateIds.length) +
          focusCandidateIds.length) % focusCandidateIds.length;
        var nodeId = focusCandidateIds[focusCandidateIndex];
        var requestId = ++focusRequestId;
        var currentScale = network.getScale();
        network.redraw();
        focusResultStatus.textContent = (focusCandidateIndex + 1) +
          ' / ' + focusCandidateIds.length;
        window.requestAnimationFrame(function() {{
          if (requestId !== focusRequestId) return;
          network.redraw();
          window.requestAnimationFrame(function() {{
            if (requestId !== focusRequestId) return;
            network.focus(nodeId, {{
              scale: Math.max(currentScale, 0.85),
              animation: {{ duration: 350, easingFunction: 'easeInOutQuad' }}
            }});
          }});
        }});
      }}

      function applyCluster() {{
        if (clusterCheckbox && clusterCheckbox.checked) {{
          try {{
            network.clustering.clusterByGroup('group');
          }} catch (e) {{ clusterCheckbox.checked = false; }}
        }} else {{
          network.clustering.unclusterAll();
        }}
      }}

      slider.addEventListener('input', updateEdges);
      function handleLibraryCheckboxChange(event, filters, allCheckbox, checkboxClass) {{
        if (!event.target.matches('input[type="checkbox"]')) return;
        var libraryCheckboxes = Array.from(
          filters.querySelectorAll('.' + checkboxClass)
        );
        if (event.target === allCheckbox) {{
          libraryCheckboxes.forEach(function(checkbox) {{
            checkbox.checked = allCheckbox.checked;
          }});
          allCheckbox.indeterminate = false;
        }} else {{
          var checkedCount = libraryCheckboxes.filter(function(checkbox) {{
            return checkbox.checked;
          }}).length;
          allCheckbox.checked = checkedCount === libraryCheckboxes.length;
          allCheckbox.indeterminate = checkedCount > 0 &&
            checkedCount < libraryCheckboxes.length;
        }}
        updateLibrarySummaries();
        if (clusterCheckbox && clusterCheckbox.checked) {{
          clusterCheckbox.checked = false;
          network.clustering.unclusterAll();
        }}
        updateEdges();
        if (getExplicitLibraryCount() === 1 && focusCandidateIds.length > 0) {{
          focusCandidate(0);
        }}
      }}
      libraryFilters.addEventListener('change', function(event) {{
        handleLibraryCheckboxChange(
          event, libraryFilters, allLibrariesCheckbox, 'library-checkbox'
        );
      }});
      connectedLibraryFilters.addEventListener('change', function(event) {{
        handleLibraryCheckboxChange(
          event, connectedLibraryFilters, allConnectedLibrariesCheckbox,
          'connected-library-checkbox'
        );
      }});
      findLibraryJoins.addEventListener('click', function() {{
        interLibraryJoinsMode = !interLibraryJoinsMode;
        findLibraryJoins.setAttribute(
          'aria-pressed', interLibraryJoinsMode ? 'true' : 'false'
        );
        findJoinsStatus.textContent = '';
        updateEdges();
      }});
      focusLibraryResult.addEventListener('click', function() {{
        focusCandidate(focusCandidateIndex + 1);
      }});
      if (bridgeModeCheckbox) bridgeModeCheckbox.addEventListener('change', updateEdges);
      if (clusterCheckbox) clusterCheckbox.addEventListener('change', applyCluster);
      simInput.addEventListener('change', function() {{
        var v = parseFloat(simInput.value);
        if (isNaN(v)) v = {min_s:.3f};
        v = Math.max({min_s:.3f}, Math.min({max_s:.3f}, v));
        simInput.value = v.toFixed(3);
        slider.value = v;
        updateEdges();
      }});
      updateEdges();
      if (clusterCheckbox && clusterCheckbox.checked) applyCluster();

      network.on("click", function (params) {{
        if (params.edges && params.edges.length > 0) {{
          var edgeId = params.edges[0];
          var edge = edgesDS.get(edgeId);
          if (edge) {{
            var fromId = edge.from, toId = edge.to;
            var fromNode = nodes.get(fromId) || {{}}, toNode = nodes.get(toId) || {{}};
            var fromMs = fromNode.ms_id || "(none)", toMs = toNode.ms_id || "(none)";
            var fromImg = fromNode.img_id || fromId, toImg = toNode.img_id || toId;
            var sim = (typeof edge.similarity === "number") ? edge.similarity.toFixed(3) : "N/A";
            var sameFlag = (typeof edge.sameManuscript === "boolean") ? edge.sameManuscript : (fromMs !== "(none)" && toMs !== "(none)" && fromMs === toMs);
            infoDiv.innerHTML =
              "<strong>From:</strong> " + fromMs + " — " + nodeShelfmark(fromNode) + ", " + (fromNode.library || "Unknown library") + "<br/><code>" + fromImg + "</code><br/>" +
              "<strong>To:</strong> " + toMs + " — " + nodeShelfmark(toNode) + ", " + (toNode.library || "Unknown library") + "<br/><code>" + toImg + "</code><br/>" +
              "<strong>Similarity:</strong> " + sim + " · <strong>Same manuscript:</strong> " + (sameFlag ? "Yes" : "No");
          }}
          return;
        }}
        if (params.nodes && params.nodes.length > 0) {{
          var nodeId = params.nodes[0];
          var node = nodes.get(nodeId);
          if (node) {{
            var ms = node.ms_id || "(none)", img = node.img_id || node.label;
            infoDiv.innerHTML =
              "<strong>Manuscript:</strong> " + ms + "<br/>" +
              "<strong>Shelfmark:</strong> " + nodeShelfmark(node) + "<br/>" +
              "<strong>Library:</strong> " + (node.library || "Unknown") + "<br/>" +
              "<strong>Image:</strong> <code>" + img + "</code>";
          }}
        }}
      }});

      function focusOnImage(imageId) {{
        if (!imageId) {{ searchStatus.textContent = "Enter id."; return; }}
        var node = nodes.get(imageId);
        if (!node) {{ searchStatus.textContent = "Not found."; return; }}
        searchStatus.textContent = "";
        network.selectNodes([imageId], false);
        network.focus(imageId, {{ scale: 1.2, animation: {{ duration: 500, easingFunction: "easeInOutQuad" }} }});
      }}
      searchBtn.addEventListener("click", function() {{ focusOnImage((searchInput.value || "").trim()); }});
      searchInput.addEventListener("keyup", function(e) {{ if (e.key === "Enter") focusOnImage((searchInput.value || "").trim()); }});
    }}

    (function loadAndInit() {{
      var overlay = document.getElementById('loadingOverlay');
      function hideLoading() {{ if (overlay) overlay.style.display = 'none'; }}
      function showError(msg) {{ if (overlay) {{ overlay.querySelector('.loading-text').textContent = msg; overlay.style.background = '#fee'; }} }}

      if (DATA_BASE) {{
        var base = DATA_BASE;
        var dir = location.pathname.replace(/[^/]+$/, '');
        var nodesUrl = dir + base + '_nodes.json';
        var edgesUrl = dir + base + '_edges.json';
        Promise.all([
          fetch(nodesUrl).then(function(r) {{ return r.ok ? r.json() : Promise.reject('nodes'); }}),
          fetch(edgesUrl).then(function(r) {{ return r.ok ? r.json() : Promise.reject('edges'); }})
        ]).then(function(arr) {{
          initNetwork(arr[0], arr[1]);
          hideLoading();
        }}).catch(function() {{
          showError('External data failed. Using embedded data.');
          initNetwork(INLINE_NODES, INLINE_EDGES);
          hideLoading();
        }});
      }} else {{
        initNetwork(INLINE_NODES, INLINE_EDGES);
        hideLoading();
      }}
    }})();
  </script>
</body>
</html>
"""


def export_image_ego_interactive(
    G: nx.Graph,
    center_image_id: str,
    html_path: str,
    height: str = "600px",
    width: str = "100%",
) -> None:
    """
    Export an interactive HTML visualization of an image ego-graph using vis-network.
    - Nodes: images around a chosen center image.
    - Edges: colored by same/different manuscript.
    - Slider and number input filter edges by similarity_score in real time.
    Uses the same page style as the manuscript graph (shared CSS, controls card, side-by-side panel).
    """
    if len(G) == 0:
        print("Image ego graph is empty; nothing to export.")
        return

    if center_image_id not in G:
        print(f"Center image {center_image_id!r} not found in ego graph; exporting anyway.")

    print(
        f"  [HTML-image] Preparing interactive ego graph for {center_image_id!r} "
        f"({len(G.nodes())} nodes, {len(G.edges())} edges)..."
    )

    # Community-aware static layout: separate clusters on a grid.
    print("  [HTML-image] Detecting communities for ego graph...")
    node_to_comm = detect_communities(G)
    n_comms = len(set(node_to_comm.values())) if node_to_comm else 0
    print(f"  [HTML-image] Detected {n_comms} communities.")

    print("  [HTML-image] Computing community-based layout for ego graph...")
    pos = community_layout(G, node_to_comm=node_to_comm)
    print("  [HTML-image] Layout computation done.")

    nodes = []
    for n, data in G.nodes(data=True):
        ms_id = data.get("manuscript_id")
        img_id = data.get("image_id", n)
        library = str(data.get("library") or "")
        shelfmark = str(data.get("shelfmark") or "")
        shelfmark_running_index = str(
            data.get("shelfmark_running_index") or ""
        )
        shelfmark_label = _format_indexed_shelfmark(
            shelfmark, shelfmark_running_index
        )
        title_lines = [f"Image: {img_id}"]
        if ms_id is not None:
            title_lines.append(f"Manuscript: {ms_id}")
        if shelfmark_label:
            title_lines.append(f"Shelfmark: {shelfmark_label}")
        if library:
            title_lines.append(f"Library: {library}")
        if n == center_image_id:
            title_lines.append("CENTER IMAGE")
        # Community id for layout / tooltip
        comm = int(node_to_comm.get(str(n), 0)) if node_to_comm else 0
        # Use manuscript id as group for coloring (different manuscripts → different colors)
        group = str(ms_id) if ms_id is not None else f"comm-{comm}"
        manuscript_label = str(ms_id) if ms_id is not None else "Unknown manuscript"
        if shelfmark_label:
            manuscript_label += f" | {shelfmark_label}"
        if library:
            manuscript_label += f" | {library}"
        nodes.append(
            {
                "id": str(n),
                "label": f"{manuscript_label}\n{img_id}",
                "title": "<br>".join(title_lines),
                "group": group,
                "x": float(pos[n][0]) * 6.0,
                "y": float(pos[n][1]) * 6.0,
                "ms_id": ms_id,
                "img_id": img_id,
                "community": comm,
                "library": library,
                "shelfmark": shelfmark,
                "shelfmarkRunningIndex": shelfmark_running_index,
                "shelfmarkLabel": shelfmark_label,
            }
        )

    edges = []
    sims: List[float] = []
    for u, v, data in G.edges(data=True):
        sim = float(data.get("similarity_score", 0.0))
        sims.append(sim)
        # Define same manuscript strictly by manuscript_id (9900... number)
        q_ms = G.nodes[u].get("manuscript_id")
        n_ms = G.nodes[v].get("manuscript_id")
        same_ms = (
            q_ms is not None
            and n_ms is not None
            and str(q_ms).strip() == str(n_ms).strip()
        )
        color = "#2ca02c" if same_ms else "#d62728"
        edges.append(
            {
                "from": str(u),
                "to": str(v),
                "value": sim,
                "similarity": sim,
                "color": color,
                "label": f"{sim:.2f}",
                "title": f"similarity={sim:.3f}",
                "sameManuscript": same_ms,
                "fromLibrary": str(G.nodes[u].get("library") or ""),
                "toLibrary": str(G.nodes[v].get("library") or ""),
            }
        )

    if not sims:
        min_s = 0.0
        max_s = 1.0
    else:
        min_s = float(min(sims))
        max_s = float(max(sims))
        if max_s == min_s:
            max_s = min_s + 0.01

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    if center_image_id:
        center_ms = G.nodes[center_image_id].get("manuscript_id") if center_image_id in G else None
        page_title = f"Image ego graph around {center_image_id} (ms={center_ms})" if center_ms is not None else f"Image ego graph around {center_image_id}"
    else:
        page_title = "Image connectivity graph (all images)"

    # For large graphs, write external JSON for faster loading (avoids huge inline HTML)
    data_base: str | None = None
    n_nodes, n_edges = len(nodes), len(edges)
    if n_nodes > 3000 or n_edges > 15000:
        html_dir = os.path.dirname(html_path)
        html_basename = os.path.splitext(os.path.basename(html_path))[0]
        data_base = html_basename
        nodes_path = os.path.join(html_dir, f"{html_basename}_nodes.json")
        edges_path = os.path.join(html_dir, f"{html_basename}_edges.json")
        print(f"  [HTML-image] Writing external data ({n_nodes:,} nodes, {n_edges:,} edges)...")
        with open(nodes_path, "w", encoding="utf-8") as f:
            f.write(nodes_json)
        with open(edges_path, "w", encoding="utf-8") as f:
            f.write(edges_json)
        print(f"  [HTML-image] Wrote {nodes_path} and {edges_path}")

    html = _build_image_ego_graph_html(
        nodes_json=nodes_json,
        edges_json=edges_json,
        min_s=min_s,
        max_s=max_s,
        page_title=page_title,
        width=width,
        height=height,
        data_base=data_base,
    )

    print(f"  [HTML-image] Writing image ego HTML to {html_path} ...")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [HTML-image] Done. Interactive image ego graph exported to {html_path}")


def make_manuscript_ego_subgraph(
    G: nx.Graph,
    center_ms_id: str,
    radius: int = 1,
    max_nodes: int = 2000,
) -> nx.Graph:
    """
    Extract an ego-subgraph around one manuscript from the global manuscript graph.

    - Nodes: manuscripts within <= `radius` hops of `center_ms_id`.
    - Edges: same as in the global manuscript graph; edge weight is the number of
      image pairs connecting the two manuscripts (already stored as "weight").
    """
    if center_ms_id not in G:
        raise ValueError(f"Manuscript id {center_ms_id!r} not found in manuscript graph.")

    nodes_radius = nx.single_source_shortest_path_length(G, center_ms_id, cutoff=radius)
    nodes = list(nodes_radius.keys())

    if len(nodes) > max_nodes:
        # Keep closest nodes first (by hop distance, then degree)
        nodes_by_dist: Dict[int, List[str]] = defaultdict(list)
        for n, d in nodes_radius.items():
            nodes_by_dist[d].append(n)
        selected: List[str] = []
        for dist in sorted(nodes_by_dist.keys()):
            chunk = nodes_by_dist[dist]
            # sort chunk by degree, descending
            chunk_sorted = sorted(chunk, key=lambda n: G.degree[n], reverse=True)
            for n in chunk_sorted:
                if len(selected) >= max_nodes:
                    break
                selected.append(n)
            if len(selected) >= max_nodes:
                break
        nodes = selected

    H = G.subgraph(nodes).copy()
    return H


def make_image_ego_subgraph(
    G: nx.Graph,
    center_image_id: str,
    radius: int = 1,
    max_nodes: int = 1000,
) -> nx.Graph:
    """
    Extract a manageable ego-subgraph around one image from the global image graph.
    This is what you should visualize interactively or with matplotlib.
    """
    if center_image_id not in G:
        raise ValueError(f"Image id {center_image_id!r} not found in global image graph.")

    nodes_radius = nx.single_source_shortest_path_length(G, center_image_id, cutoff=radius)
    nodes = list(nodes_radius.keys())

    if len(nodes) > max_nodes:
        # Keep closest nodes first (by hop distance), then by similarity to center.
        # Direct neighbors (dist=1): sort by edge similarity to center (highest first).
        # Multi-hop: sort by max similarity to any node closer to center.
        nodes_by_dist = defaultdict(list)
        for n, d in nodes_radius.items():
            nodes_by_dist[d].append(n)

        def _relevance_to_center(node: str, dist: int) -> float:
            """Higher = more relevant (prioritize high-similarity neighbors)."""
            if dist == 1:
                # Direct neighbor: use edge similarity to center
                return float(G[node][center_image_id].get("similarity_score", 0.0))
            # Multi-hop: max similarity to any node at distance dist-1
            best = 0.0
            for neighbor in G.neighbors(node):
                if nodes_radius.get(neighbor, dist) < dist:
                    best = max(best, float(G[node][neighbor].get("similarity_score", 0.0)))
            return best

        selected: List[str] = []
        for dist in sorted(nodes_by_dist.keys()):
            chunk = nodes_by_dist[dist]
            # Sort by relevance (similarity to center), then by degree as tiebreaker
            chunk_sorted = sorted(
                chunk,
                key=lambda n: (_relevance_to_center(n, dist), G.degree[n]),
                reverse=True,
            )
            for n in chunk_sorted:
                if len(selected) >= max_nodes:
                    break
                selected.append(n)
            if len(selected) >= max_nodes:
                break
        nodes = selected

    H = G.subgraph(nodes).copy()
    return H


def plot_image_graph(G: nx.Graph, title: str = "Image similarity graph (ego)") -> None:
    """Simple matplotlib plot for an image-level (ego) graph."""
    if len(G) == 0:
        print("Image graph is empty; nothing to plot.")
        return

    pos = nx.spring_layout(G, seed=42, k=None)

    nx.draw_networkx_nodes(G, pos, node_size=40, node_color="lightblue", alpha=0.8)

    edges = G.edges(data=True)
    same_edges = [(u, v) for u, v, d in edges if d.get("same_manuscript", False)]
    diff_edges = [(u, v) for u, v, d in G.edges(data=True) if not G[u][v].get("same_manuscript", False)]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=same_edges,
        edge_color="green",
        alpha=0.6,
        width=1.0,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=diff_edges,
        edge_color="red",
        alpha=0.4,
        width=0.8,
    )

    if len(G) <= 150:
        labels = {
            n: f"{G.nodes[n].get('manuscript_id', '')}\n{G.nodes[n].get('image_id', n)}"
            for n in G.nodes()
        }
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=5)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and explore Geniza graphs directly from the SQL KNN results table.",
    )
    parser.add_argument(
        "--db-config",
        type=str,
        default=CLUSTERING_DB_CONFIG_PATH,
        help="Path to DB config file (same as used for KNN generation).",
    )
    parser.add_argument(
        "--graph",
        type=str,
        choices=["manuscripts", "images", "both"],
        default="manuscripts",
        help="Which graph(s) to build. Use 'manuscripts' for the manuscript map.",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["sql", "excel", "overall_sql"],
        default="sql",
        help="Where to build image graphs from: 'sql' = geniza_knn_results (excludes same-manuscript), 'overall_sql' = geniza_knn_results_including_intermanuscripts (includes same-manuscript), 'excel' = Excel file.",
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
        "--max_image_edges",
        type=int,
        default=None,
        help="Maximum number of image-level edges to build from SQL (for global image graph).",
    )
    parser.add_argument(
        "--ms_ego_id",
        type=str,
        default="",
        help="If set (and the manuscript graph is built), extract an ego-subgraph around this manuscript id.",
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
        "--image_ego_id",
        type=str,
        default="",
        help="If set, extract and plot an ego-subgraph around this image id (image_name).",
    )
    parser.add_argument(
        "--image_ego_radius",
        type=int,
        default=1,
        help="Radius (in hops) for the image ego-subgraph.",
    )
    parser.add_argument(
        "--max_image_nodes",
        type=int,
        default=1000,
        help="Maximum number of nodes to include in the image ego-subgraph.",
    )
    parser.add_argument(
        "--interactive_ms_html",
        type=str,
        default="",
        help="If set, export an interactive HTML manuscript graph with an edge-weight slider.",
    )
    parser.add_argument(
        "--interactive_image_html",
        type=str,
        default="",
        help="If set (and --image_ego_id is provided), export an interactive HTML ego graph for that image.",
    )
    parser.add_argument(
        "--excel_path",
        type=str,
        default="aftertune/geniza_overall_neighbors_with_features.xlsx",
        help="Path to Excel neighbors file (used when --source excel).",
    )
    parser.add_argument(
        "--min_patches",
        type=int,
        default=None,
        help="Min query/neighbor num_visual_patches to include (default: system.RESULTS_ANALYSIS_MIN_PATCHES).",
    )
    parser.add_argument(
        "--min_glyphs",
        type=int,
        default=None,
        help="Min query/neighbor num_glyphs to include (default: system.RESULTS_ANALYSIS_MIN_GLYPHS).",
    )

    args = parser.parse_args()

    G_ms: nx.Graph | None = None
    G_img: nx.Graph | None = None

    # ── Manuscript graph ─────────────────────────────────────────────────────
    if args.graph in ("manuscripts", "both"):
        use_overall_for_ms = args.source in ("overall_sql", "excel")
        source_desc = "geniza_knn_results_including_intermanuscripts" if use_overall_for_ms else "geniza_knn_results"
        print(f"Building manuscript graph from SQL ({source_desc})...")
        G_ms, ms_counts = build_manuscript_graph_from_sql(
            db_config_path=args.db_config,
            min_similarity=args.min_similarity,
            min_edge_weight=args.min_ms_edge_weight,
            use_overall_table=use_overall_for_ms,
            min_patches=args.min_patches,
            min_glyphs=args.min_glyphs,
        )
        print(
            f"Manuscript graph: {len(G_ms.nodes()):,} nodes, "
            f"{len(G_ms.edges()):,} edges."
        )

        # Optional manuscript ego-subgraph around a specific manuscript id.
        H_ms: nx.Graph | None = None
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
            except ValueError as e:
                print(str(e))
                H_ms = None
            else:
                print(
                    f"Manuscript ego-subgraph: {len(H_ms.nodes()):,} nodes, "
                    f"{len(H_ms.edges()):,} edges."
                )

        # If interactive HTML is requested, export either the ego-subgraph (if built)
        # or the full manuscript graph.
        if args.interactive_ms_html:
            target_graph = H_ms if H_ms is not None else G_ms
            print(
                f"Exporting interactive manuscript HTML to {args.interactive_ms_html} "
                f"({len(target_graph.nodes())} nodes, {len(target_graph.edges())} edges)..."
            )
            export_manuscript_graph_interactive(target_graph, args.interactive_ms_html)

        # Optional static plot (can be heavy for 20K nodes; skip by default)
        # Uncomment if you want a quick visual:
        # plt.figure(figsize=(10, 8))
        # plot_manuscript_graph(G_ms, title="Manuscript connectivity graph (SQL)")

    # ── Image graph ──────────────────────────────────────────────────────────
    if args.graph in ("images", "both"):
        if args.source == "sql":
            print("Building global image graph from SQL (geniza_knn_results, excludes same-manuscript)...")
            G_img = build_image_graph_from_sql(
                db_config_path=args.db_config,
                min_similarity=args.min_similarity,
                max_edges=args.max_image_edges,
                min_patches=args.min_patches,
                min_glyphs=args.min_glyphs,
            )
        elif args.source == "overall_sql":
            print("Building global image graph from SQL (geniza_knn_results_including_intermanuscripts, includes same-manuscript)...")
            G_img = build_image_graph_from_overall_sql(
                db_config_path=args.db_config,
                min_similarity=args.min_similarity,
                max_edges=args.max_image_edges,
                min_patches=args.min_patches,
                min_glyphs=args.min_glyphs,
            )
        else:
            print("Building global image graph from Excel...")
            G_img = build_image_graph_from_excel(
                excel_path=args.excel_path,
                min_similarity=args.min_similarity,
                max_edges=args.max_image_edges,
                db_config_path=args.db_config,
            )
        print(
            f"Image graph (global): {len(G_img.nodes()):,} nodes, "
            f"{len(G_img.edges()):,} edges."
        )

        # Ego-subgraph visualization for a specific image
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
                print(str(e))
            else:
                print(
                    f"Ego-subgraph: {len(H.nodes()):,} nodes, "
                    f"{len(H.edges()):,} edges."
                )
                # Static plot (optional)
                plt.figure(figsize=(10, 8))
                plot_image_graph(H, title=f"Ego graph around {args.image_ego_id}")

                # Interactive HTML ego view
                if args.interactive_image_html:
                    export_image_ego_interactive(
                        H,
                        center_image_id=args.image_ego_id,
                        html_path=args.interactive_image_html,
                    )

    # Show any matplotlib figures created
    if plt.get_fignums():
        plt.show()


if __name__ == "__main__":
    main()
