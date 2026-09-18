#!/usr/bin/env python3
"""
Report statistics from the clustering DB (same config as compare_clusters_pairs.py).

Useful when ALTO/XML files are missing on disk: feature counts (tiles, glyphs, words)
are whatever was stored in geniza_image_latents at projection time — this script
summarizes those columns plus latent coverage, optional KNN table sizes, and
(re)prints the same **overall top-K neighbor evaluation** block as
`geniza_top_neighbors_gpu.py` (from `geniza_knn_results_including_intermanuscripts`,
including same-manuscript neighbors).

Examples:
  python results_analysis/test_set/report_db_coverage_stats.py
  python results_analysis/test_set/report_db_coverage_stats.py \\
      --metadata-csv results_analysis/test_set/clusters_images_metadata.csv
  python results_analysis/test_set/report_db_coverage_stats.py --db-config /path/to.ini
"""

from __future__ import annotations

import argparse
import configparser
import os
import sys
from typing import Any

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import psycopg2
from psycopg2.extras import RealDictCursor

from system import (
    CLUSTERING_DB_CONFIG_PATH,
    GENIZA_IMAGE_INFORMATION_TABLE,
    GENIZA_IMAGE_LATENTS_TABLE,
    GENIZA_KNN_RESULTS_TABLE,
    GENIZA_OVERALL_NEIGHBORS_TABLE,
)


def get_db_connection(db_config_path: str):
    cfg = db_config_path
    if not os.path.isabs(cfg):
        cfg = os.path.join(project_root, cfg)
    if not os.path.exists(cfg):
        raise FileNotFoundError(f"Config not found: {cfg}")
    config = configparser.ConfigParser()
    config.read(cfg)
    section = "postgresql" if "postgresql" in config else "database"
    db = config[section]
    return psycopg2.connect(
        host=db["host"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
        port=db.get("port", 5432),
    )


def _table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
            """,
            (table_name,),
        )
        return cur.fetchone() is not None


def _columns(conn, table_name: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            """,
            (table_name,),
        )
        return {r[0] for r in cur.fetchall()}


def _print_kv(title: str, rows: list[tuple[str, Any]]) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    w = max(len(k) for k, _ in rows) if rows else 0
    for k, v in rows:
        print(f"  {k:<{w}}  {v}")


def summarize_latents_global(conn, table: str) -> None:
    cols = _columns(conn, table)
    if not cols:
        print(f"Table {table!r} not found or has no columns.")
        return

    selects = [
        "COUNT(*) AS n_rows",
        "COUNT(latent_vector) AS n_latent_vector",
        "COUNT(latent_vector_search) AS n_latent_vector_search",
    ]
    if "xml_path" in cols:
        selects.append("COUNT(xml_path) AS n_xml_path_set")
        selects.append(
            "COUNT(*) FILTER (WHERE xml_path IS NOT NULL AND TRIM(xml_path) <> '') AS n_xml_path_nonempty"
        )
    for c in ("num_visual_patches", "num_glyphs", "num_words"):
        if c in cols:
            selects.extend(
                [
                    f"COUNT({c}) AS {c}_nonnull",
                    f"COUNT(*) FILTER (WHERE {c} IS NOT NULL AND {c} = 0) AS {c}_zero",
                    f"MIN({c}) AS {c}_min",
                    f"MAX({c}) AS {c}_max",
                    f"ROUND(AVG({c})::numeric, 2) AS {c}_avg",
                    f"ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {c}::float))::numeric, 2) AS {c}_median",
                ]
            )

    sql = f"SELECT {', '.join(selects)} FROM {table}"
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        row = cur.fetchone()
    if not row:
        return

    out: list[tuple[str, Any]] = [("table", table)]
    for k in sorted(row.keys()):
        out.append((k, row[k]))
    _print_kv(f"{table} — global summary", out)


def summarize_latents_for_paths(conn, table: str, image_paths: list[str]) -> None:
    if not image_paths:
        return
    cols = _columns(conn, table)
    if "image_path" not in cols:
        return

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT COUNT(*) AS n_subset
            FROM {table}
            WHERE image_path = ANY(%s)
            """,
            (image_paths,),
        )
        n_in = cur.fetchone()["n_subset"]

    selects = [
        "COUNT(*) AS n_rows",
        "COUNT(latent_vector) AS n_latent_vector",
        "COUNT(latent_vector_search) AS n_latent_vector_search",
    ]
    if "xml_path" in cols:
        selects.append("COUNT(xml_path) AS n_xml_path_set")
        selects.append(
            "COUNT(*) FILTER (WHERE xml_path IS NOT NULL AND TRIM(xml_path) <> '') AS n_xml_path_nonempty"
        )
    for c in ("num_visual_patches", "num_glyphs", "num_words"):
        if c in cols:
            selects.extend(
                [
                    f"COUNT({c}) AS {c}_nonnull",
                    f"COUNT(*) FILTER (WHERE {c} IS NOT NULL AND {c} = 0) AS {c}_zero",
                    f"MIN({c}) AS {c}_min",
                    f"MAX({c}) AS {c}_max",
                    f"ROUND(AVG({c})::numeric, 2) AS {c}_avg",
                    f"ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {c}::float))::numeric, 2) AS {c}_median",
                ]
            )

    sql = f"""
        SELECT {", ".join(selects)}
        FROM {table}
        WHERE image_path = ANY(%s)
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (image_paths,))
        row = cur.fetchone()

    out: list[tuple[str, Any]] = []
    for k in sorted(row.keys()):
        out.append((k, row[k]))
    title = f"{table} — rows matching test-set CSV ({len(image_paths)} unique paths)"
    _print_kv(title, out)
    missing = len(image_paths) - int(n_in or 0)
    _print_kv(
        "CSV ↔ DB overlap",
        [
            ("unique image_path in CSV", len(image_paths)),
            ("rows found in DB", int(n_in or 0)),
            ("CSV paths missing from DB", missing),
            (
                "fraction of CSV with DB row",
                f"{(n_in / len(image_paths)):.1%}" if image_paths else "n/a",
            ),
        ],
    )


def _pct(n: float, d: float) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "N/A"


def print_overall_topk_evaluation_from_db(conn, table: str) -> None:
    """
    Reproduce the log block from geniza_top_neighbors_gpu.py (OVERALL TOP-K NEAREST-NEIGHBOR
    EVALUATION) using only SQL on the overall-neighbors table (same-manuscript joins included).
    """
    if not _table_exists(conn, table):
        print(f"\n(Skipping overall NN evaluation: table {table!r} not found.)")
        return
    cols = _columns(conn, table)
    required = {"neighbor_rank", "same_manuscript", "query_image_path"}
    if not required.issubset(cols):
        missing = required - cols
        print(f"\n(Skipping overall NN evaluation: {table} missing columns {missing})")
        return

    key_cols = ["query_image_path"]
    if "query_manuscript_id" in cols:
        key_cols.insert(0, "query_manuscript_id")
    distinct_key = ", ".join(key_cols)

    sql = f"""
    SELECT
      (SELECT MAX(neighbor_rank) FROM {table}) AS k_max,
      (SELECT COUNT(*) FROM (SELECT DISTINCT {distinct_key} FROM {table}) x) AS n_queries,
      (SELECT COUNT(*) FROM {table} WHERE neighbor_rank = 1 AND same_manuscript) AS rank1_same,
      (SELECT COUNT(*) FROM (
         SELECT 1 FROM {table} WHERE same_manuscript
         GROUP BY {distinct_key}
      ) y) AS queries_with_any_same,
      (SELECT COUNT(*)::bigint FROM {table} WHERE same_manuscript) AS same_hits,
      (SELECT COUNT(*)::bigint FROM {table} WHERE NOT same_manuscript) AS diff_hits,
      (SELECT COUNT(*)::bigint FROM {table}) AS total_rows
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        row = cur.fetchone()

    k = row["k_max"]
    nq = int(row["n_queries"] or 0)
    r1 = int(row["rank1_same"] or 0)
    any_same = int(row["queries_with_any_same"] or 0)
    same_h = int(row["same_hits"] or 0)
    diff_h = int(row["diff_hits"] or 0)
    tot = int(row["total_rows"] or 0)

    if k is None or nq == 0:
        print(f"\n{'=' * 70}\n  OVERALL TOP-K NEAREST-NEIGHBOR EVALUATION (from DB: {table})\n{'=' * 70}")
        print("  (Table is empty or has no neighbor_rank; nothing to summarize.)")
        return

    k_int = int(k)
    print(f"\n{'=' * 70}")
    print(f"  OVERALL TOP-{k_int} NEAREST-NEIGHBOR EVALUATION (from DB: {table})")
    print(f"{'=' * 70}")
    print(f"Total query images: {nq:,}")
    print(f"Rank-1 NN same manuscript: {r1:,} ({_pct(r1, nq)})")
    print(f"At least 1 same-ms in top-{k_int}: {any_same:,} ({_pct(any_same, nq)})")
    print(f"Same-manuscript hits: {same_h:,} ({_pct(same_h, tot)})")
    print(f"Different-manuscript hits: {diff_h:,} ({_pct(diff_h, tot)})")
    if k_int > 0 and nq > 0:
        expected = nq * k_int
        if tot != expected:
            print(
                f"Note: total table rows ({tot:,}) != n_queries × K ({expected:,}); "
                "ranks per query may be incomplete or K mixed."
            )


def summarize_row_count(conn, table: str, label: str) -> None:
    if not _table_exists(conn, table):
        print(f"\n  ({label}: table {table!r} not present)")
        return
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        n = cur.fetchone()[0]
    print(f"  {label}: {n:,} rows ({table})")


def summarize_geniza_image_information(conn) -> None:
    t = GENIZA_IMAGE_INFORMATION_TABLE
    if not _table_exists(conn, t):
        print(f"\n  (optional source catalog {t!r} not present)")
        return
    cols = _columns(conn, t)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM {t}")
        n = cur.fetchone()["n"]
    extra = []
    if "image_path" in cols:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) AS n FROM {t}
                WHERE image_path IS NOT NULL AND TRIM(image_path) <> ''
                """
            )
            extra.append(("rows with nonempty image_path", cur.fetchone()["n"]))
    if "xml_path" in cols:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) AS n FROM {t}
                WHERE xml_path IS NOT NULL AND TRIM(xml_path) <> ''
                """
            )
            extra.append(("rows with nonempty xml_path", cur.fetchone()["n"]))
    rows = [("table", t), ("total_rows", n)] + extra
    _print_kv(f"{t} (projection source catalog)", rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--db-config",
        type=str,
        default=CLUSTERING_DB_CONFIG_PATH,
        help="PostgreSQL config INI (default: CLUSTERING_DB_CONFIG_PATH from system.py)",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=os.path.join(script_dir, "clusters_images_metadata.csv"),
        help="Test-set CSV with image_path column (default: clusters_images_metadata.csv next to this script). "
        "Use '' to skip CSV overlap section.",
    )
    parser.add_argument(
        "--latents-table",
        type=str,
        default=GENIZA_IMAGE_LATENTS_TABLE,
        help="Override latent table name (default from system.py)",
    )
    parser.add_argument(
        "--skip-overall-nn-stats",
        action="store_true",
        help="Do not run SQL aggregates on geniza_knn_results_including_intermanuscripts.",
    )
    parser.add_argument(
        "--overall-neighbors-table",
        type=str,
        default="",
        help=f"Table for overall NN stats (default: {GENIZA_OVERALL_NEIGHBORS_TABLE} from system.py)",
    )
    args = parser.parse_args()

    print(f"DB config: {args.db_config}")
    print(f"Project root: {project_root}")

    conn = get_db_connection(args.db_config)
    try:
        if not _table_exists(conn, args.latents_table):
            print(f"ERROR: table {args.latents_table!r} does not exist in this database.")
            return 1

        summarize_latents_global(conn, args.latents_table)
        summarize_geniza_image_information(conn)

        csv_path = (args.metadata_csv or "").strip()
        if csv_path:
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(script_dir, csv_path)
            if not os.path.isfile(csv_path):
                print(f"\n  (metadata CSV not found, skipping overlap: {csv_path})")
            else:
                import pandas as pd

                df = pd.read_csv(csv_path)
                if "image_path" not in df.columns:
                    print(f"\n  (CSV has no image_path column; columns: {list(df.columns)})")
                else:
                    paths = (
                        df["image_path"]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )
                    paths = [p for p in paths.unique().tolist() if p]
                    print(f"\nLoaded {len(paths)} unique image_path values from {csv_path}")
                    summarize_latents_for_paths(conn, args.latents_table, paths)

        overall_tbl = (args.overall_neighbors_table or "").strip() or GENIZA_OVERALL_NEIGHBORS_TABLE
        if not args.skip_overall_nn_stats:
            print_overall_topk_evaluation_from_db(conn, overall_tbl)

        print(f"\n{'=' * 72}\nRelated tables (row counts)\n{'=' * 72}")
        summarize_row_count(conn, GENIZA_KNN_RESULTS_TABLE, "Cross-manuscript KNN")
        summarize_row_count(conn, overall_tbl, "Overall neighbors (incl. same MS)")
    finally:
        conn.close()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
