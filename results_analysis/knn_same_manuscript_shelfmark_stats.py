#!/usr/bin/env python3
"""Report same-manuscript and same-shelfmark retrieval statistics from KNN rows.

The primary metric is success@K: the number and percentage of query images with
at least one matching suggestion among their first K neighbors.  The report also
shows the suggestion-level match rate (matching rows / returned rows) at each K.

By default this reads ``geniza_knn_results_including_intermanuscripts`` and
reports K = 1, 5, and 10.

Examples:
  python results_analysis/knn_same_manuscript_shelfmark_stats.py
  python results_analysis/knn_same_manuscript_shelfmark_stats.py --cutoffs 1 3 5 10
  python results_analysis/knn_same_manuscript_shelfmark_stats.py \
      --db-config /path/to/db_config.ini
"""

from __future__ import annotations

import argparse
import configparser
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system import CLUSTERING_DB_CONFIG_PATH, GENIZA_OVERALL_NEIGHBORS_TABLE


REQUIRED_COLUMNS = {
    "query_manuscript_id",
    "query_image_path",
    "neighbor_rank",
    "same_manuscript",
    "same_shelfmark",
}


@dataclass(frozen=True)
class TopKStats:
    cutoff: int
    queries_with_k_results: int
    same_manuscript_successes: int
    same_shelfmark_successes: int
    returned_suggestions: int
    same_manuscript_suggestions: int
    same_shelfmark_suggestions: int


@dataclass(frozen=True)
class KnnStats:
    query_count: int
    row_count: int
    min_rank: int | None
    max_rank: int | None
    by_cutoff: tuple[TopKStats, ...]


def get_db_connection(db_config_path: str):
    config_path = db_config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"DB config not found: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)
    section = "postgresql" if "postgresql" in config else "database"
    if section not in config:
        raise KeyError("DB config must contain a [postgresql] or [database] section")
    db = config[section]
    return psycopg2.connect(
        host=db["host"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
        port=db.get("port", 5432),
    )


def _validate_table(conn, table_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            """,
            (table_name,),
        )
        columns = {row[0] for row in cur.fetchall()}

    if not columns:
        raise ValueError(f"Table public.{table_name} does not exist")
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        raise ValueError(
            f"Table public.{table_name} is missing required columns: {', '.join(missing)}"
        )


def calculate_stats(
    conn,
    table_name: str = GENIZA_OVERALL_NEIGHBORS_TABLE,
    cutoffs: Sequence[int] = (1, 5, 10),
) -> KnnStats:
    """Calculate success@K and suggestion match rates in a single table scan."""
    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not normalized_cutoffs or normalized_cutoffs[0] < 1:
        raise ValueError("Cutoffs must be positive integers")

    _validate_table(conn, table_name)
    table = sql.Identifier("public", table_name)

    per_query_columns: list[sql.Composable] = [
        sql.SQL("query_manuscript_id"),
        sql.SQL("query_image_path"),
        sql.SQL("COUNT(*) AS row_count"),
        sql.SQL("MIN(neighbor_rank) AS min_rank"),
        sql.SQL("MAX(neighbor_rank) AS max_rank"),
    ]
    final_columns: list[sql.Composable] = [
        sql.SQL("COUNT(*) AS query_count"),
        sql.SQL("COALESCE(SUM(row_count), 0)::bigint AS row_count"),
        sql.SQL("MIN(min_rank) AS min_rank"),
        sql.SQL("MAX(max_rank) AS max_rank"),
    ]

    for cutoff in normalized_cutoffs:
        complete_alias = sql.Identifier(f"complete_k{cutoff}")
        returned_alias = sql.Identifier(f"returned_k{cutoff}")
        manuscript_hits_alias = sql.Identifier(f"manuscript_hits_k{cutoff}")
        shelfmark_hits_alias = sql.Identifier(f"shelfmark_hits_k{cutoff}")
        manuscript_success_alias = sql.Identifier(f"manuscript_success_k{cutoff}")
        shelfmark_success_alias = sql.Identifier(f"shelfmark_success_k{cutoff}")
        per_query_columns.extend(
            [
                sql.SQL(
                    "COUNT(*) FILTER (WHERE neighbor_rank <= {k}) AS {alias}"
                ).format(k=sql.Literal(cutoff), alias=returned_alias),
                sql.SQL(
                    "COUNT(*) FILTER (WHERE neighbor_rank <= {k} "
                    "AND COALESCE(same_manuscript, FALSE)) AS {alias}"
                ).format(k=sql.Literal(cutoff), alias=manuscript_hits_alias),
                sql.SQL(
                    "COUNT(*) FILTER (WHERE neighbor_rank <= {k} "
                    "AND COALESCE(same_shelfmark, FALSE)) AS {alias}"
                ).format(k=sql.Literal(cutoff), alias=shelfmark_hits_alias),
            ]
        )
        final_columns.extend(
            [
                sql.SQL(
                    "COUNT(*) FILTER (WHERE max_rank >= {k}) AS {alias}"
                ).format(k=sql.Literal(cutoff), alias=complete_alias),
                sql.SQL(
                    "COUNT(*) FILTER (WHERE {hits} > 0) AS {alias}"
                ).format(hits=manuscript_hits_alias, alias=manuscript_success_alias),
                sql.SQL(
                    "COUNT(*) FILTER (WHERE {hits} > 0) AS {alias}"
                ).format(hits=shelfmark_hits_alias, alias=shelfmark_success_alias),
                sql.SQL(
                    "COALESCE(SUM({source}), 0)::bigint AS {alias}"
                ).format(source=returned_alias, alias=returned_alias),
                sql.SQL(
                    "COALESCE(SUM({source}), 0)::bigint AS {alias}"
                ).format(source=manuscript_hits_alias, alias=manuscript_hits_alias),
                sql.SQL(
                    "COALESCE(SUM({source}), 0)::bigint AS {alias}"
                ).format(source=shelfmark_hits_alias, alias=shelfmark_hits_alias),
            ]
        )

    query = sql.SQL(
        """
        WITH per_query AS (
            SELECT {per_query_columns}
            FROM {table}
            GROUP BY query_manuscript_id, query_image_path
        )
        SELECT {final_columns}
        FROM per_query
        """
    ).format(
        per_query_columns=sql.SQL(",\n                   ").join(per_query_columns),
        final_columns=sql.SQL(",\n               ").join(final_columns),
        table=table,
    )

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query)
        row = cur.fetchone()

    if not row:
        return KnnStats(0, 0, None, None, ())

    by_cutoff = tuple(
        TopKStats(
            cutoff=cutoff,
            queries_with_k_results=int(row[f"complete_k{cutoff}"] or 0),
            same_manuscript_successes=int(row[f"manuscript_success_k{cutoff}"] or 0),
            same_shelfmark_successes=int(row[f"shelfmark_success_k{cutoff}"] or 0),
            returned_suggestions=int(row[f"returned_k{cutoff}"] or 0),
            same_manuscript_suggestions=int(row[f"manuscript_hits_k{cutoff}"] or 0),
            same_shelfmark_suggestions=int(row[f"shelfmark_hits_k{cutoff}"] or 0),
        )
        for cutoff in normalized_cutoffs
    )
    return KnnStats(
        query_count=int(row["query_count"] or 0),
        row_count=int(row["row_count"] or 0),
        min_rank=int(row["min_rank"]) if row["min_rank"] is not None else None,
        max_rank=int(row["max_rank"]) if row["max_rank"] is not None else None,
        by_cutoff=by_cutoff,
    )


def _ratio(numerator: int, denominator: int) -> str:
    if not denominator:
        return "N/A"
    return f"{numerator:,} / {denominator:,} ({100.0 * numerator / denominator:.2f}%)"


def print_report(stats: KnnStats, table_name: str) -> None:
    print(f"Source: public.{table_name}")
    print(f"Query images: {stats.query_count:,}")
    print(f"Neighbor rows: {stats.row_count:,}")
    print(f"Available rank range: {stats.min_rank}–{stats.max_rank}")
    print()
    print("Query-level success@K (at least one matching suggestion)")
    print(f"{'K':>4}  {'Same manuscript':>34}  {'Same shelfmark':>34}")
    for item in stats.by_cutoff:
        print(
            f"{item.cutoff:>4}  "
            f"{_ratio(item.same_manuscript_successes, stats.query_count):>34}  "
            f"{_ratio(item.same_shelfmark_successes, stats.query_count):>34}"
        )

    print()
    print("Suggestion-level match rate within top K")
    print(f"{'K':>4}  {'Same manuscript':>34}  {'Same shelfmark':>34}")
    for item in stats.by_cutoff:
        print(
            f"{item.cutoff:>4}  "
            f"{_ratio(item.same_manuscript_suggestions, item.returned_suggestions):>34}  "
            f"{_ratio(item.same_shelfmark_suggestions, item.returned_suggestions):>34}"
        )

    incomplete = [
        item
        for item in stats.by_cutoff
        if item.queries_with_k_results != stats.query_count
    ]
    if incomplete:
        print()
        print("Coverage warning: not every query reaches every requested K:")
        for item in incomplete:
            print(
                f"  K={item.cutoff}: "
                f"{item.queries_with_k_results:,} / {stats.query_count:,} queries"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db-config",
        default=CLUSTERING_DB_CONFIG_PATH,
        help="PostgreSQL config INI (default: system.CLUSTERING_DB_CONFIG_PATH)",
    )
    parser.add_argument(
        "--table",
        default=GENIZA_OVERALL_NEIGHBORS_TABLE,
        help=f"KNN table (default: {GENIZA_OVERALL_NEIGHBORS_TABLE})",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        metavar="K",
        help="Top-K cutoffs (default: 1 5 10)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        conn = get_db_connection(args.db_config)
        try:
            stats = calculate_stats(conn, args.table, args.cutoffs)
            print_report(stats, args.table)
        finally:
            conn.close()
    except (FileNotFoundError, KeyError, ValueError, psycopg2.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
