#!/usr/bin/env python3
"""
Launch multi-GPU projection runs for Geniza manuscripts.

This is a thin orchestrator around `aftertune/project_geniza_to_latent.py`.
It:
  1. Detects how many GPUs to use (or uses --gpus).
  2. Queries the DB for DISTINCT manuscript_id values (optionally restricted to colored images only).
  3. Divides manuscript_ids into N contiguous shards.
  4. Spawns one process per GPU, each with:
       - CUDA_VISIBLE_DEVICES set to a single GPU index
       - --manuscript-min / --manuscript-max set to that shard's [min, max]

PCA / latent_vector_search is **not** handled here; run:
  python aftertune/recompute_geniza_pca.py
once all projection jobs have completed.
"""

import os
import sys
import argparse
import configparser
import logging
from datetime import datetime
import subprocess
from typing import List, Tuple

import psycopg2
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system import (
    CLUSTERING_DB_CONFIG_PATH,
    GENIZA_IMAGE_INFORMATION_TABLE,
)


def setup_logger(log_dir: str | None = None) -> logging.Logger:
    if log_dir is None:
        log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"launch_project_geniza_multi_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def get_db_connection(db_config_path: str | None) -> psycopg2.extensions.connection:
    cfg = db_config_path or CLUSTERING_DB_CONFIG_PATH
    if not os.path.isabs(cfg):
        cfg = os.path.join(project_root, cfg)
    config = configparser.ConfigParser()
    config.read(cfg)
    db = config["postgresql"]
    return psycopg2.connect(
        host=db["host"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
        port=db.get("port", 5432),
    )


def load_distinct_manuscript_ids(
    conn: psycopg2.extensions.connection,
    colorful_only: bool,
    log: logging.Logger,
) -> List[str]:
    where = "WHERE image_path IS NOT NULL AND image_path != ''"
    if colorful_only:
        where += " AND is_color = true"
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT manuscript_id
            FROM {GENIZA_IMAGE_INFORMATION_TABLE}
            {where}
            ORDER BY manuscript_id
        """
        )
        rows = cur.fetchall()
    manuscript_ids = [str(r[0]) for r in rows if r[0] is not None]
    log.info(
        "Found %d distinct manuscript_id values in %s (colorful_only=%s)",
        len(manuscript_ids),
        GENIZA_IMAGE_INFORMATION_TABLE,
        colorful_only,
    )
    return manuscript_ids


def shard_manuscript_ids(
    manuscript_ids: List[str], num_shards: int
) -> List[Tuple[str, str]]:
    """
    Returns a list of (min_id, max_id) ranges, one per shard.
    Uses contiguous slices of the sorted manuscript_ids list.
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    n = len(manuscript_ids)
    if n == 0:
        return []
    shard_sizes = []
    base = n // num_shards
    rem = n % num_shards
    for i in range(num_shards):
        size = base + (1 if i < rem else 0)
        shard_sizes.append(size)
    ranges: List[Tuple[str, str]] = []
    start = 0
    for size in shard_sizes:
        if size == 0:
            ranges.append((manuscript_ids[0], manuscript_ids[-1]))
            continue
        end = start + size
        sub = manuscript_ids[start:end]
        ranges.append((sub[0], sub[-1]))
        start = end
    return ranges


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Launch multi-GPU runs of aftertune/project_geniza_to_latent.py, "
            "sharding by manuscript_id ranges."
        )
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (defaults to torch.cuda.device_count()).",
    )
    parser.add_argument(
        "--db-config",
        type=str,
        default=None,
        help="Path to db_config.ini (defaults to CLUSTERING_DB_CONFIG_PATH).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional: override checkpoint path passed to project_geniza_to_latent.py.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset to pass through (normally 0, since we shard by manuscript_id).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to pass through (0 = no limit, all matching rows per shard).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional: override batch size for projection script.",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help="Optional: override images_root for projection script.",
    )
    parser.add_argument(
        "--colorful-only",
        action="store_true",
        default=True,
        help="Restrict both manuscript_id discovery and projection to is_color = true.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-GPU commands but do not actually launch them.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for this launcher log file.",
    )
    args = parser.parse_args()

    log = setup_logger(args.log_dir)

    # Detect GPUs
    if args.gpus is not None:
        num_gpus = args.gpus
    else:
        num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError(
            "No CUDA devices detected and --gpus not specified. "
            "Run the projection script directly on CPU if needed."
        )
    log.info("Using %d GPUs for projection sharding", num_gpus)

    # Load manuscript_id space and compute shards
    conn = get_db_connection(args.db_config)
    manuscript_ids = load_distinct_manuscript_ids(
        conn, colorful_only=args.colorful_only, log=log
    )
    conn.close()
    if not manuscript_ids:
        log.warning("No manuscript_id values found; nothing to do.")
        return

    ranges = shard_manuscript_ids(manuscript_ids, num_gpus)
    for gpu_idx, (mn_min, mn_max) in enumerate(ranges):
        log.info(
            "GPU %d will handle manuscript_id in [%s, %s]", gpu_idx, mn_min, mn_max
        )

    # Build and optionally launch per-GPU commands
    procs: list[subprocess.Popen] = []
    for gpu_idx, (mn_min, mn_max) in enumerate(ranges):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        cmd = [sys.executable, "aftertune/project_geniza_to_latent.py"]
        if args.checkpoint is not None:
            cmd += ["--checkpoint", args.checkpoint]
        if args.images_root is not None:
            cmd += ["--images_root", args.images_root]
        if args.db_config is not None:
            cmd += ["--db-config", args.db_config]
        if args.batch_size is not None:
            cmd += ["--batch_size", str(args.batch_size)]
        cmd += [
            "--offset",
            str(args.offset),
            "--limit",
            str(args.limit),
            "--manuscript-min",
            str(mn_min),
            "--manuscript-max",
            str(mn_max),
        ]
        if args.colorful_only:
            cmd.append("--colorful-only")

        log.info("GPU %d command: %s", gpu_idx, " ".join(cmd))

        if not args.dry_run:
            proc = subprocess.Popen(cmd, env=env, cwd=project_root)
            procs.append(proc)

    if args.dry_run:
        log.info("Dry run complete; no processes were launched.")
        return

    # Wait for all children to finish
    for i, p in enumerate(procs):
        ret = p.wait()
        log.info("GPU %d process exited with code %s", i, ret)


if __name__ == "__main__":
    main()

