#!/usr/bin/env python3
"""
Recompute PCA / latent_vector_search for geniza_image_latents.

This script:
1. Connects to the clustering database.
2. Loads ALL non-null latent_vector rows from GENIZA_IMAGE_LATENTS_TABLE.
3. Runs PCA without whitening, then L2 normalises to CLUSTERING_SEARCH_VECTOR_DIM.
4. Updates latent_vector_search for every row.

Run this AFTER projection is complete (possibly multi-GPU, sharded by manuscript_id)
so that PCA sees the full population of latent vectors.
"""

import os
import sys
import argparse
import configparser
import logging
from datetime import datetime

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import psycopg2

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system import (
    CLUSTERING_DB_CONFIG_PATH,
    CLUSTERING_SEARCH_VECTOR_DIM,
    GENIZA_IMAGE_LATENTS_TABLE,
)


def setup_logger(log_dir=None):
    if log_dir is None:
        log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"recompute_geniza_pca_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def get_db_connection(db_config_path):
    cfg = db_config_path
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


def run(db_config_path, search_dim, log):
    conn = get_db_connection(db_config_path or CLUSTERING_DB_CONFIG_PATH)

    log.info("Loading ALL latent vectors from %s for PCA computation...", GENIZA_IMAGE_LATENTS_TABLE)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT image_path, latent_vector::text
            FROM {GENIZA_IMAGE_LATENTS_TABLE}
            WHERE latent_vector IS NOT NULL
            ORDER BY manuscript_id, parent_directory, picture_id
        """
        )
        all_db_rows = cur.fetchall()

    if not all_db_rows:
        log.warning("No latent vectors found in database - aborting PCA computation")
        conn.close()
        return 0

    # Parse vectors from database (pgvector stores them as text like "[0.1,0.2,...]")
    all_image_paths = []
    all_latent_vectors = []
    for image_path, vec_str in all_db_rows:
        vec_str = vec_str.strip()
        if vec_str.startswith("[") and vec_str.endswith("]"):
            vec_str = vec_str[1:-1]
        elif vec_str.startswith("(") and vec_str.endswith(")"):
            vec_str = vec_str[1:-1]
        parts = [x.strip() for x in vec_str.split(",") if x.strip()]
        if not parts:
            log.warning("Empty vector for image_path %s, skipping", image_path)
            continue
        vec_array = np.array([float(x) for x in parts], dtype=np.float32)
        all_image_paths.append(image_path)
        all_latent_vectors.append(vec_array)

    log.info("Loaded %d latent vectors from database for PCA", len(all_latent_vectors))

    all_latent_matrix = np.vstack(all_latent_vectors)

    # Sanity check: latent variance
    latent_var_per_dim = np.var(all_latent_matrix, axis=0)
    latent_var_total = np.mean(latent_var_per_dim)
    log.info(
        "All latents stats: shape=%s, per-dim variance min=%.6f max=%.6f mean=%.6f",
        all_latent_matrix.shape,
        float(np.min(latent_var_per_dim)),
        float(np.max(latent_var_per_dim)),
        float(latent_var_total),
    )
    if latent_var_total < 1e-8:
        log.warning(
            "Latent variance is near zero — vectors may be degenerate (check model and input pipeline)."
        )

    # Preserve the dominant retrieval geometry with ordinary PCA, then
    # L2-normalise so cosine distance remains meaningful.
    n_samples, n_features = all_latent_matrix.shape
    n_components = min(search_dim, n_features, n_samples)
    if n_components < search_dim:
        log.warning("PCA limited to %d components", n_components)

    log.info(
        "Computing PCA-reduced vectors (dim=%d, whiten=False) on ALL %d images...",
        search_dim,
        len(all_latent_vectors),
    )
    pca = PCA(n_components=n_components, whiten=False, svd_solver="auto", random_state=42)
    reduced = pca.fit_transform(all_latent_matrix)
    log.info(
        "PCA explained variance: total=%.4f, per-component min=%.6f max=%.6f",
        float(np.sum(pca.explained_variance_ratio_)),
        float(np.min(pca.explained_variance_ratio_)),
        float(np.max(pca.explained_variance_ratio_)),
    )

    # L2-normalise the PCA vectors so cosine distance is meaningful
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    reduced = reduced / norms

    if n_components < search_dim:
        padding = np.zeros((reduced.shape[0], search_dim - n_components), dtype=reduced.dtype)
        reduced = np.hstack([reduced, padding])

    # Update ALL rows with PCA-reduced vectors
    updated = 0
    for i, (image_path, latent_search) in enumerate(
        tqdm(
            zip(all_image_paths, reduced),
            total=len(all_image_paths),
            desc="Updating PCA vectors for all images",
        )
    ):
        search_str = "[" + ",".join(f"{x:.6f}" for x in latent_search.tolist()) + "]"
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {GENIZA_IMAGE_LATENTS_TABLE}
                    SET latent_vector_search = %s::vector
                    WHERE image_path = %s
                """,
                    (search_str, image_path),
                )
            updated += 1
        except Exception as e:
            logging.exception("Update PCA vector failed for %s: %s", image_path, e)
            conn.rollback()
        if (i + 1) % 500 == 0:
            conn.commit()
    conn.commit()
    conn.close()

    log.info("Done. Updated latent_vector_search for %d images.", updated)
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Recompute PCA (latent_vector_search) for all geniza_image_latents rows."
    )
    parser.add_argument(
        "--db-config",
        type=str,
        default=None,
        help="Path to db_config.ini (defaults to CLUSTERING_DB_CONFIG_PATH)",
    )
    parser.add_argument(
        "--search_dim",
        type=int,
        default=CLUSTERING_SEARCH_VECTOR_DIM,
        help="PCA dimension for latent_vector_search",
    )
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    log = setup_logger(args.log_dir)
    run(db_config_path=args.db_config, search_dim=args.search_dim, log=log)


if __name__ == "__main__":
    main()
