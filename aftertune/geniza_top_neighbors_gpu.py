#!/usr/bin/env python3
"""
GPU-accelerated top-K nearest neighbours for geniza images.

Uses PyTorch on GPU for batch cosine-similarity KNN. Recommended with
--vector full (full latent vectors).

- Vectors: read from SQL as stored (no rounding). Parsed to float64 and
  computed in float64 so results match PostgreSQL double precision.
- We do not write vectors to SQL; we only write KNN result rows
  (similarity_score, distance_score, etc.) as DOUBLE PRECISION.

Reads from geniza_image_latents and geniza_manuscript_shelfmark.
"Same shelfmark" means: same (normalized_library, shelfmark_root). Manuscripts
missing either field are treated as having no shelfmark (i.e., never match).

Writes two complementary tables:
- geniza_knn_results — "joins edition": only neighbors with a DIFFERENT
  manuscript AND a DIFFERENT (library, shelfmark_root) than the query (real
  cross-document candidates). Schema is unchanged.
- geniza_knn_results_including_intermanuscripts — "network evaluation":
  all neighbors (no exclusions), with the existing same_manuscript flag plus
  a new same_shelfmark flag so we can measure recall against both criteria.

Requires: PyTorch with CUDA.
"""

import os
import sys
import configparser
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from system import (
    CLUSTERING_DB_CONFIG_PATH,
    CLUSTERING_N_NEIGHBORS,
    CLUSTERING_COMMIT_INTERVAL,
    GENIZA_IMAGE_LATENTS_TABLE,
    GENIZA_KNN_RESULTS_TABLE,
    GENIZA_OVERALL_NEIGHBORS_TABLE,
    GENIZA_MANUSCRIPT_SHELFMARK_TABLE
)

VECTOR_SEARCH = "search"
VECTOR_FULL = "full"


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


def _vector_column(use_full_vector: bool) -> str:
    return "latent_vector" if use_full_vector else "latent_vector_search"


def _parse_vector(v) -> np.ndarray:
    """Parse vector from DB (string or array) to numpy float64. No rounding — full precision as stored in SQL."""
    if isinstance(v, str):
        if v.startswith("["):
            v = v.strip("[]")
            # Strip each component so " -0.026483 " parses correctly; full precision
            return np.array([float(x.strip()) for x in v.split(",")], dtype=np.float64)
        raise ValueError(f"Unexpected vector format: {v[:50]}")
    if hasattr(v, "tolist"):
        return np.array(v.tolist(), dtype=np.float64)
    return np.array(v, dtype=np.float64)


def ensure_results_table(conn):
    """Joins-edition table: neighbors with different manuscript AND different shelfmark.

    Schema is unchanged from before — we only filter which rows are inserted.
    """
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {GENIZA_KNN_RESULTS_TABLE} (
                id SERIAL PRIMARY KEY,
                query_manuscript_id TEXT NOT NULL,
                query_image_path TEXT NOT NULL,
                query_image_name TEXT NOT NULL,
                neighbor_rank INTEGER NOT NULL,
                neighbor_manuscript_id TEXT NOT NULL,
                neighbor_image_path TEXT NOT NULL,
                neighbor_image_name TEXT NOT NULL,
                similarity_score DOUBLE PRECISION NOT NULL,
                distance_score DOUBLE PRECISION NOT NULL,
                same_manuscript BOOLEAN NOT NULL,
                total_images_in_query_manuscripts INTEGER NOT NULL,
                query_num_visual_patches INTEGER,
                query_num_glyphs INTEGER,
                query_num_words INTEGER,
                neighbor_num_visual_patches INTEGER,
                neighbor_num_glyphs INTEGER,
                neighbor_num_words INTEGER
            )
        """)
        for col in ("query_num_visual_patches", "query_num_glyphs", "query_num_words",
                    "neighbor_num_visual_patches", "neighbor_num_glyphs", "neighbor_num_words"):
            cur.execute(f"ALTER TABLE {GENIZA_KNN_RESULTS_TABLE} ADD COLUMN IF NOT EXISTS {col} INTEGER")
    conn.commit()


def ensure_overall_neighbors_table(conn):
    """Network-evaluation table: ALL neighbors, with same_manuscript and same_shelfmark flags.

    Adds same_shelfmark column on top of the previous schema.
    """
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {GENIZA_OVERALL_NEIGHBORS_TABLE} (
                id SERIAL PRIMARY KEY,
                query_manuscript_id TEXT NOT NULL,
                query_image_path TEXT NOT NULL,
                query_image_name TEXT NOT NULL,
                neighbor_rank INTEGER NOT NULL,
                neighbor_manuscript_id TEXT NOT NULL,
                neighbor_image_path TEXT NOT NULL,
                neighbor_image_name TEXT NOT NULL,
                similarity_score DOUBLE PRECISION NOT NULL,
                distance_score DOUBLE PRECISION NOT NULL,
                same_manuscript BOOLEAN NOT NULL,
                same_shelfmark BOOLEAN NOT NULL DEFAULT FALSE,
                total_images_in_query_manuscripts INTEGER NOT NULL,
                query_num_visual_patches INTEGER,
                query_num_glyphs INTEGER,
                query_num_words INTEGER,
                neighbor_num_visual_patches INTEGER,
                neighbor_num_glyphs INTEGER,
                neighbor_num_words INTEGER
            )
        """)
        cur.execute(f"ALTER TABLE {GENIZA_OVERALL_NEIGHBORS_TABLE} "
                    f"ADD COLUMN IF NOT EXISTS same_shelfmark BOOLEAN NOT NULL DEFAULT FALSE")
        for col in ("query_num_visual_patches", "query_num_glyphs", "query_num_words",
                    "neighbor_num_visual_patches", "neighbor_num_glyphs", "neighbor_num_words"):
            cur.execute(f"ALTER TABLE {GENIZA_OVERALL_NEIGHBORS_TABLE} ADD COLUMN IF NOT EXISTS {col} INTEGER")
    conn.commit()


def load_manuscript_shelfmarks(conn) -> Dict[str, str]:
    """Return {manuscript_id: shelfmark_key} from geniza_manuscript_shelfmark.

    Two manuscripts are considered to share a shelfmark if they share both
    `normalized_library` AND `shelfmark_root`. We encode that as a single key
    string `f"{normalized_library}||{shelfmark_root}"` (with both sides
    stripped). Manuscripts with an empty library or empty shelfmark_root get
    "" as the key, which is treated as "no known shelfmark" and never matches
    anything (not even another "").
    """
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT manuscript_id, normalized_library, shelfmark_root
            FROM {GENIZA_MANUSCRIPT_SHELFMARK_TABLE}
        """)
        mapping: Dict[str, str] = {}
        for mid, lib, root in cur.fetchall():
            lib_s = (lib or "").strip()
            root_s = (root or "").strip()
            mapping[str(mid)] = f"{lib_s}||{root_s}" if lib_s and root_s else ""
    print(f"Loaded {len(mapping):,} manuscript shelfmark keys (library||shelfmark_root) "
          f"from {GENIZA_MANUSCRIPT_SHELFMARK_TABLE}", flush=True)
    return mapping


def load_all_vectors_to_gpu(
    conn, vector_column: str, device: torch.device,
    shelfmark_by_ms: Dict[str, str],
) -> Tuple[torch.Tensor, List[Dict]]:
    """Load all vectors and metadata; return (N, D) tensor on device and list of metadata dicts."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT id, manuscript_id, image_path, picture_id,
                   num_visual_patches, num_glyphs, num_words,
                   {vector_column} AS vector
            FROM {GENIZA_IMAGE_LATENTS_TABLE}
            WHERE {vector_column} IS NOT NULL
            ORDER BY id
        """)
        rows = cur.fetchall()

    if not rows:
        raise ValueError(f"No vectors in {GENIZA_IMAGE_LATENTS_TABLE} for {vector_column}")

    vectors_list = []
    metadata_list = []
    n_rows = len(rows)
    parse_progress_interval = max(1, n_rows // 20)
    for i, row in enumerate(rows):
        if (i + 1) % parse_progress_interval == 0 or i == n_rows - 1:
            print(f"  Parsing vectors: {i + 1:,}/{n_rows:,} ({100.0 * (i + 1) / n_rows:.0f}%)", flush=True)
        vectors_list.append(_parse_vector(row["vector"]))
        ms_id = str(row["manuscript_id"])
        metadata_list.append({
            "id": row["id"],
            "manuscript_id": row["manuscript_id"],
            "shelfmark": shelfmark_by_ms.get(ms_id, ""),
            "image_path": row["image_path"],
            "picture_id": row.get("picture_id"),
            "num_visual_patches": row.get("num_visual_patches"),
            "num_glyphs": row.get("num_glyphs"),
            "num_words": row.get("num_words"),
        })

    arr = np.stack(vectors_list)  # float64, no rounding
    tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float64)
    tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
    print(f"Loaded {len(metadata_list):,} vectors (dim={tensor.shape[1]}, float64) to {device}", flush=True)
    return tensor, metadata_list


def _similarities_to_neighbor_lists(
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    batch_size: int,
    k_actual: int,
    num_total: int,
    query_manuscript_ids: List[str],
    query_shelfmarks: List[str],
    all_metadata: List[Dict],
) -> List[List[Dict]]:
    """Build list-of-neighbor-dicts from topk values/indices (CPU tensors)."""
    topk_values_cpu = topk_values.cpu().numpy()
    topk_indices_cpu = topk_indices.cpu().numpy()
    results = []
    for b in range(batch_size):
        q_sm = query_shelfmarks[b]
        neighbors = []
        for r in range(k_actual):
            idx = int(topk_indices_cpu[b, r])
            if idx < 0 or idx >= num_total:
                continue
            sim = float(topk_values_cpu[b, r])
            meta = all_metadata[idx]
            n_sm = meta.get("shelfmark", "")
            neighbors.append({
                "manuscript_id": meta["manuscript_id"],
                "shelfmark": n_sm,
                "image_path": meta["image_path"],
                "picture_id": meta.get("picture_id"),
                "neighbor_num_visual_patches": meta.get("num_visual_patches"),
                "neighbor_num_glyphs": meta.get("num_glyphs"),
                "neighbor_num_words": meta.get("num_words"),
                "similarity": sim,
                "distance": 1.0 - sim,
                "same_manuscript": (meta["manuscript_id"] == query_manuscript_ids[b]),
                "same_shelfmark": bool(q_sm) and bool(n_sm) and (n_sm == q_sm),
            })
        results.append(neighbors)
    return results


def batch_find_top_k_gpu(
    query_vectors: torch.Tensor,
    query_ids: List[int],
    query_manuscript_ids: List[str],
    query_shelfmarks: List[str],
    all_vectors: torch.Tensor,
    all_metadata: List[Dict],
    k: int,
    exclude_same_manuscript: bool,
    exclude_same_shelfmark: bool,
    device: torch.device,
) -> List[List[Dict]]:
    """Batch KNN on GPU; returns list of neighbor lists (one per query)."""
    batch_size = query_vectors.shape[0]
    num_total = all_vectors.shape[0]
    query_vectors = torch.nn.functional.normalize(query_vectors, p=2, dim=1)
    similarities = torch.mm(query_vectors, all_vectors.T)

    all_ids = [all_metadata[j]["id"] for j in range(num_total)]
    all_ms = [all_metadata[j]["manuscript_id"] for j in range(num_total)]
    all_sm = [all_metadata[j].get("shelfmark", "") for j in range(num_total)]
    exclude_mask = torch.zeros(batch_size, num_total, dtype=torch.bool, device=device)
    for i, (qid, qms, qsm) in enumerate(zip(query_ids, query_manuscript_ids, query_shelfmarks)):
        for j in range(num_total):
            if all_ids[j] == qid:
                exclude_mask[i, j] = True
            elif exclude_same_manuscript and all_ms[j] == qms:
                exclude_mask[i, j] = True
            elif exclude_same_shelfmark and qsm and all_sm[j] and all_sm[j] == qsm:
                exclude_mask[i, j] = True

    similarities = similarities.masked_fill(exclude_mask, float("-inf"))
    k_actual = min(k, num_total)
    topk_values, topk_indices = torch.topk(similarities, k=k_actual, dim=1)
    return _similarities_to_neighbor_lists(
        topk_values, topk_indices, batch_size, k_actual, num_total,
        query_manuscript_ids, query_shelfmarks, all_metadata,
    )


def batch_find_top_k_both_gpu(
    query_vectors: torch.Tensor,
    query_ids: List[int],
    query_manuscript_ids: List[str],
    query_shelfmarks: List[str],
    all_vectors: torch.Tensor,
    all_metadata: List[Dict],
    k: int,
    device: torch.device,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """
    One matmul per batch; returns (joins_edition_topk, overall_topk).
    joins_edition_topk excludes self, same-manuscript, and same-shelfmark neighbors;
    overall_topk excludes only self (keeps same_manuscript / same_shelfmark for eval).
    """
    batch_size = query_vectors.shape[0]
    num_total = all_vectors.shape[0]
    query_vectors = torch.nn.functional.normalize(query_vectors, p=2, dim=1)
    similarities = torch.mm(query_vectors, all_vectors.T)

    all_ids = [all_metadata[j]["id"] for j in range(num_total)]
    all_ms = [all_metadata[j]["manuscript_id"] for j in range(num_total)]
    all_sm = [all_metadata[j].get("shelfmark", "") for j in range(num_total)]
    self_mask = torch.zeros(batch_size, num_total, dtype=torch.bool, device=device)
    same_ms_mask = torch.zeros(batch_size, num_total, dtype=torch.bool, device=device)
    same_sm_mask = torch.zeros(batch_size, num_total, dtype=torch.bool, device=device)
    for i, (qid, qms, qsm) in enumerate(zip(query_ids, query_manuscript_ids, query_shelfmarks)):
        for j in range(num_total):
            if all_ids[j] == qid:
                self_mask[i, j] = True
            if all_ms[j] == qms:
                same_ms_mask[i, j] = True
            if qsm and all_sm[j] and all_sm[j] == qsm:
                same_sm_mask[i, j] = True

    k_actual = min(k, num_total)

    # Joins-edition: exclude self, same manuscript, and same shelfmark
    sim_joins = similarities.masked_fill(self_mask | same_ms_mask | same_sm_mask, float("-inf"))
    topk_joins_v, topk_joins_i = torch.topk(sim_joins, k=k_actual, dim=1)
    results_joins = _similarities_to_neighbor_lists(
        topk_joins_v, topk_joins_i, batch_size, k_actual, num_total,
        query_manuscript_ids, query_shelfmarks, all_metadata,
    )

    # Overall: exclude only self
    sim_overall = similarities.masked_fill(self_mask, float("-inf"))
    topk_overall_v, topk_overall_i = torch.topk(sim_overall, k=k_actual, dim=1)
    results_overall = _similarities_to_neighbor_lists(
        topk_overall_v, topk_overall_i, batch_size, k_actual, num_total,
        query_manuscript_ids, query_shelfmarks, all_metadata,
    )

    return results_joins, results_overall


def get_manuscript_ids_with_vectors(conn, vector_column: str) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT DISTINCT manuscript_id FROM {GENIZA_IMAGE_LATENTS_TABLE}
            WHERE {vector_column} IS NOT NULL ORDER BY manuscript_id
        """)
        return [str(r[0]) for r in cur.fetchall()]


def get_total_images_per_manuscript(conn, vector_column: str) -> Dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT manuscript_id, COUNT(*) AS cnt FROM {GENIZA_IMAGE_LATENTS_TABLE}
            WHERE {vector_column} IS NOT NULL GROUP BY manuscript_id
        """)
        return {str(mid): int(cnt) for mid, cnt in cur.fetchall()}


def get_query_rows_for_manuscript(conn, manuscript_id: str, vector_column: str) -> List[Dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT id, manuscript_id, picture_id, image_path,
                   num_visual_patches AS query_num_visual_patches,
                   num_glyphs AS query_num_glyphs,
                   num_words AS query_num_words,
                   {vector_column} AS query_vector
            FROM {GENIZA_IMAGE_LATENTS_TABLE}
            WHERE manuscript_id = %s AND {vector_column} IS NOT NULL
            ORDER BY id
        """, (manuscript_id,))
        return cur.fetchall()


def _count_relevant_per_shelfmark(metadata_list: List[Dict]) -> Dict[str, int]:
    """Image counts per shelfmark key (library||shelfmark_root); empty keys omitted."""
    counts: Dict[str, int] = {}
    for m in metadata_list:
        key = m.get("shelfmark", "")
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _average_precision(neighbors: List[Dict], rel_key: str, n_relevant: int) -> float | None:
    """AP@len(neighbors): sum of P@k at each relevant hit, divided by total relevant in corpus."""
    if n_relevant <= 0:
        return None
    hits = 0
    ap_sum = 0.0
    for k, n in enumerate(neighbors, 1):
        if n[rel_key]:
            hits += 1
            ap_sum += hits / k
    return ap_sum / n_relevant


def print_knn_stats(total: int, same_ms: int, diff_ms: int, same_sm: int | None = None, diff_sm: int | None = None):
    print(f"Total rows: {total:,}")
    print(f"Same manuscript matches: {same_ms:,}")
    print(f"Different manuscript matches: {diff_ms:,}")
    if same_sm is not None and diff_sm is not None:
        print(f"Same shelfmark matches: {same_sm:,}")
        print(f"Different shelfmark matches: {diff_sm:,}")


def run(
    db_config_path: str = CLUSTERING_DB_CONFIG_PATH,
    n_neighbors: int = CLUSTERING_N_NEIGHBORS,
    commit_interval: int = CLUSTERING_COMMIT_INTERVAL,
    use_full_vector: bool = False,
    eval_overall_nn: bool = True,
    overall_export_path: str | None = None,
    gpu_batch_size: int = 256,
    device: str | None = None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use geniza_top_neighbors.py for CPU.")
    dev = torch.device(device or "cuda")

    conn = get_db_connection(db_config_path)
    ensure_results_table(conn)
    ensure_overall_neighbors_table(conn)

    vector_column = _vector_column(use_full_vector)
    shelfmark_by_ms = load_manuscript_shelfmarks(conn)
    manuscript_ids = get_manuscript_ids_with_vectors(conn, vector_column)
    total_images_per_manuscript = get_total_images_per_manuscript(conn, vector_column)
    total_query_images = sum(total_images_per_manuscript.get(mid, 0) for mid in manuscript_ids)
    print(f"Manuscripts with vectors: {len(manuscript_ids):,}")
    missing_sm = sum(1 for mid in manuscript_ids if not shelfmark_by_ms.get(mid))
    if missing_sm:
        print(f"WARNING: {missing_sm:,} manuscript_ids are missing normalized_library or "
              f"shelfmark_root in {GENIZA_MANUSCRIPT_SHELFMARK_TABLE}; they will be treated "
              "as unique (no same-shelfmark filtering or flag for them).")
    print(f"Total query images: {total_query_images:,} (vector={vector_column}, k={n_neighbors}, GPU batch={gpu_batch_size})")

    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {GENIZA_KNN_RESULTS_TABLE}")
    conn.commit()

    print("Loading all vectors to GPU...", flush=True)
    all_vectors, all_metadata = load_all_vectors_to_gpu(conn, vector_column, dev, shelfmark_by_ms)
    relevant_per_sm = _count_relevant_per_shelfmark(all_metadata)

    # Collect all queries (DB fetch + parse per manuscript — can take a few minutes)
    print("Loading query rows and parsing vectors (per manuscript)...", flush=True)
    all_queries = []
    query_meta = []
    n_ms = len(manuscript_ids)
    next_report = 5000
    for ms_idx, manuscript_id in enumerate(manuscript_ids):
        rows = get_query_rows_for_manuscript(conn, manuscript_id, vector_column)
        total_in_ms = total_images_per_manuscript.get(manuscript_id, 0)
        sm = shelfmark_by_ms.get(str(manuscript_id), "")
        for q in rows:
            all_queries.append({
                "id": q["id"], "manuscript_id": q["manuscript_id"],
                "shelfmark": sm,
                "vector": _parse_vector(q["query_vector"]),
            })
            nq = len(all_queries)
            if nq >= next_report:
                print(f"  Queries loaded: {nq:,}", flush=True)
                next_report = ((nq // 5000) + 1) * 5000
            query_meta.append({
                "manuscript_id": q["manuscript_id"],
                "shelfmark": sm,
                "image_path": q["image_path"],
                "picture_id": q.get("picture_id"),
                "num_visual_patches": q.get("query_num_visual_patches"),
                "num_glyphs": q.get("query_num_glyphs"),
                "num_words": q.get("query_num_words"),
                "total_in_ms": total_in_ms,
            })

    num_queries = len(all_queries)
    print(f"  Queries loaded: {num_queries:,}", flush=True)
    num_batches = (num_queries + gpu_batch_size - 1) // gpu_batch_size
    print(f"Building query tensor for {num_queries:,} images ({num_batches:,} batches)...", flush=True)
    print("  Stacking vectors in memory...", flush=True)
    query_vectors = torch.from_numpy(np.stack([q["vector"] for q in all_queries]))
    print("  Copying to GPU (may take 1–2 min)...", flush=True)
    query_vectors = query_vectors.to(device=dev, dtype=torch.float64)
    print("  Normalizing...", flush=True)
    query_vectors = torch.nn.functional.normalize(query_vectors, p=2, dim=1)
    query_ids = [q["id"] for q in all_queries]
    query_ms_ids = [q["manuscript_id"] for q in all_queries]
    query_sm_ids = [q["shelfmark"] for q in all_queries]
    print(f"Starting KNN processing ({num_batches:,} batches)...", flush=True)

    # When eval_overall_nn: one matmul per batch, fill both tables (no duplicate similarity computation).
    # Otherwise: one matmul per batch, fill only joins-edition table.
    overall_rows = []
    overall_rows_for_db = []
    queries_with_same_ms_nn = queries_with_any_same_ms = 0
    queries_with_same_sm_nn = queries_with_any_same_sm = 0
    queries_with_same_ms_in_top5 = queries_with_same_sm_in_top5 = 0
    queries_with_same_ms_in_top10 = queries_with_same_sm_in_top10 = 0
    sum_ap_ms_at5 = sum_ap_sm_at5 = 0.0
    sum_ap_ms_at10 = sum_ap_sm_at10 = 0.0
    n_ap_ms_at5 = n_ap_sm_at5 = 0
    n_ap_ms_at10 = n_ap_sm_at10 = 0
    total_same_ms_hits = total_diff_ms_hits = 0
    total_same_sm_hits = total_diff_sm_hits = 0
    overall_total = 0

    rows_since_commit = 0
    progress_interval = max(1, num_batches // 20)  # print ~20 times over the run
    with conn.cursor() as cur:
        batch_iter = range(0, num_queries, gpu_batch_size)
        for batch_idx, batch_start in enumerate(batch_iter):
            if batch_idx % progress_interval == 0 or batch_idx == num_batches - 1:
                pct = 100.0 * (batch_idx + 1) / num_batches
                print(f"  Batch {batch_idx + 1:,}/{num_batches:,} ({pct:.0f}%) — queries {batch_start:,}-{min(batch_start + gpu_batch_size, num_queries):,}", flush=True)
            batch_end = min(batch_start + gpu_batch_size, num_queries)
            batch_q = query_vectors[batch_start:batch_end]
            batch_ids = query_ids[batch_start:batch_end]
            batch_ms = query_ms_ids[batch_start:batch_end]
            batch_sm = query_sm_ids[batch_start:batch_end]

            if eval_overall_nn:
                batch_results_joins, batch_results_overall = batch_find_top_k_both_gpu(
                    batch_q, batch_ids, batch_ms, batch_sm,
                    all_vectors, all_metadata, n_neighbors, dev,
                )
            else:
                batch_results_joins = batch_find_top_k_gpu(
                    batch_q, batch_ids, batch_ms, batch_sm,
                    all_vectors, all_metadata, n_neighbors,
                    exclude_same_manuscript=True,
                    exclude_same_shelfmark=True,
                    device=dev,
                )
                batch_results_overall = None

            # Write joins-edition table (different manuscript AND different shelfmark; schema unchanged)
            for i, neighbors in enumerate(batch_results_joins):
                qm = query_meta[batch_start + i]
                flat = [
                    (
                        qm["manuscript_id"],
                        qm["image_path"],
                        qm.get("picture_id") or os.path.basename(qm["image_path"]),
                        rank,
                        n["manuscript_id"],
                        n["image_path"],
                        n.get("picture_id") or os.path.basename(n["image_path"]),
                        float(n["similarity"]), float(n["distance"]),
                        False, qm["total_in_ms"],
                        qm.get("num_visual_patches"), qm.get("num_glyphs"), qm.get("num_words"),
                        n.get("neighbor_num_visual_patches"), n.get("neighbor_num_glyphs"), n.get("neighbor_num_words"),
                    )
                    for rank, n in enumerate(neighbors, 1)
                ]
                if flat:
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {GENIZA_KNN_RESULTS_TABLE}
                        (query_manuscript_id, query_image_path, query_image_name, neighbor_rank,
                         neighbor_manuscript_id, neighbor_image_path, neighbor_image_name,
                         similarity_score, distance_score, same_manuscript, total_images_in_query_manuscripts,
                         query_num_visual_patches, query_num_glyphs, query_num_words,
                         neighbor_num_visual_patches, neighbor_num_glyphs, neighbor_num_words)
                        VALUES %s
                        """,
                        flat, page_size=min(len(flat), 500),
                    )
                    rows_since_commit += len(flat)
                    if rows_since_commit >= commit_interval:
                        conn.commit()
                        rows_since_commit = 0

            # Accumulate overall table and stats (from same batch, no extra matmul)
            if eval_overall_nn and batch_results_overall is not None:
                for i, neighbors in enumerate(batch_results_overall):
                    qm = query_meta[batch_start + i]
                    overall_total += 1
                    same_ms_in = sum(1 for n in neighbors if n["same_manuscript"])
                    same_sm_in = sum(1 for n in neighbors if n["same_shelfmark"])
                    total_same_ms_hits += same_ms_in
                    total_diff_ms_hits += len(neighbors) - same_ms_in
                    total_same_sm_hits += same_sm_in
                    total_diff_sm_hits += len(neighbors) - same_sm_in
                    if same_ms_in > 0:
                        queries_with_any_same_ms += 1
                    if same_sm_in > 0:
                        queries_with_any_same_sm += 1
                    if neighbors and neighbors[0]["same_manuscript"]:
                        queries_with_same_ms_nn += 1
                    if neighbors and neighbors[0]["same_shelfmark"]:
                        queries_with_same_sm_nn += 1
                    top5 = neighbors[:5]
                    top10 = neighbors[:10]
                    if any(n["same_manuscript"] for n in top5):
                        queries_with_same_ms_in_top5 += 1
                    if any(n["same_shelfmark"] for n in top5):
                        queries_with_same_sm_in_top5 += 1
                    if any(n["same_manuscript"] for n in top10):
                        queries_with_same_ms_in_top10 += 1
                    if any(n["same_shelfmark"] for n in top10):
                        queries_with_same_sm_in_top10 += 1
                    n_rel_ms = total_images_per_manuscript.get(str(qm["manuscript_id"]), 0) - 1
                    sm_key = qm.get("shelfmark", "")
                    n_rel_sm = (relevant_per_sm.get(sm_key, 0) - 1) if sm_key else 0
                    ap_ms_5 = _average_precision(top5, "same_manuscript", n_rel_ms)
                    ap_sm_5 = _average_precision(top5, "same_shelfmark", n_rel_sm)
                    ap_ms_10 = _average_precision(top10, "same_manuscript", n_rel_ms)
                    ap_sm_10 = _average_precision(top10, "same_shelfmark", n_rel_sm)
                    if ap_ms_5 is not None:
                        sum_ap_ms_at5 += ap_ms_5
                        n_ap_ms_at5 += 1
                    if ap_sm_5 is not None:
                        sum_ap_sm_at5 += ap_sm_5
                        n_ap_sm_at5 += 1
                    if ap_ms_10 is not None:
                        sum_ap_ms_at10 += ap_ms_10
                        n_ap_ms_at10 += 1
                    if ap_sm_10 is not None:
                        sum_ap_sm_at10 += ap_sm_10
                        n_ap_sm_at10 += 1
                    for rank, n in enumerate(neighbors, 1):
                        rec = (
                            qm["manuscript_id"], qm["image_path"],
                            qm.get("picture_id") or os.path.basename(qm["image_path"]),
                            rank, n["manuscript_id"], n["image_path"],
                            n.get("picture_id") or os.path.basename(n["image_path"]),
                            float(n["similarity"]), float(n["distance"]),
                            bool(n["same_manuscript"]), bool(n["same_shelfmark"]),
                            qm["total_in_ms"],
                            qm.get("num_visual_patches"), qm.get("num_glyphs"), qm.get("num_words"),
                            n.get("neighbor_num_visual_patches"), n.get("neighbor_num_glyphs"), n.get("neighbor_num_words"),
                        )
                        overall_rows_for_db.append(rec)
                        if overall_export_path:
                            overall_rows.append({
                                "query_manuscript_id": rec[0], "query_image_path": rec[1],
                                "query_image_name": rec[2],
                                "neighbor_rank": rec[3], "neighbor_manuscript_id": rec[4],
                                "neighbor_image_path": rec[5], "neighbor_image_name": rec[6],
                                "similarity_score": rec[7], "distance_score": rec[8],
                                "same_manuscript": rec[9], "same_shelfmark": rec[10],
                                "total_images_in_query_manuscripts": rec[11],
                                "query_num_visual_patches": rec[12], "query_num_glyphs": rec[13],
                                "query_num_words": rec[14],
                                "neighbor_num_visual_patches": rec[15], "neighbor_num_glyphs": rec[16],
                                "neighbor_num_words": rec[17],
                            })

    conn.commit()
    print("KNN batch processing done. Tables written.", flush=True)

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT COUNT(*), COALESCE(SUM(CASE WHEN same_manuscript THEN 1 ELSE 0 END), 0)
            FROM {GENIZA_KNN_RESULTS_TABLE}
        """)
        total_rows, same_count = cur.fetchone()
    total_rows, same_count = int(total_rows), int(same_count)
    print(f"\nJoins-edition table ({GENIZA_KNN_RESULTS_TABLE}): {total_rows:,} rows")
    print(f"  same_manuscript in table: {same_count:,} (expected 0; same-shelfmark also excluded by filter)")
    print(f"  Different manuscript & shelfmark rows: {total_rows - same_count:,}")

    if eval_overall_nn:
        print(f"\nOverall top-{n_neighbors} (including same manuscript and same shelfmark)...")
        pct = lambda n, d: f"{100.0 * n / d:.1f}%" if d else "N/A"
        print()
        print("=" * 70)
        print(f"  NETWORK PERFORMANCE EVALUATION (top-{n_neighbors})")
        print("=" * 70)
        print(f"Total query images: {overall_total:,}")
        print(f"Rank-1 NN same manuscript: {queries_with_same_ms_nn:,} ({pct(queries_with_same_ms_nn, overall_total)})")
        print(f"Rank-1 NN same shelfmark:  {queries_with_same_sm_nn:,} ({pct(queries_with_same_sm_nn, overall_total)})")
        print(f"At least 1 same-manuscript in top-5:  {queries_with_same_ms_in_top5:,} ({pct(queries_with_same_ms_in_top5, overall_total)})")
        print(f"At least 1 same-shelfmark  in top-5:  {queries_with_same_sm_in_top5:,} ({pct(queries_with_same_sm_in_top5, overall_total)})")
        print(f"At least 1 same-manuscript in top-10: {queries_with_same_ms_in_top10:,} ({pct(queries_with_same_ms_in_top10, overall_total)})")
        print(f"At least 1 same-shelfmark  in top-10: {queries_with_same_sm_in_top10:,} ({pct(queries_with_same_sm_in_top10, overall_total)})")
        if n_neighbors not in (5, 10):
            print(f"At least 1 same-manuscript in top-{n_neighbors}: {queries_with_any_same_ms:,} ({pct(queries_with_any_same_ms, overall_total)})")
            print(f"At least 1 same-shelfmark  in top-{n_neighbors}: {queries_with_any_same_sm:,} ({pct(queries_with_any_same_sm, overall_total)})")
        map_ms_5 = sum_ap_ms_at5 / n_ap_ms_at5 if n_ap_ms_at5 else float("nan")
        map_sm_5 = sum_ap_sm_at5 / n_ap_sm_at5 if n_ap_sm_at5 else float("nan")
        map_ms_10 = sum_ap_ms_at10 / n_ap_ms_at10 if n_ap_ms_at10 else float("nan")
        map_sm_10 = sum_ap_sm_at10 / n_ap_sm_at10 if n_ap_sm_at10 else float("nan")
        print(f"mAP@5  same manuscript: {map_ms_5:.4f}  ({n_ap_ms_at5:,} queries with ≥1 other image in manuscript)")
        print(f"mAP@5  same shelfmark:  {map_sm_5:.4f}  ({n_ap_sm_at5:,} queries with ≥1 other image in shelfmark)")
        print(f"mAP@10 same manuscript: {map_ms_10:.4f}  ({n_ap_ms_at10:,} queries with ≥1 other image in manuscript)")
        print(f"mAP@10 same shelfmark:  {map_sm_10:.4f}  ({n_ap_sm_at10:,} queries with ≥1 other image in shelfmark)")
        total_hits = total_same_ms_hits + total_diff_ms_hits
        print(f"Same-manuscript hits: {total_same_ms_hits:,} ({pct(total_same_ms_hits, total_hits)})")
        print(f"Same-shelfmark hits:  {total_same_sm_hits:,} ({pct(total_same_sm_hits, total_hits)})")
        print(f"Different-manuscript hits: {total_diff_ms_hits:,} ({pct(total_diff_ms_hits, total_hits)})")
        print(f"Different-shelfmark hits:  {total_diff_sm_hits:,} ({pct(total_diff_sm_hits, total_hits)})")

        if overall_rows_for_db:
            print(f"Writing {len(overall_rows_for_db):,} overall rows to {GENIZA_OVERALL_NEIGHBORS_TABLE}...", flush=True)
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {GENIZA_OVERALL_NEIGHBORS_TABLE}")
            conn.commit()
            batch_size_db = min(commit_interval, 500)
            num_write_batches = (len(overall_rows_for_db) + batch_size_db - 1) // batch_size_db
            with conn.cursor() as cur:
                for wi, i in enumerate(range(0, len(overall_rows_for_db), batch_size_db)):
                    if wi % 20 == 0 or wi == num_write_batches - 1:
                        print(f"  Write overall: batch {wi + 1:,}/{num_write_batches:,}", flush=True)
                    batch = overall_rows_for_db[i : i + batch_size_db]
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {GENIZA_OVERALL_NEIGHBORS_TABLE}
                        (query_manuscript_id, query_image_path, query_image_name, neighbor_rank,
                         neighbor_manuscript_id, neighbor_image_path, neighbor_image_name,
                         similarity_score, distance_score, same_manuscript, same_shelfmark, total_images_in_query_manuscripts,
                         query_num_visual_patches, query_num_glyphs, query_num_words,
                         neighbor_num_visual_patches, neighbor_num_glyphs, neighbor_num_words)
                        VALUES %s
                        """,
                        batch, page_size=len(batch),
                    )
            conn.commit()
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT COUNT(*),
                           SUM(CASE WHEN same_manuscript THEN 1 ELSE 0 END),
                           SUM(CASE WHEN NOT same_manuscript THEN 1 ELSE 0 END),
                           SUM(CASE WHEN same_shelfmark THEN 1 ELSE 0 END),
                           SUM(CASE WHEN NOT same_shelfmark THEN 1 ELSE 0 END)
                    FROM {GENIZA_OVERALL_NEIGHBORS_TABLE}
                """)
                tot, same_ms_db, diff_ms_db, same_sm_db, diff_sm_db = cur.fetchone()
            print(f"\nWrote {GENIZA_OVERALL_NEIGHBORS_TABLE}:")
            print_knn_stats(
                int(tot), int(same_ms_db or 0), int(diff_ms_db or 0),
                int(same_sm_db or 0), int(diff_sm_db or 0),
            )

        if overall_export_path and overall_rows:
            parent = os.path.dirname(overall_export_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            print(f"Writing overall Excel to {overall_export_path}...", flush=True)
            df = pd.DataFrame(overall_rows)
            df.to_excel(overall_export_path, index=False, engine="openpyxl")
            print(f"Overall Excel saved: {overall_export_path}", flush=True)
            print_knn_stats(
                len(df),
                int(df["same_manuscript"].sum()),
                len(df) - int(df["same_manuscript"].sum()),
                int(df["same_shelfmark"].sum()),
                len(df) - int(df["same_shelfmark"].sum()),
            )

    del query_vectors, all_vectors
    torch.cuda.empty_cache()
    conn.close()
    print(
        f"\nDone. Joins-edition rows in {GENIZA_KNN_RESULTS_TABLE} "
        f"(different manuscript AND different shelfmark)."
        + (f"\nNetwork-evaluation rows in {GENIZA_OVERALL_NEIGHBORS_TABLE} "
           f"(all neighbors, with same_manuscript/same_shelfmark flags)." if eval_overall_nn else ""),
        flush=True,
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description="GPU-accelerated top-K neighbours for geniza (same outputs as geniza_top_neighbors.py).")
    p.add_argument("--db-config", type=str, default=CLUSTERING_DB_CONFIG_PATH)
    p.add_argument("--n_neighbors", type=int, default=CLUSTERING_N_NEIGHBORS)
    p.add_argument("--commit_interval", type=int, default=CLUSTERING_COMMIT_INTERVAL)
    p.add_argument("--vector", type=str, choices=[VECTOR_SEARCH, VECTOR_FULL], default=VECTOR_FULL,
                   help="'search' = PCA, 'full' = full latent (default: full)")
    p.add_argument("--export", type=str, default="", help="Export KNN table to this Excel path after run.")
    p.add_argument("--skip-overall-nn-eval", action="store_true", help="Skip overall NN evaluation.")
    p.add_argument("--export-overall", type=str, default="", help="Export overall neighbors to this Excel path.")
    p.add_argument("--gpu-batch-size", type=int, default=256, help="Batch size for GPU KNN.")
    p.add_argument("--device", type=str, default=None, help="CUDA device (e.g. cuda:0). Default: cuda.")
    args = p.parse_args()

    run(
        db_config_path=args.db_config,
        n_neighbors=args.n_neighbors,
        commit_interval=args.commit_interval,
        use_full_vector=(args.vector == VECTOR_FULL),
        eval_overall_nn=not args.skip_overall_nn_eval,
        overall_export_path=args.export_overall or None,
        gpu_batch_size=args.gpu_batch_size,
        device=args.device,
    )
    if args.export:
        from geniza_top_neighbors import export_to_excel  # noqa: E402
        export_to_excel(args.db_config, args.export)


if __name__ == "__main__":
    main()
