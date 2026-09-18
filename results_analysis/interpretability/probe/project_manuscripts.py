#!/usr/bin/env python3
"""Project manuscript images into modality-specific and shared latent spaces."""

from __future__ import annotations

import argparse
import configparser
import csv
import logging
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import psycopg2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import MultiModal  # noqa: E402
from system import (  # noqa: E402
    DB_CONFIG_PATH,
    GLYPH_INPUT_ALPHABET,
    GLYPHS_PER_CLASS,
    INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE,
    MAX_TILES_EVAL,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    PRETRAIN_TABLE_NAME,
    TILE_STRIDE,
)
from train.dataset import ManuscriptDataset, tile_collate_with_padding  # noqa: E402


LOGGER = logging.getLogger("interpretability.project")


def safe_identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(value)) is None:
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return str(value)


def get_connection(config_path: str):
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = configparser.ConfigParser()
    if not config.read(path):
        raise FileNotFoundError(f"Database config not found: {path}")
    db = config["postgresql"]
    return psycopg2.connect(
        host=db["host"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
        port=db.get("port", 5432),
    )


def read_manuscript_file(path: str) -> list[str]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Manuscript list not found: {source}")
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "manuscript_id" not in reader.fieldnames:
                raise ValueError("CSV manuscript list must contain a manuscript_id column")
            values = [str(row["manuscript_id"]).strip() for row in reader]
    else:
        values = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]
    return sorted({value for value in values if value})


def resolve_manuscript_ids(
    manuscript_ids: Optional[Iterable[str]], manuscript_list: Optional[str]
) -> list[str]:
    values = {str(value).strip() for value in (manuscript_ids or []) if str(value).strip()}
    if manuscript_list:
        values.update(read_manuscript_file(manuscript_list))
    return sorted(values)


def load_source_rows(
    conn,
    *,
    source_table: str,
    manuscript_ids: list[str],
    limit: int = 0,
) -> list[dict]:
    from psycopg2.extras import RealDictCursor

    table = safe_identifier(source_table)
    where = "WHERE image_path IS NOT NULL AND image_path <> ''"
    params: list[object] = []
    if manuscript_ids:
        where += " AND manuscript_id = ANY(%s)"
        params.append(manuscript_ids)
    limit_sql = ""
    if limit > 0:
        limit_sql = " LIMIT %s"
        params.append(int(limit))
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT manuscript_id, picture_id, parent_directory, page_number,
                   image_path, xml_path, is_oriental, dataset_split
            FROM {table}
            {where}
            ORDER BY manuscript_id, parent_directory, picture_id
            {limit_sql}
            """,
            tuple(params),
        )
        rows = [dict(row) for row in cur.fetchall()]
    # A duplicate source row would otherwise make projection accounting and the
    # table primary key disagree. Keep the first deterministic occurrence.
    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(os.path.normpath(str(row["image_path"])), row)
    return list(unique.values())


def _strip_module_prefix(state_dict: dict) -> dict:
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint_model(checkpoint_path: str, device: torch.device):
    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state-dict or training-checkpoint mapping")
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    state_dict = _strip_module_prefix(state_dict)
    config = dict(checkpoint.get("model_config") or {})
    required = ("num_classes", "tile_size", "use_visual_mod", "use_char_mod", "use_word_mod")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            "Checkpoint model_config is missing projection-critical fields: "
            + ", ".join(missing)
        )
    glyph_weight = state_dict.get("glyph_branch.token_enrichment.char_class_embed.weight")
    if bool(config["use_char_mod"]):
        if not torch.is_tensor(glyph_weight) or glyph_weight.ndim != 2:
            raise ValueError("Checkpoint is missing the glyph character-class embedding")
        num_glyph_classes = int(glyph_weight.shape[0])
        alphabet = config.get("glyph_input_alphabet")
        if alphabet is None:
            # Older checkpoints did not record the alphabet. The repository's
            # alphabet evolves by appending letters, so the embedding row count
            # identifies the compatible prefix; the final row is unknown/other.
            alphabet = "".join(GLYPH_INPUT_ALPHABET[: num_glyph_classes - 1])
        elif isinstance(alphabet, (list, tuple)):
            alphabet = "".join(str(value) for value in alphabet)
        else:
            alphabet = str(alphabet)
        if len(alphabet) + 1 != num_glyph_classes:
            raise ValueError(
                "Checkpoint glyph alphabet is incompatible with its character embedding: "
                f"alphabet={alphabet!r}, embedding_rows={num_glyph_classes}"
            )
    else:
        alphabet = ""
        num_glyph_classes = 1
    config["glyph_input_alphabet"] = alphabet
    config["num_glyph_classes"] = num_glyph_classes
    model = MultiModal(
        num_classes=int(config["num_classes"]),
        tile_size=int(config["tile_size"]),
        encoder_pretrained=False,
        num_glyph_classes=num_glyph_classes,
        symmetric_modalities_from_instance=True,
        use_visual_mod=bool(config["use_visual_mod"]),
        use_char_mod=bool(config["use_char_mod"]),
        use_word_mod=bool(config["use_word_mod"]),
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture is incompatible with the active model configuration. "
            "Projection refuses a partial load."
        ) from exc
    model.to(device).eval()
    return model, config, checkpoint.get("epoch"), path


def vector_text(value: Optional[torch.Tensor]) -> Optional[str]:
    if value is None:
        return None
    tensor = value.detach().cpu().float().flatten()
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise RuntimeError("Projection produced an empty or non-finite vector")
    return "[" + ",".join(format(float(item), ".9g") for item in tensor.tolist()) + "]"


def _branch_value(
    aux: dict,
    name: str,
    sample_index: int,
    evidence_count: int,
    enabled: bool,
) -> tuple[Optional[int], Optional[str]]:
    if not enabled or evidence_count <= 0:
        return None, None
    value = aux.get(name)
    if not torch.is_tensor(value) or value.ndim != 2 or sample_index >= value.shape[0]:
        raise RuntimeError(f"Model did not return the enabled {name!r} latent")
    return int(value.shape[1]), vector_text(value[sample_index])


def ensure_destination_exists(conn, table_name: str) -> None:
    table = safe_identifier(table_name)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        if cur.fetchone()[0] is None:
            raise RuntimeError(
                f"Destination table {table!r} does not exist. Run "
                "Drafts/create_interpretability_experiment_vectors_table.py --apply first."
            )


def project(
    *,
    checkpoint: str,
    projection_run: str,
    manuscript_ids: list[str],
    source_table: str,
    vectors_table: str,
    db_config: str,
    batch_size: int,
    num_workers: int,
    limit: int,
    device_name: Optional[str],
) -> int:
    from psycopg2.extras import execute_values

    if not projection_run.strip():
        raise ValueError("projection_run cannot be empty")
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, model_config, checkpoint_epoch, checkpoint_path = load_checkpoint_model(
        checkpoint, device
    )
    conn = get_connection(db_config)
    try:
        ensure_destination_exists(conn, vectors_table)
        source_rows = load_source_rows(
            conn,
            source_table=source_table,
            manuscript_ids=manuscript_ids,
            limit=limit,
        )
        if not source_rows:
            raise RuntimeError("No source images matched the manuscript selection")

        paths = [str(row["image_path"]) for row in source_rows]
        labels = [str(row["manuscript_id"]) for row in source_rows]
        xml_paths = [row.get("xml_path") or None for row in source_rows]
        label2idx = {label: index for index, label in enumerate(sorted(set(labels)))}
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ]
        )
        dataset = ManuscriptDataset(
            paths,
            labels,
            transform,
            label2idx,
            xml_paths=xml_paths,
            patch_size=int(model_config["tile_size"]),
            stride=TILE_STRIDE,
            max_tiles_per_image=MAX_TILES_EVAL,
            split="probe",
            use_visual_mod=bool(model_config["use_visual_mod"]),
            use_char_mod=bool(model_config["use_char_mod"]),
            use_word_mod=bool(model_config["use_word_mod"]),
            glyph_input_alphabet=model_config["glyph_input_alphabet"],
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=partial(
                tile_collate_with_padding,
                glyph_input_alphabet=model_config["glyph_input_alphabet"],
                glyphs_per_class=GLYPHS_PER_CLASS,
            ),
        )
        source_by_path = {
            os.path.normpath(str(row["image_path"])): row for row in source_rows
        }
        table = safe_identifier(vectors_table)
        inserted = 0
        coverage = {"shared": 0, "tile": 0, "glyph": 0, "word": 0}
        seen: set[str] = set()
        enabled = {
            "tile": bool(model_config["use_visual_mod"]),
            "glyph": bool(model_config["use_char_mod"]),
            "word": bool(model_config["use_word_mod"]),
        }

        with torch.no_grad():
            for batch in tqdm(loader, desc="Projecting manuscripts"):
                if batch is None:
                    continue
                (
                    tiles,
                    tile_mask,
                    tile_coords,
                    tile_segments,
                    glyphs,
                    glyph_mask,
                    glyph_coords,
                    glyph_segments,
                    char_class_ids,
                    _char_metadata,
                    words,
                    word_metadata,
                    _labels,
                    batch_paths,
                ) = batch
                tiles = tiles.to(device, non_blocking=True)
                tile_mask = tile_mask.to(device, non_blocking=True)
                tile_coords = tile_coords.to(device, non_blocking=True)
                tile_segments = tile_segments.to(device, non_blocking=True)
                glyphs = glyphs.to(device, non_blocking=True)
                glyph_mask = glyph_mask.to(device, non_blocking=True)
                glyph_coords = glyph_coords.to(device, non_blocking=True)
                glyph_segments = glyph_segments.to(device, non_blocking=True)
                char_class_ids = char_class_ids.to(device, non_blocking=True)
                element_indices = torch.arange(tiles.shape[0], device=device)
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    shared, aux = model.forward_features(
                        tiles=tiles,
                        tile_coords=tile_coords,
                        tile_valid_mask=tile_mask,
                        tile_page_segments=tile_segments,
                        glyph_patches=glyphs,
                        glyph_coords=glyph_coords,
                        glyph_valid_mask=glyph_mask,
                        glyph_page_segments=glyph_segments,
                        char_class_ids=char_class_ids,
                        words=words,
                        word_metadata=word_metadata,
                        paths=list(batch_paths),
                        batch_element_indices=element_indices,
                        device=device,
                        return_aux_latents=True,
                    )
                shared = shared.float()
                if shared.ndim != 2 or not torch.isfinite(shared).all():
                    raise RuntimeError("Projection produced invalid shared latents")
                availability = aux.get("modality_available") if isinstance(aux, dict) else None
                availability_names = tuple(
                    getattr(getattr(model, "fusion", None), "enabled_modalities", ())
                )

                def is_available(name: str, sample_index: int) -> bool:
                    if not enabled[name]:
                        return False
                    if not torch.is_tensor(availability) or name not in availability_names:
                        return True
                    modality_index = availability_names.index(name)
                    return bool(availability[sample_index, modality_index].item())

                values = []
                for index, image_path in enumerate(batch_paths):
                    key = os.path.normpath(str(image_path))
                    row = source_by_path.get(key)
                    if row is None:
                        raise RuntimeError(f"Projected path is absent from source rows: {image_path}")
                    num_tiles = int(tile_mask[index].sum().item())
                    num_glyphs = int(glyph_mask[index].sum().item())
                    num_words = len(words[index]) if words and index < len(words) else 0
                    tile_dim, tile_vector = _branch_value(
                        aux, "tile", index, num_tiles, is_available("tile", index)
                    )
                    glyph_dim, glyph_vector = _branch_value(
                        aux, "glyph", index, num_glyphs, is_available("glyph", index)
                    )
                    word_dim, word_vector = _branch_value(
                        aux, "word", index, num_words, is_available("word", index)
                    )
                    coverage["shared"] += 1
                    coverage["tile"] += int(tile_vector is not None)
                    coverage["glyph"] += int(glyph_vector is not None)
                    coverage["word"] += int(word_vector is not None)
                    seen.add(key)
                    values.append(
                        (
                            projection_run,
                            str(checkpoint_path),
                            int(checkpoint_epoch) if checkpoint_epoch is not None else None,
                            source_table,
                            str(row["manuscript_id"]),
                            row.get("picture_id"),
                            row.get("parent_directory"),
                            str(row.get("page_number") or ""),
                            str(row["image_path"]),
                            row.get("xml_path"),
                            row.get("is_oriental"),
                            row.get("dataset_split"),
                            int(shared.shape[1]),
                            vector_text(shared[index]),
                            tile_dim,
                            tile_vector,
                            glyph_dim,
                            glyph_vector,
                            word_dim,
                            word_vector,
                            num_tiles,
                            num_glyphs,
                            num_words,
                        )
                    )
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {table} (
                            projection_run, checkpoint_path, checkpoint_epoch, source_table,
                            manuscript_id, picture_id, parent_directory, page_number,
                            image_path, xml_path, is_oriental, source_dataset_split,
                            shared_dim, shared_vector, tile_dim, tile_vector,
                            glyph_dim, glyph_vector, word_dim, word_vector,
                            num_tiles, num_glyphs, num_words
                        ) VALUES %s
                        ON CONFLICT (projection_run, image_path) DO UPDATE SET
                            checkpoint_path = EXCLUDED.checkpoint_path,
                            checkpoint_epoch = EXCLUDED.checkpoint_epoch,
                            source_table = EXCLUDED.source_table,
                            manuscript_id = EXCLUDED.manuscript_id,
                            picture_id = EXCLUDED.picture_id,
                            parent_directory = EXCLUDED.parent_directory,
                            page_number = EXCLUDED.page_number,
                            xml_path = EXCLUDED.xml_path,
                            is_oriental = EXCLUDED.is_oriental,
                            source_dataset_split = EXCLUDED.source_dataset_split,
                            shared_dim = EXCLUDED.shared_dim,
                            shared_vector = EXCLUDED.shared_vector,
                            tile_dim = EXCLUDED.tile_dim,
                            tile_vector = EXCLUDED.tile_vector,
                            glyph_dim = EXCLUDED.glyph_dim,
                            glyph_vector = EXCLUDED.glyph_vector,
                            word_dim = EXCLUDED.word_dim,
                            word_vector = EXCLUDED.word_vector,
                            num_tiles = EXCLUDED.num_tiles,
                            num_glyphs = EXCLUDED.num_glyphs,
                            num_words = EXCLUDED.num_words,
                            projected_at = NOW()
                        """,
                        values,
                        template=(
                            "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                            "%s,%s::vector,%s,%s::vector,%s,%s::vector,%s,%s::vector,%s,%s,%s)"
                        ),
                        page_size=max(1, len(values)),
                    )
                conn.commit()
                inserted += len(values)

        missing = sorted(set(source_by_path) - seen)
        if missing:
            raise RuntimeError(f"Projection missed {len(missing)} source images; first: {missing[:3]}")
        LOGGER.info("Stored %d rows in %s for run %s", inserted, table, projection_run)
        LOGGER.info("Latent coverage: %s", coverage)
        return inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", default=None, help="Defaults to checkpoint filename stem.")
    parser.add_argument("--manuscript-id", action="append", default=[])
    parser.add_argument("--manuscript-list", default=None)
    parser.add_argument("--source-table", default=PRETRAIN_TABLE_NAME)
    parser.add_argument("--vectors-table", default=INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE)
    parser.add_argument("--db-config", default=DB_CONFIG_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test row limit.")
    parser.add_argument("--device", default=None, help="For example cuda, cuda:1, or cpu.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manuscript_ids = resolve_manuscript_ids(args.manuscript_id, args.manuscript_list)
    run_name = args.run_name or Path(args.checkpoint).stem
    project(
        checkpoint=args.checkpoint,
        projection_run=run_name,
        manuscript_ids=manuscript_ids,
        source_table=args.source_table,
        vectors_table=args.vectors_table,
        db_config=args.db_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
