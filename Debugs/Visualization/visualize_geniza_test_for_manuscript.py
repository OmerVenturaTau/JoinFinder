#!/usr/bin/env python3
"""
Run visualize_all_for_manuscript.py for every image in the Geniza test list.

The default test list is Drafts/DataTables/geniza_test_case.txt, which contains
one manuscript_id per line. For each manuscript, this script selects all Geniza
images that have image_path and xml_path in the DB, then invokes:

  python Debugs/Visualization/visualize_all_for_manuscript.py --image <manuscript_id>/<picture_id> --mode geniza

Examples:
  python Debugs/Visualization/visualize_geniza_test_for_manuscript.py --dry-run
  python Debugs/Visualization/visualize_geniza_test_for_manuscript.py --skip-existing --skip-glyph-tighten-comparison
  python Debugs/Visualization/visualize_geniza_test_for_manuscript.py --limit-manuscripts 2 --limit-images 5
"""

from __future__ import annotations

import argparse
import configparser
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system import (  # noqa: E402
    DB_CONFIG_PATH,
    GENIZA_IMAGE_INFORMATION_TABLE,
    GENIZA_IMAGE_LATENTS_TABLE,
)


DEFAULT_MANUSCRIPT_LIST = PROJECT_ROOT / "Drafts" / "DataTables" / "geniza_test_case.txt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "Debugs" / "Visualization" / "outputs"
VISUALIZE_SCRIPT = PROJECT_ROOT / "Debugs" / "Visualization" / "visualize_all_for_manuscript.py"


@dataclass(frozen=True)
class GenizaImage:
    manuscript_id: str
    picture_id: str
    image_path: str
    xml_path: str

    @property
    def image_arg(self) -> str:
        return f"{self.manuscript_id}/{self.picture_id}"


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _read_manuscript_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Manuscript list not found: {path}")

    manuscript_ids: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        manuscript_id = line.split()[0].strip()
        if manuscript_id and manuscript_id not in seen:
            manuscript_ids.append(manuscript_id)
            seen.add(manuscript_id)
    if not manuscript_ids:
        raise ValueError(f"No manuscript IDs found in: {path}")
    return manuscript_ids


def _load_db_config(db_config_path: str | Path) -> Any:
    cfg_path = _resolve_project_path(db_config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"DB config not found: {cfg_path}")

    config = configparser.ConfigParser()
    config.read(cfg_path)
    if "postgresql" not in config:
        raise ValueError(f"Missing [postgresql] section in DB config: {cfg_path}")
    return config["postgresql"]


def _get_db_connection(db_config_path: str | Path):
    db = _load_db_config(db_config_path)
    from psycopg2 import connect as pg_connect

    return pg_connect(
        host=db["host"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
        port=db.get("port", 5432),
    )


def _fetch_images_for_manuscript(conn: Any, manuscript_id: str, *, require_counts: bool) -> list[GenizaImage]:
    if require_counts:
        query = f"""
            SELECT
                gii.manuscript_id,
                gii.picture_id,
                gii.image_path,
                gii.xml_path
            FROM {GENIZA_IMAGE_INFORMATION_TABLE} gii
            JOIN {GENIZA_IMAGE_LATENTS_TABLE} gil
              ON gil.manuscript_id = gii.manuscript_id
             AND gil.picture_id = gii.picture_id
            WHERE gii.manuscript_id = %s
              AND gii.image_path IS NOT NULL
              AND gii.xml_path IS NOT NULL
              AND COALESCE(gil.num_visual_patches, 0) > 0
              AND COALESCE(gil.num_glyphs, 0) > 0
            ORDER BY gii.picture_id
        """
    else:
        query = f"""
            SELECT
                manuscript_id,
                picture_id,
                image_path,
                xml_path
            FROM {GENIZA_IMAGE_INFORMATION_TABLE}
            WHERE manuscript_id = %s
              AND image_path IS NOT NULL
              AND xml_path IS NOT NULL
            ORDER BY picture_id
        """

    with conn.cursor() as cur:
        cur.execute(query, (manuscript_id,))
        rows = cur.fetchall()

    return [
        GenizaImage(
            manuscript_id=str(row[0]),
            picture_id=str(row[1]),
            image_path=str(row[2]),
            xml_path=str(row[3]),
        )
        for row in rows
    ]


def _fetch_geniza_test_images(
    db_config_path: str | Path,
    manuscript_ids: Iterable[str],
    *,
    require_counts: bool,
) -> list[GenizaImage]:
    conn = _get_db_connection(db_config_path)
    try:
        images: list[GenizaImage] = []
        for manuscript_id in manuscript_ids:
            rows = _fetch_images_for_manuscript(conn, manuscript_id, require_counts=require_counts)
            print(f"{manuscript_id}: {len(rows)} image(s)")
            images.extend(rows)
        return images
    finally:
        conn.close()


def _expected_summary_path(out_dir: Path, image: GenizaImage) -> Path:
    image_stem = Path(image.image_path).stem
    return out_dir / image.manuscript_id / image_stem / "summary.txt"


def _build_command(args: argparse.Namespace, image: GenizaImage) -> list[str]:
    cmd = [
        args.python,
        str(VISUALIZE_SCRIPT),
        "--image",
        image.image_arg,
        "--mode",
        "geniza",
        "--db-config",
        str(_resolve_project_path(args.db_config)),
        "--out-dir",
        str(Path(args.out_dir).resolve()),
    ]
    if args.no_require_counts:
        cmd.append("--no-require-counts")
    if args.skip_glyph_tighten_comparison:
        cmd.append("--skip-glyph-tighten-comparison")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manuscript-list",
        default=str(DEFAULT_MANUSCRIPT_LIST),
        help="Text file with one manuscript_id per line.",
    )
    parser.add_argument("--db-config", default=DB_CONFIG_PATH, help="Path to db_config.ini.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output root directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child runs.")
    parser.add_argument(
        "--no-require-counts",
        action="store_true",
        help="Include rows without positive latent patch/glyph counts.",
    )
    parser.add_argument(
        "--skip-glyph-tighten-comparison",
        action="store_true",
        help="Pass through to visualize_all_for_manuscript.py to avoid extra glyph extraction runs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose summary.txt already exists in the output tree.",
    )
    parser.add_argument("--limit-manuscripts", type=int, default=None, help="Only process the first N manuscripts.")
    parser.add_argument("--limit-images", type=int, default=None, help="Only process the first N selected images.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed child process instead of continuing.",
    )
    args = parser.parse_args()

    if not VISUALIZE_SCRIPT.exists():
        raise FileNotFoundError(f"Missing visualization script: {VISUALIZE_SCRIPT}")

    manuscript_list = _resolve_project_path(args.manuscript_list)
    manuscript_ids = _read_manuscript_ids(manuscript_list)
    if args.limit_manuscripts is not None:
        manuscript_ids = manuscript_ids[: args.limit_manuscripts]

    out_dir = Path(args.out_dir).resolve()
    images = _fetch_geniza_test_images(
        args.db_config,
        manuscript_ids,
        require_counts=not bool(args.no_require_counts),
    )
    if args.limit_images is not None:
        images = images[: args.limit_images]

    if args.skip_existing:
        before = len(images)
        images = [image for image in images if not _expected_summary_path(out_dir, image).exists()]
        print(f"Skipping existing outputs: {before - len(images)} skipped, {len(images)} remaining")

    print(f"Selected {len(images)} image(s) from {len(manuscript_ids)} manuscript(s).")
    if not images:
        return 0

    failures: list[tuple[GenizaImage, int]] = []
    for index, image in enumerate(images, start=1):
        cmd = _build_command(args, image)
        print(f"\n[{index}/{len(images)}] {image.image_arg}")
        print(" ".join(cmd))
        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append((image, result.returncode))
            print(f"FAILED ({result.returncode}): {image.image_arg}", file=sys.stderr)
            if args.stop_on_error:
                break

    if failures:
        print("\nFailed images:", file=sys.stderr)
        for image, returncode in failures:
            print(f"  {image.image_arg} -> exit {returncode}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
