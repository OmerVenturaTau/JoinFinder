#!/usr/bin/env python3
"""Count Hebrew letters and plot raw OCR confidence in ALTO collections.

The analysis deliberately reads every ALTO glyph without a confidence filter.

Examples
--------
Database-backed, balanced by manuscript::

    python Debugs/LetterAppearance/analyze_geniza_letters.py \
        --manuscripts 50 --pages-per-manuscript 3

Reproducible input list (columns: manuscript_id, xml_path)::

    python Debugs/LetterAppearance/analyze_geniza_letters.py --input-csv pages.csv
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system import (  # noqa: E402
    CLUSTERING_DB_CONFIG_PATH,
    GENIZA_IMAGE_INFORMATION_TABLE,
    HEBREW_ALPHABET,
    OCR_GLYPH_CONFIDENCE_THRESHOLD,
    PRETRAIN_TABLE_NAME,
)
from utilities.VisionModule.alto_parser import parse_alto_strings  # noqa: E402


FINAL_TO_BASE = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}


def _db_connection(config_path: str):
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = configparser.ConfigParser()
    if not config.read(path):
        raise FileNotFoundError(f"DB config not found: {path}")
    section = "postgresql" if "postgresql" in config else "database"
    db = config[section]
    import psycopg2

    return psycopg2.connect(
        host=db["host"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
        port=db.get("port", 5432),
    )


def load_db_pages(
    *, config_path: str, source: str, manuscripts: int, pages_per_manuscript: int, seed: int
) -> list[dict[str, str]]:
    """Select manuscripts first, then equally many deterministic pages from each."""
    tables = {
        "geniza": GENIZA_IMAGE_INFORMATION_TABLE,
        "pretraining": PRETRAIN_TABLE_NAME,
    }
    table = tables[source]
    query = f"""
        WITH source_rows AS (
            SELECT DISTINCT manuscript_id, picture_id, page_number, xml_path
            FROM {table}
            WHERE manuscript_id IS NOT NULL
              AND xml_path IS NOT NULL AND xml_path <> ''
        ), chosen_manuscripts AS (
            SELECT manuscript_id, md5(manuscript_id::text || %s) AS sample_key
            FROM source_rows
            GROUP BY manuscript_id
            ORDER BY sample_key
            LIMIT %s
        ), ranked_pages AS (
            SELECT i.manuscript_id, i.picture_id, i.page_number, i.xml_path,
                   m.sample_key,
                   ROW_NUMBER() OVER (
                       PARTITION BY i.manuscript_id
                       ORDER BY md5(i.xml_path || %s), i.xml_path
                   ) AS page_rank
            FROM source_rows i
            JOIN chosen_manuscripts m USING (manuscript_id)
            WHERE i.xml_path IS NOT NULL AND i.xml_path <> ''
        )
        SELECT manuscript_id, picture_id, page_number, xml_path
        FROM ranked_pages
        WHERE page_rank <= %s
        ORDER BY sample_key, page_rank
    """
    with _db_connection(config_path) as conn, conn.cursor() as cur:
        cur.execute(query, (str(seed), manuscripts, str(seed), pages_per_manuscript))
        names = [desc.name for desc in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]


def load_csv_pages(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"manuscript_id", "xml_path"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"Input CSV is missing columns: {', '.join(sorted(missing))}")
    return rows


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def collect_glyphs(
    pages: Iterable[dict[str, Any]], *, alphabet: list[str], merge_final_forms: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    glyph_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    alphabet_set = set(alphabet)
    for page_index, page in enumerate(pages):
        xml_path = str(page["xml_path"])
        status = "ok"
        error = ""
        if not Path(xml_path).is_file():
            strings = []
            status, error = "missing", "XML path does not exist"
        else:
            try:
                strings = parse_alto_strings(xml_path)
            except Exception as exc:  # parser normally logs and returns [], but retain diagnostics
                strings = []
                status, error = "parse_error", f"{type(exc).__name__}: {exc}"

        raw: list[dict[str, Any]] = []
        for word in strings:
            for glyph in word.glyphs:
                char = FINAL_TO_BASE.get(glyph.char, glyph.char) if merge_final_forms else glyph.char
                if char not in alphabet_set:
                    continue
                raw.append(
                    {
                        "letter": char,
                        "confidence": float(glyph.gc) if glyph.gc is not None else math.nan,
                    }
                )

        page_key = f"{page.get('manuscript_id', '')}:{page_index}:{xml_path}"
        for row in raw:
            row.update(
                {
                    "page_key": page_key,
                    "manuscript_id": str(page.get("manuscript_id", "")),
                    "xml_path": xml_path,
                }
            )
            glyph_rows.append(row)
        if status == "ok" and not strings:
            status = "empty_or_unreadable"
        page_rows.append(
            {
                **{key: page.get(key, "") for key in ("manuscript_id", "picture_id", "page_number", "xml_path")},
                "page_key": page_key,
                "status": status,
                "error": error,
                "hebrew_glyph_count": len(raw),
            }
        )
    return pd.DataFrame(glyph_rows), pd.DataFrame(page_rows)


def select_complete_sample(
    page_status: pd.DataFrame,
    glyphs: pd.DataFrame,
    *,
    requested_manuscripts: int,
    pages_per_manuscript: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fill every requested page slot from the randomized manuscript pool.

    Candidate order is already randomized by the database query. Each
    manuscript contributes at most ``pages_per_manuscript`` pages. Missing,
    unreadable, empty, or filter-emptied pages are rejected, and later random
    manuscripts fill their slots until the requested total page count is met.
    """
    status = page_status.copy()
    selected_counts = glyphs.groupby("page_key").size()
    status["selected_glyph_count"] = status["page_key"].map(selected_counts).fillna(0).astype(int)
    status["selection_status"] = "reserve_not_needed"
    status["selection_reason"] = ""
    target_pages = requested_manuscripts * pages_per_manuscript
    accepted_page_keys: list[str] = []

    for manuscript_id in dict.fromkeys(status["manuscript_id"].tolist()):
        mask = status["manuscript_id"] == manuscript_id
        group = status.loc[mask]
        if len(accepted_page_keys) >= target_pages:
            continue
        unusable = (group["status"] != "ok") | (group["selected_glyph_count"] <= 0)
        unusable_indices = group.index[unusable]
        status.loc[unusable_indices, "selection_status"] = "rejected"
        status.loc[unusable_indices, "selection_reason"] = "page had no usable glyphs"
        usable_group = group.loc[~unusable]
        slots_left = target_pages - len(accepted_page_keys)
        accepted_indices = usable_group.index[:slots_left]
        status.loc[accepted_indices, "selection_status"] = "accepted"
        accepted_page_keys.extend(status.loc[accepted_indices, "page_key"].tolist())

    if len(accepted_page_keys) < target_pages:
        raise RuntimeError(
            f"Only {len(accepted_page_keys)} usable pages were found, but {target_pages} "
            "were requested. Increase the candidate pool or reduce "
            "--manuscripts/--pages-per-manuscript."
        )

    accepted_pages = status[status["selection_status"] == "accepted"].copy()
    accepted_keys = set(accepted_page_keys)
    accepted_glyphs = glyphs[glyphs["page_key"].isin(accepted_keys)].copy()
    rejected = status[status["selection_status"] != "accepted"].copy()
    return accepted_pages, accepted_glyphs, rejected


def build_page_features(glyphs: pd.DataFrame, good_pages: pd.DataFrame, alphabet: list[str]) -> pd.DataFrame:
    base = good_pages[["page_key", "manuscript_id", "xml_path"]].drop_duplicates().set_index("page_key")
    features = base.copy()
    if glyphs.empty:
        return features.reset_index()
    counts = pd.crosstab(glyphs["page_key"], glyphs["letter"]).reindex(columns=alphabet, fill_value=0)
    totals = counts.sum(axis=1).replace(0, np.nan)
    mean_gc = glyphs.pivot_table(index="page_key", columns="letter", values="confidence", aggfunc="mean")
    for letter in alphabet:
        features[f"freq_{letter}"] = (counts.get(letter, 0) / totals).reindex(features.index).fillna(0.0)
        features[f"present_{letter}"] = (counts.get(letter, 0) > 0).astype(float).reindex(features.index).fillna(0.0)
        features[f"confidence_{letter}"] = mean_gc.get(letter, pd.Series(dtype=float)).reindex(features.index)
    features["page_mean_confidence"] = glyphs.groupby("page_key")["confidence"].mean().reindex(features.index)
    features["page_glyph_count"] = glyphs.groupby("page_key").size().reindex(features.index).fillna(0)
    return features.reset_index()


def summarize_letters(glyphs: pd.DataFrame, features: pd.DataFrame, alphabet: list[str]) -> pd.DataFrame:
    total_glyphs = len(glyphs)
    total_pages = len(features)
    records: list[dict[str, Any]] = []

    for letter in alphabet:
        subset = glyphs[glyphs["letter"] == letter]
        confidence = subset["confidence"].dropna()
        count = len(subset)
        page_count = int(subset["page_key"].nunique())
        probability_ci = wilson_interval(count, total_glyphs)
        presence_ci = wilson_interval(page_count, total_pages)
        record: dict[str, Any] = {
            "letter": letter,
            "glyph_count": count,
            "page_count": page_count,
            "glyph_probability": count / total_glyphs if total_glyphs else math.nan,
            "glyph_probability_ci_low": probability_ci[0],
            "glyph_probability_ci_high": probability_ci[1],
            "page_presence_probability": page_count / total_pages if total_pages else math.nan,
            "page_presence_ci_low": presence_ci[0],
            "page_presence_ci_high": presence_ci[1],
            "confidence_n": len(confidence),
            "confidence_mean": confidence.mean(),
            "confidence_median": confidence.median(),
            "confidence_std": confidence.std(),
        }
        records.append(record)
    result = pd.DataFrame(records)
    result["frequency_rank"] = result["glyph_probability"].rank(method="min", ascending=False)
    result["confidence_rank"] = result["confidence_mean"].rank(method="min", ascending=False)
    return result


def _plot_outputs(
    glyphs: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    targets: set[str],
    apply_confidence_filter: bool,
    confidence_threshold: float,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )
    colors = ["#c44e52" if letter in targets else "#4c72b0" for letter in summary["letter"]]
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(16, 7))
    counts = summary["glyph_count"].to_numpy(int)
    ax.bar(x, counts, color=colors)
    mean_count = float(np.mean(counts))
    ax.axhline(
        mean_count,
        color="#333333",
        linestyle="--",
        linewidth=1.8,
        label=f"mean across letters = {mean_count:,.1f}",
    )
    count_title = (
        f"Letter counts after GC ≥ {confidence_threshold:.2f} filtering"
        if apply_confidence_filter
        else "Letter counts (no confidence filtering)"
    )
    ax.set(xticks=x, xticklabels=summary["letter"], ylabel="Recognized glyph count", title=count_title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "letter_counts.png", dpi=180)
    plt.close(fig)

    data = [glyphs.loc[glyphs["letter"] == letter, "confidence"].dropna().to_numpy() for letter in summary["letter"]]
    positions_and_data = [(position, values) for position, values in enumerate(data, start=1) if len(values)]
    positions = [position for position, _values in positions_and_data]
    nonempty_data = [values for _position, values in positions_and_data]
    observed_confidence = glyphs["confidence"].dropna()
    confidence_mean = float(observed_confidence.mean())
    confidence_std = float(observed_confidence.std())
    confidence_lower = max(0.0, confidence_mean - 3.0 * confidence_std)
    confidence_upper = float(observed_confidence.max())
    if confidence_lower >= confidence_upper:
        confidence_lower = float(observed_confidence.min())

    # GC is ceiling-heavy. A compact interval plot is more legible than either
    # boxes or KDE violins: it exposes the lower tail without drawing thousands
    # of outliers or producing needle-shaped densities.
    fig, ax = plt.subplots(figsize=(16, 7))
    intervals = np.asarray([np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95]) for values in nonempty_data])
    means = np.asarray([np.mean(values) for values in nonempty_data])
    point_colors = [colors[position - 1] for position in positions]
    ax.vlines(positions, intervals[:, 0], intervals[:, 4], color=point_colors, linewidth=1.4, alpha=0.8)
    ax.vlines(positions, intervals[:, 1], intervals[:, 3], color=point_colors, linewidth=6.0, alpha=0.9)
    ax.scatter(positions, intervals[:, 2], color="white", edgecolor=point_colors, linewidth=1.5, s=34, zorder=3, label="median")
    ax.scatter(positions, means, color=point_colors, marker="D", edgecolor="black", linewidth=0.5, s=22, zorder=4, label="mean")
    ax.set_xticks(np.arange(1, len(summary) + 1), summary["letter"])
    for tick, letter in zip(ax.get_xticklabels(), summary["letter"]):
        if letter in targets:
            tick.set_color("#c44e52")
            tick.set_fontweight("bold")
    ax.set(
        ylabel="ALTO glyph confidence (GC)",
        title="OCR confidence by recognized letter (thin = 5–95%, thick = IQR)",
        ylim=(confidence_lower, confidence_upper),
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(loc="lower right")
    ax.text(
        0.005,
        0.015,
        f"Display range: global mean − 3 SD ({confidence_lower:.3f}) to max ({confidence_upper:.3f})",
        transform=ax.transAxes,
        fontsize=12,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(output / "confidence_by_letter.png", dpi=180)
    plt.close(fig)

    # Retain a conventional box plot as a secondary diagnostic using the same
    # scale, so the two views can be compared directly.
    fig, ax = plt.subplots(figsize=(16, 7))
    boxplot = ax.boxplot(
        data,
        tick_labels=summary["letter"],
        showfliers=False,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 3},
        patch_artist=True,
    )
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for tick, letter in zip(ax.get_xticklabels(), summary["letter"]):
        if letter in targets:
            tick.set_color("#c44e52")
            tick.set_fontweight("bold")
    ax.set(
        ylabel="ALTO glyph confidence (GC)",
        title="OCR confidence by recognized letter (box = IQR, line = median, dot = mean)",
        ylim=(confidence_lower, confidence_upper),
    )
    fig.tight_layout()
    fig.savefig(output / "confidence_boxplot_by_letter.png", dpi=180)
    plt.close(fig)


def write_report(
    *, output: Path, pages: pd.DataFrame, glyphs: pd.DataFrame, summary: pd.DataFrame,
    targets: list[str], merge_final_forms: bool, source: str, seed: int | None,
    apply_confidence_filter: bool, confidence_threshold: float,
) -> None:
    count_column = "selected_glyph_count" if "selected_glyph_count" in pages else "hebrew_glyph_count"
    usable = (pages["status"] == "ok") & (pages[count_column] > 0)
    usable_pages = int(usable.sum())
    mean_glyphs_per_page = len(glyphs) / usable_pages if usable_pages else math.nan
    mean_glyphs_per_letter = len(glyphs) / len(summary) if len(summary) else math.nan
    lines = [
        "# Letter counts and OCR confidence", "",
        f"Source: **{source}**. Random-sampling seed: **{seed if seed is not None else 'input CSV'}**.",
        f"Analyzed **{usable_pages}** pages containing recognized Hebrew glyphs from **{pages.loc[usable, 'manuscript_id'].nunique()}** manuscripts and **{len(glyphs):,}** Hebrew glyphs.",
        f"Mean glyphs per analyzed page: **{mean_glyphs_per_page:,.1f}**. Mean count across letter classes: **{mean_glyphs_per_letter:,.1f}**.",
        (
            f"Applied the glyph-confidence filter **GC >= {confidence_threshold:.2f}**."
            if apply_confidence_filter
            else "All recognized Hebrew glyphs were counted; no glyph-confidence threshold was applied."
        ),
        f"Final forms were {'merged into base letters' if merge_final_forms else 'kept as separate letters'}.", "",
        "## Target letters", "",
        "| letter | count | glyph probability | pages present | mean GC | median GC | frequency rank | GC rank |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    indexed = summary.set_index("letter")
    count_description = "confidence-filtered" if apply_confidence_filter else "unfiltered"

    def rank_text(value: Any) -> str:
        return str(int(value)) if pd.notna(value) else "n/a"

    for letter in targets:
        if letter not in indexed.index:
            continue
        row = indexed.loc[letter]
        lines.append(
            f"| {letter} | {int(row.glyph_count)} | {row.glyph_probability:.3%} | {row.page_presence_probability:.3%} | "
            f"{row.confidence_mean:.3f} | {row.confidence_median:.3f} | "
            f"{rank_text(row.frequency_rank)} | {rank_text(row.confidence_rank)} |"
        )
    lines += [
        "", "## Interpretation notes", "",
        "- Counts describe OCR output, not ground-truth linguistic frequency: OCR substitutions can change both counts and confidence.",
        "- DB sampling caps each manuscript at the requested pages-per-manuscript value and draws replacement manuscripts until every page slot is filled. Glyph-level probability still gives more weight to text-heavy pages; use `page_presence_probability` as its page-level companion.",
        "", "## Files", "",
        "- `letter_summary.csv`: counts, probabilities, and raw-confidence summaries.",
        f"- `letter_counts.png`: {count_description} recognized-letter counts.",
        f"- `confidence_by_letter.png`: confidence quantile intervals for the {count_description} glyphs, displayed from their global mean minus three SD to their maximum.",
        "- `confidence_boxplot_by_letter.png`: secondary box-plot view on the same scale.",
        "- `glyph_measurements.csv`: auditable raw measurements used by the summaries.",
        "- `sampled_pages.csv`: the final complete page sample with capped per-manuscript contribution.",
        "- `rejected_candidates.csv`: replacement-pool manuscripts that were incomplete, unusable, or not needed.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, help="Optional CSV with manuscript_id and xml_path; bypasses DB sampling.")
    parser.add_argument("--db-config", default=str(CLUSTERING_DB_CONFIG_PATH))
    parser.add_argument("--source", choices=("geniza", "pretraining"), default="geniza", help="Database collection to sample (default: geniza).")
    parser.add_argument("--manuscripts", type=int, default=50)
    parser.add_argument("--pages-per-manuscript", type=int, default=3)
    parser.add_argument("--candidate-multiplier", type=int, default=2, help="Random candidate manuscripts fetched to replace incomplete ones (default: 2).")
    parser.add_argument("--seed", type=int, help="Reproduce a random sample; omitted generates a new seed each run.")
    parser.add_argument("--targets", default="אמ", help="Letters highlighted in plots/report (default: aleph and medial mem).")
    parser.add_argument("--merge-final-forms", action="store_true", help="Merge ךםןףץ into כ מנפצ before analysis.")
    parser.add_argument("--apply-confidence-filter", action="store_true", help="Keep only glyphs whose GC meets --confidence-threshold.")
    parser.add_argument("--confidence-threshold", type=float, default=OCR_GLYPH_CONFIDENCE_THRESHOLD, help=f"GC cutoff used with --apply-confidence-filter (default: {OCR_GLYPH_CONFIDENCE_THRESHOLD}).")
    parser.add_argument("--output-dir", type=Path, help="Default: Debugs/LetterAppearance/outputs/<source>_letters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manuscripts <= 0 or args.pages_per_manuscript <= 0 or args.candidate_multiplier <= 0:
        raise ValueError("--manuscripts, --pages-per-manuscript, and --candidate-multiplier must be positive")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")
    alphabet = [letter for letter in HEBREW_ALPHABET if not (args.merge_final_forms and letter in FINAL_TO_BASE)]
    targets = list(dict.fromkeys(FINAL_TO_BASE.get(ch, ch) if args.merge_final_forms else ch for ch in args.targets))
    unknown = set(targets) - set(alphabet)
    if unknown:
        raise ValueError(f"Target letters are not in the analysis alphabet: {sorted(unknown)}")

    sample_seed: int | None = None
    if args.input_csv:
        pages = load_csv_pages(args.input_csv)
        effective_source = "input_csv"
    else:
        sample_seed = args.seed if args.seed is not None else secrets.randbelow(2**63)
        pages = load_db_pages(
            config_path=args.db_config,
            source=args.source,
            manuscripts=args.manuscripts * args.candidate_multiplier,
            pages_per_manuscript=args.pages_per_manuscript,
            seed=sample_seed,
        )
        effective_source = args.source
    if not pages:
        raise RuntimeError("No pages were sampled")
    output_suffix = "_filtered" if args.apply_confidence_filter else ""
    output = (
        args.output_dir
        or (PROJECT_ROOT / f"Debugs/LetterAppearance/outputs/{effective_source}_letters{output_suffix}")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw_glyphs, page_status = collect_glyphs(pages, alphabet=alphabet, merge_final_forms=args.merge_final_forms)
    if raw_glyphs.empty:
        page_status.to_csv(output / "sampled_pages.csv", index=False)
        raise RuntimeError(f"No Hebrew glyphs found; inspect {output / 'sampled_pages.csv'}")
    if args.apply_confidence_filter:
        glyphs = raw_glyphs[
            raw_glyphs["confidence"].notna() & (raw_glyphs["confidence"] >= args.confidence_threshold)
        ].copy()
    else:
        glyphs = raw_glyphs
    if glyphs.empty:
        page_status.to_csv(output / "sampled_pages.csv", index=False)
        raise RuntimeError(f"No glyphs remain after confidence filtering; inspect {output / 'sampled_pages.csv'}")
    rejected_candidates = pd.DataFrame()
    if args.input_csv:
        selected_counts = glyphs.groupby("page_key").size()
        page_status["selected_glyph_count"] = page_status["page_key"].map(selected_counts).fillna(0).astype(int)
        good_pages = page_status[(page_status["status"] == "ok") & (page_status["selected_glyph_count"] > 0)]
    else:
        page_status, glyphs, rejected_candidates = select_complete_sample(
            page_status,
            glyphs,
            requested_manuscripts=args.manuscripts,
            pages_per_manuscript=args.pages_per_manuscript,
        )
        good_pages = page_status
    features = build_page_features(glyphs, good_pages, alphabet)
    summary = summarize_letters(glyphs, features, alphabet)

    page_status.to_csv(output / "sampled_pages.csv", index=False)
    if not rejected_candidates.empty:
        rejected_candidates.to_csv(output / "rejected_candidates.csv", index=False)
    glyphs.to_csv(output / "glyph_measurements.csv", index=False)
    summary.to_csv(output / "letter_summary.csv", index=False)
    (output / "analysis_config.json").write_text(
        json.dumps(
            {
                "source": effective_source,
                "seed": sample_seed,
                "requested_manuscripts": args.manuscripts if not args.input_csv else None,
                "pages_per_manuscript": args.pages_per_manuscript if not args.input_csv else None,
                "candidate_multiplier": args.candidate_multiplier if not args.input_csv else None,
                "merge_final_forms": args.merge_final_forms,
                "targets": targets,
                "confidence_filter_applied": args.apply_confidence_filter,
                "confidence_threshold": args.confidence_threshold if args.apply_confidence_filter else None,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot_outputs(
        glyphs, summary, output, set(targets),
        args.apply_confidence_filter, args.confidence_threshold,
    )
    write_report(
        output=output, pages=page_status, glyphs=glyphs, summary=summary,
        targets=targets, merge_final_forms=args.merge_final_forms, source=effective_source, seed=sample_seed,
        apply_confidence_filter=args.apply_confidence_filter, confidence_threshold=args.confidence_threshold,
    )
    skipped = len(page_status) - len(good_pages)
    rejected_page_count = (
        int((rejected_candidates["selection_status"] == "rejected").sum())
        if not rejected_candidates.empty
        else 0
    )
    print(f"Source: {effective_source}; random seed: {sample_seed if sample_seed is not None else 'input CSV'}")
    print(f"Analyzed {len(good_pages)} pages and {len(glyphs):,} Hebrew glyphs; skipped {skipped} pages.")
    if not args.input_csv:
        print(
            f"Used {page_status['manuscript_id'].nunique()} random manuscripts to fill "
            f"{args.manuscripts * args.pages_per_manuscript} page slots; replaced "
            f"{rejected_page_count} unusable candidate pages."
        )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
