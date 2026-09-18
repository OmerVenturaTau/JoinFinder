import os
import logging
import argparse
import sys
import math
import random
import numpy as np
from collections import defaultdict

# Set PyTorch CUDA allocator to use expandable segments to reduce memory fragmentation
# This helps prevent OOM errors when total memory is available but fragmented
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# DataLoader workers fork after the parent may have used HF tokenizers; without this,
# tokenizers prints "forked after parallelism" warnings and can deadlock in some setups.
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parse command-line arguments BEFORE importing system.py
# This allows us to modify system configuration based on mode
parser = argparse.ArgumentParser(description='Train manuscript classification model')
parser.add_argument('--init', choices=('scratch', 'checkpoint'), default=None,
                   help='Model initialization: scratch starts from random weights; checkpoint loads --checkpoint.')
parser.add_argument('--checkpoint', '--pretrain_checkpoint', dest='checkpoint', type=str, default=None,
                   help='Checkpoint path when --init checkpoint is used.')
parser.add_argument('--stage', choices=('stage1', 'stage2', 'demo'), default=None,
                   help='Classification dataset/table preset. stage1 uses STAGE1_TABLE_NAME, stage2 uses STAGE2_TABLE_NAME plus stage2 sampling.')
parser.add_argument('--geniza', choices=('none', 'classification', 'contrastive'), default='none',
                   help='Optional Geniza training: omit (or use none) to exclude Geniza; classification merges Geniza manuscripts into the classification train/val splits; contrastive enables the separate Geniza memory-bank objective.')
legacy_group = parser.add_mutually_exclusive_group()
legacy_group.add_argument('--pretrain', action='store_true',
                       help='Deprecated alias for --init scratch --stage stage1.')
legacy_group.add_argument('--finetune', action='store_true',
                       help='Deprecated alias for --init checkpoint --stage stage2.')
legacy_group.add_argument('--demo', action='store_true',
                       help='Deprecated alias for --init scratch --stage demo.')
parser.add_argument('--freeze_backbone_from', type=int, default=None,
                   help='First epoch (inclusive) to freeze TILE branch. With --freeze_backbone_until defines the freeze window. Defaults from system.py.')
parser.add_argument('--freeze_backbone_until', type=int, default=None,
                   help='Last epoch (inclusive) to freeze TILE branch. Must be set together with --freeze_backbone_from. Defaults from system.py.')
parser.add_argument('--num_epochs', type=int, default=None,
                   help='Number of training epochs. Defaults to system.NUM_EPOCHS.')
parser.add_argument('--table', type=str, default=None,
                   help='Classification DB table override. If not set, uses the selected stage table.')
parser.add_argument('--xml_base_path', type=str, default=None,
                   help='Override default XML base path for find_xml_path_pretrain() fallback')
parser.add_argument('--aggregation_gpu', type=int, default=0,
                   help='GPU index for aggregation/main process (default: 0)')
parser.add_argument('--disable_geniza_contrastive', action='store_true',
                   help='Deprecated alias for --geniza none.')
args = parser.parse_args()

# Validate arguments
legacy_flags = [args.pretrain, args.finetune, args.demo]
if sum(bool(x) for x in legacy_flags) > 0:
    legacy_init = "checkpoint" if args.finetune else "scratch"
    legacy_stage = "stage2" if args.finetune else ("demo" if args.demo else "stage1")
    if args.init is not None and args.init != legacy_init:
        parser.error(f'legacy mode conflicts with --init {args.init!r}; use --init {legacy_init}')
    if args.stage is not None and args.stage != legacy_stage:
        parser.error(f'legacy mode conflicts with --stage {args.stage!r}; use --stage {legacy_stage}')
    args.init = legacy_init
    args.stage = legacy_stage

if args.init is None:
    args.init = "checkpoint" if args.checkpoint else "scratch"
if args.stage is None:
    args.stage = "stage1"
if args.init == "checkpoint" and not args.checkpoint:
    parser.error('--init checkpoint requires --checkpoint')
if args.init == "scratch" and args.checkpoint:
    parser.error('--checkpoint was provided with --init scratch; use --init checkpoint or remove --checkpoint')
if (args.freeze_backbone_from is not None) != (args.freeze_backbone_until is not None):
    parser.error('--freeze_backbone_from and --freeze_backbone_until must be set together')
if args.disable_geniza_contrastive and args.geniza != "none":
    parser.error('--disable_geniza_contrastive conflicts with --geniza; use --geniza none or remove --disable_geniza_contrastive')

# Configure system based on mode BEFORE importing from system
import system

# Initialization and dataset stage are deliberately separate.
system.TRAINING_MODE = args.stage
system.GENIZA_CONTRASTIVE_ENABLED = args.geniza == "contrastive"
if args.stage == "stage1":
    default_table = system.STAGE1_TABLE_NAME
elif args.stage == "stage2":
    default_table = system.STAGE2_TABLE_NAME
else:
    default_table = system.STAGE1_TABLE_NAME
system.recompute_run_name()
system.recompute_best_model_path()

table_name = args.table if args.table is not None else default_table
if args.xml_base_path is not None:
    system.FALLBACK_XML_BASE_PATH = args.xml_base_path

# Resolve per-branch freeze ranges from system.py; CLI can override tile range (any mode).
args.tile_branch_freeze_from_epoch = system.TILE_BRANCH_FREEZE_FROM_EPOCH
args.tile_branch_freeze_until_epoch = system.TILE_BRANCH_FREEZE_UNTIL_EPOCH
args.glyph_branch_freeze_from_epoch = system.GLYPH_BRANCH_FREEZE_FROM_EPOCH
args.glyph_branch_freeze_until_epoch = system.GLYPH_BRANCH_FREEZE_UNTIL_EPOCH
args.word_branch_freeze_from_epoch = system.WORD_BRANCH_FREEZE_FROM_EPOCH
args.word_branch_freeze_until_epoch = system.WORD_BRANCH_FREEZE_UNTIL_EPOCH
if args.freeze_backbone_from is not None and args.freeze_backbone_until is not None:
    args.tile_branch_freeze_from_epoch = args.freeze_backbone_from
    args.tile_branch_freeze_until_epoch = args.freeze_backbone_until

print(f"\n{'='*80}")
print(f"INIT: {args.init.upper()}")
if args.init == "checkpoint":
    print(f"  Checkpoint: {args.checkpoint}")
print(f"STAGE: {args.stage.upper()}")
print(f"GENIZA MODE: {args.geniza.upper()}")
if args.stage == "demo":
    print("  Demo subset: small 50/50 oriental vs non-oriental pipeline sanity check")
print(f"  Tile branch freeze: epochs {args.tile_branch_freeze_from_epoch}-{args.tile_branch_freeze_until_epoch} (inclusive)")
print(f"  Glyph branch freeze: epochs {args.glyph_branch_freeze_from_epoch}-{args.glyph_branch_freeze_until_epoch} (inclusive)")
print(f"  Word branch freeze: epochs {args.word_branch_freeze_from_epoch}-{args.word_branch_freeze_until_epoch} (inclusive)")
if args.table:
    print(f"  TABLE OVERRIDE: {args.table}")
if args.xml_base_path:
    print(f"  XML fallback base: {args.xml_base_path}")
print(f"{'='*80}")
print(f"  Classification table: {table_name}")
print("  Classification split policy: DB dataset_split column when present; fallback is page-level split within manuscript")
if bool(system.GENIZA_CONTRASTIVE_ENABLED):
    print(f"  Geniza contrastive table: {system.GENIZA_CONTRASTIVE_TABLE} (independent of --stage)")
elif args.geniza == "classification":
    print(f"  Geniza classification table: {system.GENIZA_CONTRASTIVE_TABLE} (merged into classification train/val splits)")
else:
    print("  Geniza training: disabled")
print(f"  XML paths: from DB (xml_path column)" + (f", fallback base: {system.FALLBACK_XML_BASE_PATH}" if system.FALLBACK_XML_BASE_PATH else "None"))
print("  Augmentation config:")
print(
    "    PIL apply gate:"
    f" p={float(system.AUGMENT_APPLY_PROB):.3f}"
    " (candidate weights normalized inside gate)"
)
print(
    "    raw background weights:"
    f" random_library={float(system.AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB):.3f}"
    f" white={float(system.AUGMENT_WHITE_BACKGROUND_PROB):.3f}"
    f" alpha={float(system.AUGMENT_BACKGROUND_SAMPLING_ALPHA):.3f}"
    f" floor={float(system.AUGMENT_BACKGROUND_SAMPLING_FLOOR):.1f}"
)
print(
    "    local_texture/tone_contrast/tilt/zoom/border_crop:"
    f" {float(system.AUGMENT_LOCAL_TEXTURE_PROB):.3f}/"
    f"{float(system.AUGMENT_TONE_CONTRAST_PROB):.3f}/"
    f"{float(system.AUGMENT_TILT_PROB):.3f}/"
    f"{float(system.AUGMENT_ZOOM_PROB):.3f}/"
    f"{float(system.AUGMENT_BORDER_CROP_PROB):.3f}"
)
print(
    "    token_subsample prob/range:"
    f" {float(system.TOKEN_SUBSAMPLE_PROB):.3f}/{tuple(system.TOKEN_SUBSAMPLE_FRAC_RANGE)}"
)
print(f"{'='*80}\n")

from train.split_data import (
    _prepare_sorted_df,
    build_page_level_classification_splits,
    build_splits,
    merge_classification_splits,
    paths_labels_from_dataset_split_column,
)
from train.db_loader import get_db_connection, get_table_as_df
from train.dataset import ManuscriptDataset, tile_collate_with_padding
from train.metric_learning import LabeledFeatureQueue, MemoryBankSupConLoss
from train.pk_sampler import PKBatchSampler
from train.optimizer_groups import (
    build_optimizer_param_groups,
    build_ratio_preserving_cosine_scheduler,
)
from models import MultiModal
from losses.combined_loss import CombinedLoss
from train.trainer import train
from utilities.checkpoint_utils import without_classification_head
from utilities.augmentations.background_overlays import RandomLibraryBackgroundOverlay
from utilities.augmentations.manuscript_augmentations import (
    RandomResolutionJitter,
    RandomZoomJitter,
    RandomTiltJitter,
    RandomLocalTexturePerturbation,
    RandomToneAndContrastJitter,
    RandomBorderMaskAndEdgeCrop,
    RandomWhiteBackground,
    WeightedOneOf,
)
from torchvision import transforms
import torch
from torch import optim
from torch.utils.data import DataLoader
import wandb
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('joinsfinder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
from system import (
    BASE_DIR,
    MAX_TILES_TRAIN,
    MAX_TILES_EVAL,
    MAX_SELECTED_MANUSCRIPTS,
    PER_GPU_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    LEARNING_RATE_STAGE1,
    LEARNING_RATE_STAGE2,
    LEARNING_RATE_ALEPHBERT,
    LEARNING_RATE_NEW_WORD,
    TRAINING_SEED,
    WEIGHT_DECAY,
    BETAS,
    # CE_WEIGHT,
    LATENT_SPARSITY_WEIGHT,
    ARCFACE_WEIGHT,
    ARCFACE_MARGIN,
    ARCFACE_SCALE,
    CE_WEIGHT,
    LATENT_DIM,
    FUSION_METHOD,
    SHARED_BRANCH_ARCFACE,
    USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
    SYMMETRIC_BRANCH_DIM,
    GATE_ENTROPY_WEIGHT,
    GATE_ENTROPY_DECAY_EPOCHS,
    WORD_AUX_FULL_WEIGHT_EPOCHS,
    WORD_AUX_DECAY_END_EPOCH,
    NUM_EPOCHS,
    PROJECT_NAME,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    SCHEDULER_ETA_MIN,
    TILE_SIZE,
    TRAINING_MODE,
    LABEL_HEAD,
    MAIN_USE_DATAPARALLEL,
    MAIN_DATAPARALLEL_NUM_WORKERS,
    USE_VISUAL_MOD,
    USE_CHAR_MOD,
    USE_WORD_MOD,
    WORD_BRANCH_USE_TFIDF_GATING,
    TILE_NUM_SUMMARY_TOKENS,
    GLYPH_ENCODER_TYPE,
    GLYPH_NUM_SUMMARY_TOKENS,
    WORD_NUM_SUMMARY_TOKENS,
    OCR_GLYPH_CONFIDENCE_THRESHOLD,
    GLYPH_QUALITY_MIN_AREA,
    GLYPH_QUALITY_MIN_WIDTH,
    GLYPH_QUALITY_MIN_HEIGHT,
    GLYPH_QUALITY_MAX_ASPECT_RATIO,
    ENCODER_MODEL_NAME,
    AUGMENT_COLOR_JITTER,
    AUGMENT_APPLY_PROB,
    AUGMENT_COLOR_JITTER_PROB,
    AUGMENT_COLOR_JITTER_BRIGHTNESS,
    AUGMENT_COLOR_JITTER_CONTRAST,
    AUGMENT_COLOR_JITTER_SATURATION,
    AUGMENT_COLOR_JITTER_HUE,
    AUGMENT_RANDOM_GRAYSCALE_PROB,
    AUGMENT_GAUSSIAN_BLUR_PROB,
    AUGMENT_GAUSSIAN_BLUR_KERNEL,
    AUGMENT_BACKGROUND_PATTERN_PROB,
    AUGMENT_BACKGROUND_SURFACE_PATTERN_PROB,
    AUGMENT_BACKGROUND_LIBRARY_PATTERN_PROB,
    AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB,
    AUGMENT_PARCHMENT_STAINS_PROB,
    AUGMENT_WHITE_BACKGROUND_PROB,
    AUGMENT_BACKGROUND_SAMPLING_ALPHA,
    AUGMENT_BACKGROUND_SAMPLING_FLOOR,
    AUGMENT_BACKGROUND_SURFACE_PATTERN_TYPES,
    AUGMENT_BACKGROUND_LIBRARY_PATTERN_TYPES,
    AUGMENT_BACKGROUND_PATTERN_TYPES,
    AUGMENT_BACKGROUND_PATTERN_ALPHA_RANGE,
    AUGMENT_BACKGROUND_PATTERN_GRID_SPACING_RANGE,
    AUGMENT_BACKGROUND_PATTERN_GRID_LINE_WIDTH,
    AUGMENT_RANDOM_ERASING_PROB,
    AUGMENT_RANDOM_ERASING_SCALE,
    AUGMENT_RANDOM_ERASING_RATIO,
    AUGMENT_LOCAL_TEXTURE_PROB,
    AUGMENT_LOCAL_TEXTURE_ALPHA_RANGE,
    AUGMENT_TONE_CONTRAST_PROB,
    AUGMENT_BORDER_CROP_PROB,
    AUGMENT_BORDER_MAX_FRAC,
    AUGMENT_EDGE_CROP_MAX_FRAC,
    AUGMENT_ZOOM_PROB,
    AUGMENT_ZOOM_SCALE_RANGE,
    AUGMENT_TILT_PROB,
    AUGMENT_TILT_DEGREES,
    AUGMENT_RESOLUTION_JITTER_PROB,
    AUGMENT_RANDAUGMENT_PROB,
    TILE_AUX_LOSS_WEIGHT,
    GLYPH_AUX_LOSS_WEIGHT,
    FUSION_AUX_LOSS_WEIGHT,
    WORD_AUX_LOSS_WEIGHT,
    D_MODEL,
    MODALITY_DROPOUT_ENABLED,
    MODALITY_DROPOUT_PROB_VISUAL,
    MODALITY_DROPOUT_PROB_CHAR,
    MODALITY_DROPOUT_PROB_WORD,
    GENIZA_CONTRASTIVE_ENABLED,
    GENIZA_CONTRASTIVE_TABLE,
    GENIZA_CONTRASTIVE_WEIGHT,
    GENIZA_CONTRASTIVE_TEMPERATURE,
    GENIZA_QUEUE_SIZE,
    GENIZA_PK_K,
    GENIZA_DEMO_FRACTION,
    GENIZA_DEMO_MIN_TRAIN_MANUSCRIPTS,
    GENIZA_DEMO_MIN_VAL_MANUSCRIPTS,
    GENIZA_DEMO_MAX_IMAGES_PER_MANUSCRIPT,
)

from tasks.label_heads.factory import create_label_head
from tasks.label_heads.base import build_label_maps


def _load_geniza_training_df(table_name: str):
    """Load and validate rows shared by the two Geniza training modes."""
    conn = get_db_connection()
    try:
        df = get_table_as_df(conn, table_name)
    finally:
        conn.close()

    required_columns = {"manuscript_id", "picture_id", "parent_directory", "dataset_split"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise RuntimeError(
            f"Geniza training table {table_name!r} is missing required columns: {missing_columns}"
        )

    normalized_split = df["dataset_split"].astype(str).str.strip().str.lower()
    df = df[normalized_split.isin(["train", "val"])].copy()
    if df.empty:
        raise RuntimeError(
            f"Geniza training table {table_name!r} has no rows with dataset_split in ('train', 'val')."
        )

    return _prepare_sorted_df(df)


def _load_geniza_contrastive_split_lists(base_dir: str, table_name: str):
    """
    Load Geniza train/val lists directly from dataset_split.

    This deliberately bypasses build_splits() because --demo mode changes the
    generic table splitting behavior; Geniza contrastive should always respect
    the real manuscript-disjoint split created in geniza_train_set.
    """
    df = _load_geniza_training_df(table_name)

    (
        train_paths,
        train_labels,
        train_xmls,
        val_paths,
        val_labels,
        val_xmls,
        _test_paths,
        _test_labels,
        _test_xmls,
    ) = paths_labels_from_dataset_split_column(df, base_dir)
    return (
        train_paths,
        [str(x) for x in train_labels],
        train_xmls,
        val_paths,
        [str(x) for x in val_labels],
        val_xmls,
    )


def _load_geniza_classification_splits(base_dir: str, table_name: str):
    """Load Geniza rows and page-split every manuscript for classification."""
    df = _load_geniza_training_df(table_name)
    train_fraction = float(system.TRAIN_RATIO) / float(system.TRAIN_RATIO + system.VAL_RATIO)
    return build_page_level_classification_splits(
        df,
        base_dir,
        train_fraction=train_fraction,
    )


def _sample_geniza_demo_split(
    paths,
    labels,
    xmls,
    *,
    fraction: float,
    min_manuscripts: int,
    max_images_per_manuscript: int,
    seed: int,
):
    groups = defaultdict(list)
    for path, label, xml in zip(paths, labels, xmls):
        groups[str(label)].append((path, str(label), xml))

    manuscript_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(manuscript_ids)

    target_manuscripts = max(int(min_manuscripts), int(math.ceil(len(manuscript_ids) * float(fraction))))
    target_manuscripts = min(target_manuscripts, len(manuscript_ids))
    selected_manuscripts = manuscript_ids[:target_manuscripts]

    sampled = []
    for manuscript_id in selected_manuscripts:
        # Stable page/path ordering keeps demo deterministic inside each selected manuscript.
        items = sorted(groups[manuscript_id], key=lambda item: item[0])
        sampled.extend(items[: int(max_images_per_manuscript)])

    sampled_paths = [item[0] for item in sampled]
    sampled_labels = [item[1] for item in sampled]
    sampled_xmls = [item[2] for item in sampled]
    stats = {
        "source_manuscripts": len(manuscript_ids),
        "source_images": len(paths),
        "selected_manuscripts": len(set(sampled_labels)),
        "selected_images": len(sampled_paths),
    }
    return sampled_paths, sampled_labels, sampled_xmls, stats


if __name__ == '__main__':
    # --- Optional DDP (torchrun) setup ---
    # Run with: torchrun --nproc_per_node=NUM_GPUS main.py
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # DDP collectives (including barriers) share this timeout. Keep it high enough to tolerate
        # epoch-end analysis/visualization on rank0 without triggering NCCL watchdog aborts.
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(hours=2),
        )
        device = torch.device("cuda", local_rank)
        # Disable wandb on non-zero ranks to avoid duplicate runs/logging.
        if rank != 0:
            os.environ.setdefault("WANDB_MODE", "disabled")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(args.aggregation_gpu)
            device = torch.device(f"cuda:{args.aggregation_gpu}")
        else:
            device = torch.device("cpu")

    process_seed = int(TRAINING_SEED) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info(
        "Deterministic training seed: base=%d rank=%d effective=%d",
        TRAINING_SEED,
        rank,
        process_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Full training requires CUDA, but no CUDA device is visible. "
            "Aborting before dataset/model construction instead of falling back to CPU."
        )

    # Optional single-process multi-GPU (DataParallel). This is mutually exclusive with DDP.
    use_dataparallel = (
        (not is_distributed)
        and bool(MAIN_USE_DATAPARALLEL)
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    )
    dp_num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if use_dataparallel:
        logger.warning(
            f"DataParallel enabled: num_gpus={dp_num_gpus}. "
            f"Batch will be scaled from PER_GPU_BATCH_SIZE={PER_GPU_BATCH_SIZE} to {PER_GPU_BATCH_SIZE * dp_num_gpus}."
        )
        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            names = [torch.cuda.get_device_name(i) for i in range(dp_num_gpus)]
            logger.warning(f"DataParallel CUDA_VISIBLE_DEVICES={visible!r} device_names={names}")
        except Exception as e:
            logger.debug(f"Could not get CUDA device names: {type(e).__name__}: {e}")

    # 1. Data split
    splits, split_stats = build_splits(BASE_DIR, table_name=table_name, dataset_stage=args.stage)
    if args.geniza == "classification":
        geniza_classification_splits, geniza_classification_stats = _load_geniza_classification_splits(
            BASE_DIR,
            GENIZA_CONTRASTIVE_TABLE,
        )
        splits = merge_classification_splits(splits, geniza_classification_splits)
        split_stats["geniza_classification"] = geniza_classification_stats
        logger.info(
            "[GENIZA CLASSIFICATION] merged table=%s: "
            "train_images=%d, val_images=%d, manuscripts=%d; "
            "each multi-page manuscript appears in both classification splits",
            GENIZA_CONTRASTIVE_TABLE,
            geniza_classification_stats["images_train"],
            geniza_classification_stats["images_val"],
            geniza_classification_stats["manuscripts_total"],
        )

    # 2. Prepare label encoding (task / label head)
    label_head = create_label_head(name=LABEL_HEAD)
    logger.info(f"[DEBUG] Label head: {label_head.name}")
    
    flat = label_head.flatten_splits(splits)
    all_train_paths, all_train_labels, all_train_xmls = flat.train_paths, flat.train_labels, flat.train_xmls
    all_val_paths, all_val_labels, all_val_xmls = flat.val_paths, flat.val_labels, flat.val_xmls
    all_test_paths, all_test_labels, all_test_xmls = flat.test_paths, flat.test_labels, flat.test_xmls

    # ---------------------------------------------------------------------
    # Leak-proofing: ensure there is ZERO image overlap between splits.
    # Manuscript-ID classification intentionally uses page-level splits, so the
    # same manuscript/class may appear in train, val and test.
    # ---------------------------------------------------------------------
    def _norm_path(p: str) -> str:
        try:
            # use abspath instead of realpath to avoid slow stats on NAS
            return os.path.abspath(str(p))
        except Exception:
            return str(p)

    train_set = {_norm_path(p) for p in all_train_paths}
    val_set = {_norm_path(p) for p in all_val_paths}
    test_set = {_norm_path(p) for p in all_test_paths}

    # Duplicates *within* a split
    dup_train = len(all_train_paths) - len(train_set)
    dup_val = len(all_val_paths) - len(val_set)
    dup_test = len(all_test_paths) - len(test_set)

    # Overlap *between* splits
    tv_overlap = sorted(train_set & val_set)
    tt_overlap = sorted(train_set & test_set)
    vt_overlap = sorted(val_set & test_set)

    logger.info(
        "[SPLIT CHECK] images: "
        f"train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"dups(train/val/test)={dup_train}/{dup_val}/{dup_test} "
        f"overlap(tv/tt/vt)={len(tv_overlap)}/{len(tt_overlap)}/{len(vt_overlap)}"
    )
    if dup_train or dup_val or dup_test or tv_overlap or tt_overlap or vt_overlap:
        # Print a small sample to make debugging actionable.
        if tv_overlap:
            logger.error(f"[SPLIT CHECK] train∩val sample: {tv_overlap[:5]}")
        if tt_overlap:
            logger.error(f"[SPLIT CHECK] train∩test sample: {tt_overlap[:5]}")
        if vt_overlap:
            logger.error(f"[SPLIT CHECK] val∩test sample: {vt_overlap[:5]}")
        raise RuntimeError(
            "Data leakage detected: duplicate/overlapping image paths across splits. "
            "See [SPLIT CHECK] logs above."
        )

    # Manuscript overlap is expected for page-level manuscript-ID classification,
    # but still useful to log because it confirms the intended split shape.
    def _manuscript_id_from_path(p: str) -> str:
        parts = os.path.normpath(str(p)).split(os.sep)
        return parts[-3] if len(parts) >= 3 else "unknown"

    train_manuscript_ids = {_manuscript_id_from_path(p) for p in all_train_paths}
    val_manuscript_ids = {_manuscript_id_from_path(p) for p in all_val_paths}
    test_manuscript_ids = {_manuscript_id_from_path(p) for p in all_test_paths}
    tv_ms = train_manuscript_ids & val_manuscript_ids
    tt_ms = train_manuscript_ids & test_manuscript_ids
    vt_ms = val_manuscript_ids & test_manuscript_ids
    logger.info(
        "[SPLIT CHECK] manuscripts: "
        f"train={len(train_manuscript_ids)} val={len(val_manuscript_ids)} test={len(test_manuscript_ids)} "
        f"train∩val={len(tv_ms)} train∩test={len(tt_ms)} val∩test={len(vt_ms)}"
    )
    if label_head.name != "manuscript_id" and (tv_ms or tt_ms or vt_ms):
        logger.error(
            "[SPLIT CHECK] Expected manuscript-disjoint split for this label head but found overlap: "
            f"train∩val={len(tv_ms)}, train∩test={len(tt_ms)}, val∩test={len(vt_ms)}"
        )
        raise RuntimeError(
            "Split error: some manuscripts appear in more than one split. "
            f"LABEL_HEAD={label_head.name!r} requires manuscript-level train/val/test separation."
        )
    if label_head.name == "manuscript_id":
        logger.info("[SPLIT CHECK] Manuscript overlap across splits is allowed for page-level manuscript-ID classification.")
        print(
            "[SPLIT CHECK] Manuscripts may overlap across train/val/test "
            f"(train∩val={len(tv_ms)}, train∩test={len(tt_ms)}, val∩test={len(vt_ms)})."
        )
    else:
        logger.info("[SPLIT CHECK] ✓ No manuscript appears in more than one split (train/val/test disjoint).")
        print("[SPLIT CHECK] Manuscripts: train/val/test are disjoint (no manuscript in more than one split).")

    # DEBUG: Check label consistency
    logger.info(f"[DEBUG] Label statistics:")
    logger.info(f"  Train labels: {len(all_train_labels)} samples, {len(set(all_train_labels))} unique labels")
    logger.info(f"  Val labels: {len(all_val_labels)} samples, {len(set(all_val_labels))} unique labels")
    logger.info(f"  Test labels: {len(all_test_labels)} samples, {len(set(all_test_labels))} unique labels")
    
    all_labels_combined = all_train_labels + all_val_labels + all_test_labels
    unique_labels = sorted(set(all_labels_combined))
    logger.info(f"  Total unique labels across all splits: {len(unique_labels)}")
    logger.info(f"  Sample labels (first 10): {unique_labels[:10]}")
    
    label2idx, idx2label = build_label_maps(all_labels=all_labels_combined)
    
    # DEBUG: Verify label2idx mapping
    logger.info(f"[DEBUG] Label2idx mapping created:")
    logger.info(f"  Total classes: {len(label2idx)}")
    logger.info(f"  Label range: [{min(label2idx.values())}, {max(label2idx.values())}]")
    logger.info(f"  Sample mappings (first 5): {dict(list(label2idx.items())[:5])}")
    
    # Verify all labels in splits are in label2idx
    train_labels_set = set(all_train_labels)
    val_labels_set = set(all_val_labels)
    test_labels_set = set(all_test_labels)
    missing_train = train_labels_set - set(label2idx.keys())
    missing_val = val_labels_set - set(label2idx.keys())
    missing_test = test_labels_set - set(label2idx.keys())
    
    if missing_train or missing_val or missing_test:
        logger.error(f"[DEBUG] CRITICAL: Labels missing from label2idx!")
        if missing_train:
            logger.error(f"  Missing in train: {list(missing_train)[:10]}")
        if missing_val:
            logger.error(f"  Missing in val: {list(missing_val)[:10]}")
        if missing_test:
            logger.error(f"  Missing in test: {list(missing_test)[:10]}")
        raise ValueError("Label mismatch detected! Some labels in splits are not in label2idx.")
    else:
        logger.info(f"[DEBUG] ✓ All labels in splits are present in label2idx mapping")

    # 3. Transforms
    # Note: These transforms are applied to individual TILES (560x560), not the full image
    # The high-resolution image is kept at full resolution and tiled in ManuscriptDataset
    tile_oneof_candidates = [
        (
            "jitter",
            transforms.ColorJitter(
                brightness=AUGMENT_COLOR_JITTER_BRIGHTNESS,
                contrast=AUGMENT_COLOR_JITTER_CONTRAST,
                saturation=AUGMENT_COLOR_JITTER_SATURATION,
                hue=AUGMENT_COLOR_JITTER_HUE,
            ) if AUGMENT_COLOR_JITTER else None,
            AUGMENT_COLOR_JITTER_PROB,
        ),
        ("gray", transforms.RandomGrayscale(p=1.0), AUGMENT_RANDOM_GRAYSCALE_PROB),
        ("blur", transforms.GaussianBlur(kernel_size=AUGMENT_GAUSSIAN_BLUR_KERNEL), AUGMENT_GAUSSIAN_BLUR_PROB),
        (
            "random_library_background",
            RandomLibraryBackgroundOverlay(
                p=1.0,
                pattern_types=("random_library_background",),
                alpha_range=AUGMENT_BACKGROUND_PATTERN_ALPHA_RANGE,
                grid_spacing_range=AUGMENT_BACKGROUND_PATTERN_GRID_SPACING_RANGE,
                grid_line_width=AUGMENT_BACKGROUND_PATTERN_GRID_LINE_WIDTH,
                background_sampling_alpha=AUGMENT_BACKGROUND_SAMPLING_ALPHA,
                background_sampling_floor=AUGMENT_BACKGROUND_SAMPLING_FLOOR,
            ),
            AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB,
        ),
        (
            "parchment_stains",
            RandomLibraryBackgroundOverlay(
                p=1.0,
                pattern_types=("parchment_stains",),
                alpha_range=AUGMENT_BACKGROUND_PATTERN_ALPHA_RANGE,
                grid_spacing_range=AUGMENT_BACKGROUND_PATTERN_GRID_SPACING_RANGE,
                grid_line_width=AUGMENT_BACKGROUND_PATTERN_GRID_LINE_WIDTH,
                background_sampling_alpha=AUGMENT_BACKGROUND_SAMPLING_ALPHA,
                background_sampling_floor=AUGMENT_BACKGROUND_SAMPLING_FLOOR,
            ),
            AUGMENT_PARCHMENT_STAINS_PROB,
        ),
        ("white_background", RandomWhiteBackground(p=1.0), AUGMENT_WHITE_BACKGROUND_PROB),
        ("local_texture", RandomLocalTexturePerturbation(p=1.0, alpha_range=AUGMENT_LOCAL_TEXTURE_ALPHA_RANGE), AUGMENT_LOCAL_TEXTURE_PROB),
        ("tone_contrast", RandomToneAndContrastJitter(p=1.0), AUGMENT_TONE_CONTRAST_PROB),
        ("tilt", RandomTiltJitter(degrees=AUGMENT_TILT_DEGREES, p=1.0), AUGMENT_TILT_PROB),
        ("border_crop", RandomBorderMaskAndEdgeCrop(p=1.0, max_border_frac=AUGMENT_BORDER_MAX_FRAC, max_crop_frac=AUGMENT_EDGE_CROP_MAX_FRAC), AUGMENT_BORDER_CROP_PROB),
        ("zoom", RandomZoomJitter(scale_range=AUGMENT_ZOOM_SCALE_RANGE, p=1.0), AUGMENT_ZOOM_PROB),
        ("resolution_jitter", RandomResolutionJitter(scale_range=(0.5, 0.8), p=1.0), AUGMENT_RESOLUTION_JITTER_PROB),
        ("randaugment", transforms.RandAugment(num_ops=2, magnitude=6), AUGMENT_RANDAUGMENT_PROB),
    ]
    tile_augment_ops = [WeightedOneOf(tile_oneof_candidates, p=AUGMENT_APPLY_PROB)]
    train_transform = transforms.Compose([
        *tile_augment_ops,
        # Candidate weights are normalized inside WeightedOneOf. With p=0.8,
        # 20% of samples skip PIL augmentation entirely.
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=AUGMENT_RANDOM_ERASING_PROB,
            scale=AUGMENT_RANDOM_ERASING_SCALE,
            ratio=AUGMENT_RANDOM_ERASING_RATIO,
            value="random",
        ) if AUGMENT_RANDOM_ERASING_PROB > 0 else transforms.Lambda(lambda x: x),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),  # No resizing - tiles are already extracted at correct size
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

    # 4. Datasets and loaders
    # Note: Each dataset only receives paths from its split, ensuring no data leakage.
    # When loading patches from DB, we query by image_path, so we can only get patches
    # for images that are in the current dataset's path list.
    train_dataset = ManuscriptDataset(all_train_paths, all_train_labels, train_transform, label2idx, xml_paths=all_train_xmls, max_tiles_per_image=MAX_TILES_TRAIN, split='train')
    val_dataset = ManuscriptDataset(all_val_paths, all_val_labels, eval_transform, label2idx, xml_paths=all_val_xmls, max_tiles_per_image=MAX_TILES_EVAL, split='val')
    test_dataset = ManuscriptDataset(all_test_paths, all_test_labels, eval_transform, label2idx, xml_paths=all_test_xmls, max_tiles_per_image=MAX_TILES_EVAL, split='test')

    # Batch size:
    # - DDP: PER PROCESS (per GPU). Global effective batch becomes:
    #   PER_GPU_BATCH_SIZE * world_size * GRADIENT_ACCUMULATION_STEPS
    # - DataParallel: single process, we need to scale the batch so each GPU gets ~PER_GPU_BATCH_SIZE.
    if use_dataparallel:
        effective_batch = PER_GPU_BATCH_SIZE * max(1, dp_num_gpus)
    else:
        effective_batch = PER_GPU_BATCH_SIZE

    train_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(TRAINING_SEED),
            drop_last=False,
        )

    def make_loader_generator(offset: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(process_seed + int(offset))
        return generator

    train_loader_generator = make_loader_generator(0)

    # In DataParallel mode, keep DataLoader multiprocessing conservative to reduce hang risk.
    loader_num_workers = MAIN_DATAPARALLEL_NUM_WORKERS if use_dataparallel else NUM_WORKERS
    loader_pin = PIN_MEMORY and torch.cuda.is_available()
    # Pinning can crash (or severely thrash) when the batch tensors are extremely large.
    # With TILE_SIZE=560, MAX_TILES_TRAIN=32, float32 tiles can approach ~1GB per batch (before prefetching).
    # If we estimate the pinned-host allocation is too large, disable pinning proactively.
    try:
        bytes_per_float = 4  # transforms output float32 tensors
        est_train_tile_bytes = int(effective_batch) * int(MAX_TILES_TRAIN) * 3 * int(TILE_SIZE) * int(TILE_SIZE) * bytes_per_float
        est_eval_tile_bytes = int(effective_batch) * int(MAX_TILES_EVAL) * 3 * int(TILE_SIZE) * int(TILE_SIZE) * bytes_per_float
        # Conservative threshold: if a single batch would pin >512MB for tiles alone, disable pinning.
        pin_threshold_bytes = 512 * 1024 * 1024
        if loader_pin and (est_train_tile_bytes > pin_threshold_bytes or est_eval_tile_bytes > pin_threshold_bytes):
            loader_pin = False
            logger.warning(
                "[DATALOADER] Disabling pin_memory: estimated tile batch is too large to pin safely. "
                f"est_train_tile_bytes={est_train_tile_bytes}, est_eval_tile_bytes={est_eval_tile_bytes}, "
                f"threshold={pin_threshold_bytes}, effective_batch={effective_batch}, TILE_SIZE={TILE_SIZE}, "
                f"MAX_TILES_TRAIN={MAX_TILES_TRAIN}, MAX_TILES_EVAL={MAX_TILES_EVAL}"
            )
    except Exception:
        # Never let estimation logic break training.
        pass
    loader_kwargs = dict(
        batch_size=effective_batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=loader_num_workers,
        pin_memory=loader_pin,
        collate_fn=tile_collate_with_padding,
        generator=train_loader_generator,
    )
    if loader_num_workers and int(loader_num_workers) > 0:
        loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset,
        **loader_kwargs,
    )
    logger.info(f"[DEBUG] Train Loader Config: batch_size={effective_batch}, shuffle={loader_kwargs.get('shuffle')}, sampler={loader_kwargs.get('sampler')}, num_workers={loader_kwargs.get('num_workers')}")
    # Validation and fixed retrieval tests run only on rank0 in DDP.
    if (not is_distributed) or rank == 0:
        eval_kwargs = dict(
            batch_size=effective_batch,
            shuffle=False,
            num_workers=loader_num_workers,
            pin_memory=loader_pin,
            collate_fn=tile_collate_with_padding,
            generator=make_loader_generator(1),
        )
        if loader_num_workers and int(loader_num_workers) > 0:
            eval_kwargs["persistent_workers"] = PERSISTENT_WORKERS
            eval_kwargs["prefetch_factor"] = PREFETCH_FACTOR
        val_loader = DataLoader(val_dataset, **eval_kwargs)
        cluster_test_loaders = {}
    else:
        val_loader = None
        cluster_test_loaders = {}

    geniza_contrastive_enabled = bool(GENIZA_CONTRASTIVE_ENABLED)
    geniza_train_loader = None
    geniza_val_loader = None
    geniza_queue = None
    geniza_contrastive_loss = None
    geniza_train_sampler = None
    geniza_demo_stats = None
    if geniza_contrastive_enabled:
        (
            geniza_train_paths,
            geniza_train_labels,
            geniza_train_xmls,
            geniza_val_paths,
            geniza_val_labels,
            geniza_val_xmls,
        ) = _load_geniza_contrastive_split_lists(BASE_DIR, GENIZA_CONTRASTIVE_TABLE)

        if args.stage == "demo":
            (
                geniza_train_paths,
                geniza_train_labels,
                geniza_train_xmls,
                train_demo_stats,
            ) = _sample_geniza_demo_split(
                geniza_train_paths,
                geniza_train_labels,
                geniza_train_xmls,
                fraction=GENIZA_DEMO_FRACTION,
                min_manuscripts=GENIZA_DEMO_MIN_TRAIN_MANUSCRIPTS,
                max_images_per_manuscript=GENIZA_DEMO_MAX_IMAGES_PER_MANUSCRIPT,
                seed=123,
            )
            (
                geniza_val_paths,
                geniza_val_labels,
                geniza_val_xmls,
                val_demo_stats,
            ) = _sample_geniza_demo_split(
                geniza_val_paths,
                geniza_val_labels,
                geniza_val_xmls,
                fraction=GENIZA_DEMO_FRACTION,
                min_manuscripts=GENIZA_DEMO_MIN_VAL_MANUSCRIPTS,
                max_images_per_manuscript=GENIZA_DEMO_MAX_IMAGES_PER_MANUSCRIPT,
                seed=456,
            )
            geniza_demo_stats = {
                "train": train_demo_stats,
                "val": val_demo_stats,
            }

        if len(geniza_train_paths) == 0 or len(geniza_val_paths) == 0:
            raise RuntimeError(
                f"Geniza contrastive is enabled but table {GENIZA_CONTRASTIVE_TABLE!r} "
                f"does not provide non-empty train/val splits."
            )
        geniza_split_overlap = sorted(set(geniza_train_labels).intersection(set(geniza_val_labels)))
        if geniza_split_overlap:
            raise RuntimeError(
                f"Geniza contrastive train/val leakage detected in {GENIZA_CONTRASTIVE_TABLE!r}: "
                f"{len(geniza_split_overlap)} overlapping manuscript_ids. Examples: {geniza_split_overlap[:10]}"
            )
        if int(effective_batch) < 2:
            raise RuntimeError(f"Geniza contrastive requires effective batch >= 2, got {effective_batch}")

        geniza_labels_combined = geniza_train_labels + geniza_val_labels
        geniza_label2idx, _geniza_idx2label = build_label_maps(all_labels=geniza_labels_combined)
        geniza_train_dataset = ManuscriptDataset(
            geniza_train_paths,
            geniza_train_labels,
            train_transform,
            geniza_label2idx,
            xml_paths=geniza_train_xmls,
            max_tiles_per_image=MAX_TILES_TRAIN,
            split='train',
        )
        geniza_val_dataset = ManuscriptDataset(
            geniza_val_paths,
            geniza_val_labels,
            eval_transform,
            geniza_label2idx,
            xml_paths=geniza_val_xmls,
            max_tiles_per_image=MAX_TILES_EVAL,
            split='val',
        )
        geniza_train_sampler = PKBatchSampler(
            geniza_train_labels,
            batch_size=int(effective_batch),
            k=int(GENIZA_PK_K),
            seed=int(TRAINING_SEED) + rank,
            drop_last=False,
        )
        geniza_loader_kwargs = dict(
            batch_sampler=geniza_train_sampler,
            num_workers=loader_num_workers,
            pin_memory=loader_pin,
            collate_fn=tile_collate_with_padding,
            generator=make_loader_generator(2),
        )
        if loader_num_workers and int(loader_num_workers) > 0:
            geniza_loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
            geniza_loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
        geniza_train_loader = DataLoader(geniza_train_dataset, **geniza_loader_kwargs)

        if (not is_distributed) or rank == 0:
            geniza_eval_kwargs = dict(
                batch_size=int(effective_batch),
                shuffle=False,
                num_workers=loader_num_workers,
                pin_memory=loader_pin,
                collate_fn=tile_collate_with_padding,
                generator=make_loader_generator(3),
            )
            if loader_num_workers and int(loader_num_workers) > 0:
                geniza_eval_kwargs["persistent_workers"] = PERSISTENT_WORKERS
                geniza_eval_kwargs["prefetch_factor"] = PREFETCH_FACTOR
            geniza_val_loader = DataLoader(geniza_val_dataset, **geniza_eval_kwargs)

        queue_capacity = min(int(GENIZA_QUEUE_SIZE), len(geniza_train_dataset))
        geniza_queue = LabeledFeatureQueue(capacity=queue_capacity, feature_dim=LATENT_DIM, device=device)
        geniza_contrastive_loss = MemoryBankSupConLoss(temperature=GENIZA_CONTRASTIVE_TEMPERATURE).to(device)
        logger.info(
            "[GENIZA CONTRASTIVE] enabled: "
            f"table={GENIZA_CONTRASTIVE_TABLE}, train_images={len(geniza_train_dataset)}, "
            f"val_images={len(geniza_val_dataset)}, train_ms={len(set(geniza_train_labels))}, "
            f"val_ms={len(set(geniza_val_labels))}, batch_size={geniza_train_sampler.batch_size}, "
            f"P={geniza_train_sampler.p}, K={geniza_train_sampler.k}, queue_capacity={queue_capacity}, "
            f"weight={GENIZA_CONTRASTIVE_WEIGHT}"
        )
        if geniza_demo_stats is not None:
            logger.info(
                "[GENIZA CONTRASTIVE][DEMO] sampled from real geniza_train_set splits: "
                f"fraction={GENIZA_DEMO_FRACTION}, max_images_per_ms={GENIZA_DEMO_MAX_IMAGES_PER_MANUSCRIPT}, "
                f"train_ms={geniza_demo_stats['train']['selected_manuscripts']}/"
                f"{geniza_demo_stats['train']['source_manuscripts']}, "
                f"train_images={geniza_demo_stats['train']['selected_images']}/"
                f"{geniza_demo_stats['train']['source_images']}, "
                f"val_ms={geniza_demo_stats['val']['selected_manuscripts']}/"
                f"{geniza_demo_stats['val']['source_manuscripts']}, "
                f"val_images={geniza_demo_stats['val']['selected_images']}/"
                f"{geniza_demo_stats['val']['source_images']}"
            )

    # 5. Model
    num_classes = len(label2idx)
    # The model is initialized for single-GPU training to resolve a deadlock
    # issue with DataParallel. To use multiple GPUs, this should be refactored
    # to use DistributedDataParallel (DDP), which is more efficient.
    # Use MultiModal with modalities specified by flags in system.py
    model = MultiModal(
        num_classes=num_classes,
        tile_size=TILE_SIZE,
    ).to(device)
    
    # Optionally initialize from checkpoint. The dataset stage is independent.
    if args.init == "checkpoint":
        logger.info(f"\n{'='*80}")
        logger.info("CHECKPOINT INIT: Loading model checkpoint")
        logger.info(f"{'='*80}")
        logger.info(f"  Checkpoint: {args.checkpoint}")
        load_ok = False
        try:
            try:
                checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(args.checkpoint, map_location="cpu")
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
                if state_dict is checkpoint:
                    logger.info(f"  Checkpoint format: raw state_dict")
                else:
                    logger.info(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}, val_accuracy: {checkpoint.get('val_accuracy', 'unknown')}")
            else:
                state_dict = checkpoint
                logger.info(f"  Checkpoint format: raw state_dict")
            transfer_state_dict = without_classification_head(state_dict)
            removed_head_keys = len(state_dict) - len(transfer_state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(transfer_state_dict, strict=False)
            num_loaded = len(transfer_state_dict) - len(unexpected_keys)
            # Classifier keys are always new. A pre-tile-summarizer checkpoint
            # may also omit the new query module; those weights are intentionally
            # initialized and learned during this transfer run.
            head_missing = [k for k in missing_keys if 'head.' in k]
            tile_summarizer_missing = [
                k for k in missing_keys
                if k.startswith('tile_branch.set_summarizer.')
            ]
            expected_missing = set(head_missing) | set(tile_summarizer_missing)
            if len(missing_keys) == len(expected_missing) and not unexpected_keys and num_loaded > 0:
                load_ok = True
            logger.info(
                f"  Loaded {num_loaded} transfer keys; removed {removed_head_keys} checkpoint classifier keys; "
                f"missing {len(missing_keys)} (classifier={len(head_missing)}, "
                f"new tile summarizer={len(tile_summarizer_missing)}); unexpected {len(unexpected_keys)}"
            )
            logger.info(f"  Classification head initialized for {num_classes} classes (not loaded from checkpoint)")
        except Exception as e:
            logger.exception(f"  Checkpoint load FAILED: {e}")
            raise RuntimeError(
                f"Checkpoint initialization failed for {args.checkpoint!r}; aborting before training."
            ) from e
        if load_ok:
            logger.info("  Checkpoint loaded successfully.")
        else:
            logger.error("  Checkpoint load FAILED or incomplete (missing keys beyond head). Check path and model compatibility.")
            raise RuntimeError(
                f"Checkpoint initialization was incomplete for {args.checkpoint!r}; aborting before training."
            )
        logger.info(f"{'='*80}\n")
    
    if is_distributed:
        # find_unused_parameters=True is safer while iterating on modality flags / conditional paths.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    elif use_dataparallel:
        logger.warning("Using torch.nn.DataParallel for multi-GPU (single process).")
        # Reorder device_ids so that aggregation_gpu is first (index 0).
        # DataParallel expects the model to be on device_ids[0].
        all_ids = list(range(dp_num_gpus))
        if args.aggregation_gpu in all_ids:
            device_ids = [args.aggregation_gpu] + [i for i in all_ids if i != args.aggregation_gpu]
        else:
            logger.warning(f"Aggregation GPU {args.aggregation_gpu} not in available IDs {all_ids}. Falling back to default order.")
            device_ids = all_ids
            
        # Use dim=0 for batch dimension splitting (default, but explicit is clearer)
        # Note: DataParallel splits batch along dim=0 automatically
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=args.aggregation_gpu, dim=0)
        logger.warning(f"DataParallel device_ids={device_ids} output_device={args.aggregation_gpu}")

    # Freeze each branch when epoch 0 is inside that branch's freeze window (pretrain and finetune)
    base_model = model.module if hasattr(model, 'module') else model

    def _epoch_in_range(epoch, from_ep, until_ep):
        return from_ep >= 0 and from_ep <= epoch <= until_ep

    freeze_tile = _epoch_in_range(0, args.tile_branch_freeze_from_epoch, args.tile_branch_freeze_until_epoch)
    freeze_glyph = _epoch_in_range(0, args.glyph_branch_freeze_from_epoch, args.glyph_branch_freeze_until_epoch)
    freeze_word = _epoch_in_range(0, args.word_branch_freeze_from_epoch, args.word_branch_freeze_until_epoch)
    if freeze_tile or freeze_glyph or freeze_word:
        logger.info(f"\n{'='*80}")
        logger.info("FREEZING BRANCHES for epoch 0 (within configured freeze ranges):")
        if freeze_tile:
            logger.info(f"  Tile branch: epochs {args.tile_branch_freeze_from_epoch}-{args.tile_branch_freeze_until_epoch}")
        if freeze_glyph:
            logger.info(f"  Glyph branch: epochs {args.glyph_branch_freeze_from_epoch}-{args.glyph_branch_freeze_until_epoch}")
        if freeze_word:
            logger.info(f"  Word branch: epochs {args.word_branch_freeze_from_epoch}-{args.word_branch_freeze_until_epoch}")
        logger.info(f"{'='*80}")
        frozen_counts = {"tile": 0, "glyph": 0, "word": 0}
        for name, param in base_model.named_parameters():
            if "head.classifier" in name or "head.arcface_head" in name:
                param.requires_grad = True
                continue
            if name.startswith("tile_branch.") and freeze_tile:
                param.requires_grad = False
                frozen_counts["tile"] += param.numel()
            elif name.startswith("glyph_branch.") and freeze_glyph:
                param.requires_grad = False
                frozen_counts["glyph"] += param.numel()
            elif name.startswith("word_branch.") and freeze_word:
                param.requires_grad = False
                frozen_counts["word"] += param.numel()
        logger.info(f"Frozen: tile {frozen_counts['tile']:,}, glyph {frozen_counts['glyph']:,}, word {frozen_counts['word']:,}")
        logger.info(f"{'='*80}\n")

    # 6. Combined Loss, Optimizer, and Scheduler
    combined_loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=LATENT_DIM,
        arcface_weight=ARCFACE_WEIGHT,
        ce_weight=CE_WEIGHT,
        arcface_margin=ARCFACE_MARGIN,
        arcface_scale=ARCFACE_SCALE,
        sparsity_weight=LATENT_SPARSITY_WEIGHT,
        tile_aux_weight=TILE_AUX_LOSS_WEIGHT,
        glyph_aux_weight=GLYPH_AUX_LOSS_WEIGHT,
        fusion_aux_weight=FUSION_AUX_LOSS_WEIGHT,
        word_aux_weight=WORD_AUX_LOSS_WEIGHT,
        word_aux_full_weight_epochs=WORD_AUX_FULL_WEIGHT_EPOCHS,
        word_aux_decay_end_epoch=WORD_AUX_DECAY_END_EPOCH,
        aux_embedding_dim=(
            SYMMETRIC_BRANCH_DIM
            if FUSION_METHOD == "symmetric"
            and USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
            else D_MODEL
        ),
        shared_branch_arcface=SHARED_BRANCH_ARCFACE if FUSION_METHOD == "symmetric" else False,
        gate_entropy_weight=GATE_ENTROPY_WEIGHT if FUSION_METHOD == "symmetric" else 0.0,
        gate_entropy_decay_epochs=GATE_ENTROPY_DECAY_EPOCHS,
    ).to(device)
    
    # Create differential-LR groups. Frozen parameters remain tracked so an
    # epoch-range unfreeze can resume them without rebuilding the optimizer and
    # discarding Adam moments or LR ratios.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    loss_params = [p for p in combined_loss.parameters() if p.requires_grad]
    logger.info(f"\nOptimizer configuration:")
    logger.info(f"  Trainable model parameters: {sum(p.numel() for p in trainable_params):,}")
    logger.info(f"  Trainable loss parameters (ArcFace, etc.): {sum(p.numel() for p in loss_params):,}")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use lower learning rate only when initializing from an existing checkpoint.
    if args.init == "checkpoint":
        learning_rate = 1e-4
        logger.info(f"  Learning rate: {learning_rate} (checkpoint initialization)")
    else:
        learning_rate = LEARNING_RATE_STAGE1
        logger.info(f"  Learning rate: {learning_rate} (scratch initialization)")
    
    optimizer_groups = build_optimizer_param_groups(
        model,
        combined_loss,
        base_lr=learning_rate,
        alephbert_lr=LEARNING_RATE_ALEPHBERT,
        new_word_lr=LEARNING_RATE_NEW_WORD,
    )
    for group in optimizer_groups:
        logger.info(
            "  %-10s lr=%.8f parameters=%s",
            group["name"],
            group["lr"],
            f"{sum(parameter.numel() for parameter in group['params']):,}",
        )
    optimizer = optim.AdamW(optimizer_groups, weight_decay=WEIGHT_DECAY, betas=BETAS)
    training_num_epochs = args.num_epochs if args.num_epochs is not None else NUM_EPOCHS
    scheduler = build_ratio_preserving_cosine_scheduler(
        optimizer,
        total_epochs=training_num_epochs,
        base_lr=learning_rate,
        eta_min=SCHEDULER_ETA_MIN,
    )

    # 7. Report dataset/GPU info
    gpu_info = {
        'cuda_is_available': torch.cuda.is_available(),
        'num_devices': torch.cuda.device_count(),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
    }
    if torch.cuda.is_available():
        gpu_info['current_device.index'] = torch.cuda.current_device()
        gpu_info['current_device.name'] = torch.cuda.get_device_name(torch.cuda.current_device())
    for i in range(torch.cuda.device_count()):
        gpu_info[f'device_{i}.name'] = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        gpu_info[f'device_{i}.total_memory_GB'] = round(props.total_memory / (1024**3), 2)

    # Build dataset_info safely in DDP: val/test loaders may be None on non-rank0.
    if (not is_distributed) or rank == 0:
        dataset_info = {
        'train.num_images': len(train_loader.dataset),
            'val.num_images': len(val_loader.dataset) if val_loader is not None else 0,
        'classes.num_classes': num_classes,
        'filter.total_manuscripts': split_stats.get('total_manuscripts', 0),
        'filter.eligible_manuscripts': split_stats.get('eligible_manuscripts', 0),
        'filter.selected_manuscripts': split_stats.get('selected_manuscripts', 0),
        'filter.images_total_selected': split_stats.get('images_total_selected', 0),
        'filter.images_expected': split_stats.get('images_expected', 0),
        'filter.images_dropped_non_color': split_stats.get('images_dropped_non_color', 0),
        'filter.color_flags_present': ','.join(map(str, split_stats.get('color_flags_present', []))),
        'filter.selected_examples_ids': ','.join(map(str, split_stats.get('selected_examples_ids', []))),
        'filter.selected_examples_detailed': split_stats.get('selected_examples_detailed', []),
        'split.strategy': split_stats.get('split_strategy', 'unknown'),
        'tiling.patch_size': train_loader.dataset.patch_size,
        'tiling.stride': train_loader.dataset.stride,
        'tiling.max_tiles_train': MAX_TILES_TRAIN,
        'tiling.max_tiles_eval': MAX_TILES_EVAL,
        'tiling.center_bias': True,
        'loader.batch_size': train_loader.batch_size,
        'loader.num_workers': train_loader.num_workers,
    }
        if geniza_contrastive_enabled:
            dataset_info.update({
                'geniza.mode': args.geniza,
                'geniza.table': GENIZA_CONTRASTIVE_TABLE,
                'geniza.train.num_images': len(geniza_train_loader.dataset) if geniza_train_loader is not None else 0,
                'geniza.val.num_images': len(geniza_val_loader.dataset) if geniza_val_loader is not None else 0,
                'geniza.queue_capacity': geniza_queue.capacity if geniza_queue is not None else 0,
            })
            if geniza_demo_stats is not None:
                dataset_info.update({
                    'geniza.demo.enabled': True,
                    'geniza.demo.fraction': GENIZA_DEMO_FRACTION,
                    'geniza.demo.max_images_per_manuscript': GENIZA_DEMO_MAX_IMAGES_PER_MANUSCRIPT,
                    'geniza.demo.train.source_manuscripts': geniza_demo_stats['train']['source_manuscripts'],
                    'geniza.demo.train.selected_manuscripts': geniza_demo_stats['train']['selected_manuscripts'],
                    'geniza.demo.train.source_images': geniza_demo_stats['train']['source_images'],
                    'geniza.demo.train.selected_images': geniza_demo_stats['train']['selected_images'],
                    'geniza.demo.val.source_manuscripts': geniza_demo_stats['val']['source_manuscripts'],
                    'geniza.demo.val.selected_manuscripts': geniza_demo_stats['val']['selected_manuscripts'],
                    'geniza.demo.val.source_images': geniza_demo_stats['val']['source_images'],
                    'geniza.demo.val.selected_images': geniza_demo_stats['val']['selected_images'],
                })
        elif args.geniza == "classification":
            dataset_info.update({
                'geniza.mode': args.geniza,
                'geniza.table': GENIZA_CONTRASTIVE_TABLE,
                'geniza.classification.train.num_images': geniza_classification_stats['images_train'],
                'geniza.classification.val.num_images': geniza_classification_stats['images_val'],
                'geniza.classification.num_manuscripts': geniza_classification_stats['manuscripts_total'],
            })
        else:
            dataset_info['geniza.mode'] = args.geniza
    else:
        dataset_info = {}
    
    # Build modules info (which modalities/encoders are active)
    if (not is_distributed) or rank == 0:
        modules_info = {
            'modalities.use_visual': USE_VISUAL_MOD,
            'modalities.use_char': USE_CHAR_MOD,
            'modalities.use_word': USE_WORD_MOD,
            'fusion.method': FUSION_METHOD,
            'fusion.latent_dim': LATENT_DIM,
            'fusion.branch_adapters_and_summarizers': USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
            'fusion.branch_dim': (
                SYMMETRIC_BRANCH_DIM
                if FUSION_METHOD == "symmetric"
                and USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
                else D_MODEL if FUSION_METHOD == "symmetric" else None
            ),
            'encoder.tile_summary_tokens': (
                TILE_NUM_SUMMARY_TOKENS
                if USE_VISUAL_MOD and USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
                else 0
            ),
            'encoder.visual_model': ENCODER_MODEL_NAME if USE_VISUAL_MOD else None,
            'encoder.glyph_type': GLYPH_ENCODER_TYPE if USE_CHAR_MOD else None,
            'encoder.glyph_summary_tokens': (
                GLYPH_NUM_SUMMARY_TOKENS
                if USE_CHAR_MOD and USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
                else 0
            ),
            'encoder.word_summary_tokens': (
                WORD_NUM_SUMMARY_TOKENS
                if USE_WORD_MOD and USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
                else 0
            ),
            'encoder.glyph_min_conf': OCR_GLYPH_CONFIDENCE_THRESHOLD if USE_CHAR_MOD else None,
            'encoder.word_tfidf_gating': WORD_BRANCH_USE_TFIDF_GATING if USE_WORD_MOD else None,
            # Static modality dropout configuration (logged as hyperparameters, not time series).
            'dropout.enabled': MODALITY_DROPOUT_ENABLED,
            'dropout.prob_visual': MODALITY_DROPOUT_PROB_VISUAL,
            'dropout.prob_char': MODALITY_DROPOUT_PROB_CHAR,
            'dropout.prob_word': MODALITY_DROPOUT_PROB_WORD,
            # Auxiliary loss weights per modality.
            'aux_loss.tile_weight': TILE_AUX_LOSS_WEIGHT,
            'aux_loss.glyph_weight': GLYPH_AUX_LOSS_WEIGHT,
            'aux_loss.fusion_weight': FUSION_AUX_LOSS_WEIGHT,
            'aux_loss.word_weight': WORD_AUX_LOSS_WEIGHT,
            'aux_loss.word_full_weight_epochs': WORD_AUX_FULL_WEIGHT_EPOCHS,
            'aux_loss.word_decay_end_epoch': WORD_AUX_DECAY_END_EPOCH,
            'optimizer.seed': TRAINING_SEED,
            'optimizer.base_lr': learning_rate,
            'optimizer.alephbert_lr': LEARNING_RATE_ALEPHBERT,
            'optimizer.new_word_lr': LEARNING_RATE_NEW_WORD,
            'geniza.mode': args.geniza,
            'geniza_contrastive.enabled': geniza_contrastive_enabled,
            'geniza_contrastive.weight': GENIZA_CONTRASTIVE_WEIGHT,
            'geniza_contrastive.temperature': GENIZA_CONTRASTIVE_TEMPERATURE,
            'geniza_contrastive.pk_k': GENIZA_PK_K,
            'geniza_contrastive.demo_fraction': GENIZA_DEMO_FRACTION,
            'geniza_contrastive.demo_max_images_per_manuscript': GENIZA_DEMO_MAX_IMAGES_PER_MANUSCRIPT,
        }
        # Create a summary string of active modules
        active_modules = []
        if USE_VISUAL_MOD:
            active_modules.append(f"Visual({ENCODER_MODEL_NAME})")
        if USE_CHAR_MOD:
            char_desc = f"Char({GLYPH_ENCODER_TYPE}, tokens={GLYPH_NUM_SUMMARY_TOKENS}, min_conf={OCR_GLYPH_CONFIDENCE_THRESHOLD})"
            active_modules.append(char_desc)
        if USE_WORD_MOD:
            active_modules.append("Word")
        modules_info['modalities.active_summary'] = " + ".join(active_modules) if active_modules else "None"
    else:
        modules_info = {}
    
    # The `start_info` dictionary, which contains all static configuration,
    # is now passed directly to wandb.init's `config` parameter in the `train` function.
    # This ensures that these values are treated as hyperparameters and displayed
    # in a table in the run's "Overview" tab, not as time-series graphs.
    start_info = {'gpu': gpu_info, 'dataset': dataset_info, 'modules': modules_info} if ((not is_distributed) or rank == 0) else None

    # Pretty print
    def print_pretty_info(gpu: dict, data: dict, modules: dict):
        print("\n" + "=" * 80)
        print("GPU Info")
        print("=" * 80)
        print(f"- CUDA available: {gpu.get('cuda_is_available')}")
        print(f"- Num devices:   {gpu.get('num_devices')}")
        cvd = gpu.get('CUDA_VISIBLE_DEVICES', '')
        print(f"- CUDA_VISIBLE_DEVICES: '{cvd}'")
        if gpu.get('cuda_is_available'):
            idx = gpu.get('current_device.index')
            name = gpu.get('current_device.name')
            print(f"- Current device: {idx} ({name})")
        # Per-device lines
        nd = int(gpu.get('num_devices') or 0)
        if nd > 0:
            print("- Devices:")
            for i in range(nd):
                n = gpu.get(f'device_{i}.name', 'unknown')
                mem = gpu.get(f'device_{i}.total_memory_GB', 'n/a')
                print(f"  - [{i}] {n} — {mem} GB")

        print("\n" + "=" * 80)
        print("Dataset Info")
        print("=" * 80)
        print(f"- Train images:  {data.get('train.num_images')}")
        print(f"- Val images:    {data.get('val.num_images')}")
        print(f"- Num classes:   {data.get('classes.num_classes')}")
        print(f"- Manuscripts:   total={data.get('filter.total_manuscripts')}, eligible(≥100)={data.get('filter.eligible_manuscripts')}, selected(top {MAX_SELECTED_MANUSCRIPTS})={data.get('filter.selected_manuscripts')}")
        print(f"- Images sel.:   total_selected={data.get('filter.images_total_selected')} (expected={data.get('filter.images_expected')})")
        print(f"- Split policy:  {data.get('split.strategy')}")
        print(f"- Color check:   dropped_non_color={data.get('filter.images_dropped_non_color')}, flags={data.get('filter.color_flags_present')}")
        ids_line = data.get('filter.selected_examples_ids', '')
        if ids_line:
            print(f"- Example IDs:   {ids_line}")
        detailed = data.get('filter.selected_examples_detailed', [])
        if detailed:
            print(f"- Examples (5x2):")
            for mid, paths in detailed[:5]:
                p1 = paths[0] if len(paths) > 0 else ''
                p2 = paths[1] if len(paths) > 1 else ''
                print(f"  - {mid}:\n      * {p1}\n      * {p2}")
        print(f"- Tiling:        patch_size={data.get('tiling.patch_size')}, stride={data.get('tiling.stride')}, center_bias={data.get('tiling.center_bias')}")
        print(f"- Max tiles/img: train={data.get('tiling.max_tiles_train')}, eval={data.get('tiling.max_tiles_eval')}")
        print(f"- Loader:        batch_size={data.get('loader.batch_size')}, num_workers={data.get('loader.num_workers')}")
        
        print("\n" + "=" * 80)
        print("Active Modules")
        print("=" * 80)
        print(f"- Visual:        {modules.get('modalities.use_visual', False)}")
        if modules.get('modalities.use_visual', False):
            print(f"  Model:         {modules.get('encoder.visual_model', 'N/A')}")
        print(f"- Character:     {modules.get('modalities.use_char', False)}")
        if modules.get('modalities.use_char', False):
            print(f"  Encoder:       {modules.get('encoder.glyph_type', 'N/A')}")
            print(f"  Summary Toks:  {modules.get('encoder.glyph_summary_tokens', 'N/A')}")
            print(f"  Min Conf:      {modules.get('encoder.glyph_min_conf', 'N/A')}")
            print(f"  Quality:       area>={GLYPH_QUALITY_MIN_AREA}, w>={GLYPH_QUALITY_MIN_WIDTH}, h>={GLYPH_QUALITY_MIN_HEIGHT}, aspect<={GLYPH_QUALITY_MAX_ASPECT_RATIO}")
        print(f"- Word:          {modules.get('modalities.use_word', False)}")
        print(f"- Summary:       {modules.get('modalities.active_summary', 'None')}")
        print("=" * 80 + "\n")

    if (not is_distributed) or rank == 0:
        print_pretty_info(gpu_info, dataset_info, modules_info)

    # Check available GPU memory before training
    if (not is_distributed) or rank == 0:
        print("\n" + "=" * 80)
        print("GPU Memory Check")
        print("=" * 80)
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free = total - reserved
            print(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {free:.1f}GB free out of {total:.1f}GB")
        print("=" * 80 + "\n")

    # 8. Train
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        combined_loss,
        num_epochs=training_num_epochs,
        tile_branch_freeze_from_epoch=args.tile_branch_freeze_from_epoch,
        tile_branch_freeze_until_epoch=args.tile_branch_freeze_until_epoch,
        glyph_branch_freeze_from_epoch=args.glyph_branch_freeze_from_epoch,
        glyph_branch_freeze_until_epoch=args.glyph_branch_freeze_until_epoch,
        word_branch_freeze_from_epoch=args.word_branch_freeze_from_epoch,
        word_branch_freeze_until_epoch=args.word_branch_freeze_until_epoch,
        lr_stage2=LEARNING_RATE_STAGE2,
        project_name=PROJECT_NAME,
        run_start_info=start_info if ((not is_distributed) or rank == 0) else None,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        idx2label=idx2label,
        training_mode=TRAINING_MODE,
        device=device,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        train_sampler=train_sampler,
        geniza_train_loader=geniza_train_loader,
        geniza_val_loader=geniza_val_loader,
        geniza_train_sampler=geniza_train_sampler,
        geniza_contrastive_loss=geniza_contrastive_loss,
        geniza_feature_queue=geniza_queue,
        geniza_contrastive_weight=GENIZA_CONTRASTIVE_WEIGHT if geniza_contrastive_enabled else 0.0,
        cluster_test_loaders=cluster_test_loaders,
    )

    if is_distributed:
        dist.destroy_process_group()
