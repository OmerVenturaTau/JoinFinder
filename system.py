"""
Global configuration for JoinsFinder.

This file is intentionally *constants-only*:
- Put configuration knobs here (with short comments).
- Avoid heavy imports or runtime side effects.
- Derived values (like RUN_NAME) are computed near the bottom so dependencies are obvious.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path


_SYSTEM_DIR = Path(__file__).resolve().parent
MODEL_ARCHITECTURE_CONFIG_PATH = _SYSTEM_DIR / "config" / "model_architecture.json"
MODEL_REGULARIZATION_CONFIG_PATH = _SYSTEM_DIR / "config" / "model_regularization.json"
TRAINING_CONFIG_PATH = _SYSTEM_DIR / "config" / "training.json"
LOSS_CONFIG_PATH = _SYSTEM_DIR / "config" / "loss.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _cfg(cfg: dict, path: str):
    value = cfg
    for part in path.split("."):
        value = value[part]
    return value


def _deep_merge(*configs: dict) -> dict:
    merged: dict = {}
    for config in configs:
        for key, value in config.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
    return merged


def _ensure(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(f"[CONFIG] {message}")


def _ensure_prob(name: str, value: float) -> None:
    _ensure(0.0 <= float(value) <= 1.0, f"{name} must be in [0, 1], got {value!r}")


def _ensure_positive_int(name: str, value: int) -> None:
    _ensure(int(value) > 0, f"{name} must be > 0, got {value!r}")


def _ensure_range(name: str, value: tuple | list, *, lo: float | None = None, hi: float | None = None) -> None:
    _ensure(len(value) == 2, f"{name} must have exactly 2 values, got {value!r}")
    a, b = float(value[0]), float(value[1])
    _ensure(a <= b, f"{name} must be ordered low<=high, got {value!r}")
    if lo is not None:
        _ensure(a >= lo, f"{name} lower bound must be >= {lo}, got {value!r}")
    if hi is not None:
        _ensure(b <= hi, f"{name} upper bound must be <= {hi}, got {value!r}")


MODEL_ARCHITECTURE_CONFIG = _load_json(MODEL_ARCHITECTURE_CONFIG_PATH)
MODEL_REGULARIZATION_CONFIG = _load_json(MODEL_REGULARIZATION_CONFIG_PATH)
TRAINING_CONFIG = _load_json(TRAINING_CONFIG_PATH)
LOSS_CONFIG = _load_json(LOSS_CONFIG_PATH)
MODEL_CONFIG = _deep_merge(
    MODEL_ARCHITECTURE_CONFIG,
    MODEL_REGULARIZATION_CONFIG,
    TRAINING_CONFIG,
    LOSS_CONFIG,
)

# =============================================================================
# 1) Paths / DB
# =============================================================================

# INI file used by `train/db_loader.py` to connect to the manuscript metadata DB.
DB_CONFIG_PATH = "db_config.ini"

# Root folder containing manuscript images (used by split builder + datasets).
BASE_DIR = "/nas/archive/NLI_MANUSCRIPTS_jpgs/"


# =============================================================================
# 2) Dataset splits (DB-table driven)
# =============================================================================

# Split ratios within each manuscript (used for partitioning images in the DB tables).
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =============================================================================
# 3) Preprocessing (Preparation of DB tables and patch coordinates)
# =============================================================================

# Only consider manuscripts that have at least this many "colored" pages in the DB.
MIN_COLORED_IMAGES_PER_MANUSCRIPT = 30


# =============================================================================
# 4) Tiling / patch extraction (visual tiles)
# =============================================================================

# Tile size in pixels (tile encoder input resolution).
TILE_SIZE = 560

# Stride between tile centers in pixels.
TILE_STRIDE = 420

# If True, prioritize selecting tiles near the page center first when limiting max tiles.
CENTER_BIAS = True

# Max tiles per image to cap memory/time.
# Keep visual branch focused on a small set of high-text tiles; this reduces
# reliance on page/background/library cues during Geniza projection.
MAX_TILES_TRAIN = 8
MAX_TILES_EVAL = 8

# Patch coordinate source:
# - "db": DB only (fails if missing)
# - "extract": XML-only extraction each time
# - "auto": DB then XML fallback
PATCH_LOADING_METHOD = "extract"

# If True, tile extraction may fall back to center/grid tiles when XML-guided
# extraction cannot produce candidates. If False, XML-guided extraction with
# text regions returns an empty visual stream instead of background/grid tiles.
TILES_FALLBACK_GRID = True

# XML-based tile extraction heuristics (only relevant when extracting tile coords from XML).
XML_PATCH_STRIDE_MULTIPLIER = 0.7
XML_PATCH_OVERLAP_THRESHOLD = 0.2
XML_PATCH_RELAXED_THRESHOLDS = [0.3, 0.4, 0.5, 0.6]
XML_PATCH_EXPAND_RATIO = 0.0
XML_PATCH_CONSTRAIN_TO_BOUNDS = True
XML_PATCH_FILTER_BY_TEXT_COVERAGE = True
XML_PATCH_MIN_TEXT_COVERAGE = 0.95
# Ink-aware ranking: prefer tiles with the most actual ink (dark pixels)
# instead of just the smallest distance to a text-region center. This
# pushes the selector toward dense text rows and away from blank parchment.
XML_PATCH_RANK_BY_INK_DENSITY = True
# Grayscale value below which a pixel is considered ink. Manuscripts are
# dark ink on lighter parchment; tune higher to catch faded ink, lower to
# ignore mid-tone parchment stains.
XML_PATCH_INK_THRESHOLD = 160
# Drop candidate tiles with less ink than this fraction (0..1). Removes
# mostly-blank patches that still pass the bbox-coverage filter because
# TextLine bboxes include ascender/descender slack.
XML_PATCH_MIN_INK_RATIO = 0.04
XML_PATCH_APPLY_READING_ORDER = True
XML_PATCH_READING_DIRECTION_RTL = True

# If True, detect/correct page rotation before XML patch extraction.
XML_PATCH_DETECT_ROTATION = True


# =============================================================================
# 5) Model architecture (main model)
# =============================================================================

# Tile-level encoder backbone (timm model name).
ENCODER_MODEL_NAME = _cfg(MODEL_CONFIG, "tile.encoder_model_name")

# Tile encoder selection.
# - "dinov2": ViT backbone (default, uses ENCODER_MODEL_NAME)
# - "convnext": ConvNeXt backbone (uses TILE_CONVNEXT_MODEL_NAME by default)
TILE_ENCODER_TYPE = _cfg(MODEL_CONFIG, "tile.encoder_type")
# A solid ConvNeXt default in timm. You can override by passing a different
# model name into TileBranch/MultiModal (or by changing this constant).
TILE_CONVNEXT_MODEL_NAME = _cfg(MODEL_CONFIG, "tile.convnext_model_name")

# Fusion transformer / aggregator dimensions.
# Shared hidden size used by branch projections and fusion tokens.
D_MODEL = _cfg(MODEL_CONFIG, "fusion.d_model")
# Fusion method and optional legacy-style branch compression path. The single
# switch enables learned set summarizers in every branch and the per-modality
# projection adapters in symmetric fusion.
FUSION_METHOD = _cfg(MODEL_CONFIG, "fusion.method")
USE_BRANCH_ADAPTERS_AND_SUMMARIZERS = bool(
    _cfg(MODEL_CONFIG, "fusion.use_branch_adapters_and_summarizers")
)
SYMMETRIC_BRANCH_DIM = int(_cfg(MODEL_CONFIG, "fusion.symmetric_branch_dim"))
# Number of fusion blocks in the main multimodal stack.
NUM_LAYERS = _cfg(MODEL_CONFIG, "fusion.num_layers")
# Attention heads per fusion block.
NUM_HEADS = _cfg(MODEL_CONFIG, "fusion.num_heads")
# Dropout used in fusion / branch transformer modules.
DROPOUT = _cfg(MODEL_CONFIG, "fusion.dropout")

# Page-level representation (also used for clustering/export). Symmetric
# fusion has a structurally determined width, so changing the architecture
# flag is sufficient; no second latent-dimension setting must be synchronized.
_CONFIGURED_LATENT_DIM = int(_cfg(MODEL_CONFIG, "fusion.latent_dim"))
if FUSION_METHOD == "symmetric":
    _configured_modality_count = sum(
        bool(_cfg(MODEL_CONFIG, f"modalities.{name}"))
        for name in ("use_visual", "use_char", "use_word")
    )
    _symmetric_output_width = (
        SYMMETRIC_BRANCH_DIM
        if USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
        else int(D_MODEL)
    )
    LATENT_DIM = _symmetric_output_width * _configured_modality_count
else:
    LATENT_DIM = _CONFIGURED_LATENT_DIM


# =============================================================================
# 6) Training (optimizer/schedule/dataloader)
# =============================================================================

# Base batch size. Interpretation:
# - DDP: per-process/per-GPU batch size
# - DataParallel: `main.py` scales by #GPUs so each GPU sees ~PER_GPU_BATCH_SIZE; global batch = PER_GPU_BATCH_SIZE * num_gpus.
# For 2×46GB GPUs: 2 per GPU avoids OOM. Effective batch = 2*2*4 = 16 (fine for training).
# Bump PER_GPU_BATCH_SIZE to 3 if you get headroom; or use accum 6 for effective 24.
PER_GPU_BATCH_SIZE = _cfg(MODEL_CONFIG, "training.per_gpu_batch_size")

# Gradient accumulation steps (effective batch = global_batch * this).
# With 2 GPUs, PER_GPU_BATCH_SIZE=2: global 4/step; accum 4 → effective 16; accum 6 → effective 24.
GRADIENT_ACCUMULATION_STEPS = _cfg(MODEL_CONFIG, "training.gradient_accumulation_steps")
# Clip accumulated gradients before each optimizer step. Set <= 0 to disable.
GRADIENT_CLIP_NORM = _cfg(MODEL_CONFIG, "training.gradient_clip_norm")

# DataLoader settings.
# Number of worker processes used by DataLoaders.
NUM_WORKERS = _cfg(MODEL_CONFIG, "training.num_workers")
# Number of prefetched batches per worker.
PREFETCH_FACTOR = _cfg(MODEL_CONFIG, "training.prefetch_factor")
# If True, pin CPU memory before GPU transfer.
PIN_MEMORY = _cfg(MODEL_CONFIG, "training.pin_memory")
# If True, keep worker processes alive across epochs.
PERSISTENT_WORKERS = _cfg(MODEL_CONFIG, "training.persistent_workers")

# Single-process multi-GPU for `main.py` (avoids torchrun/DDP). Global batch = PER_GPU_BATCH_SIZE * num_gpus (multiple of GPU count).
MAIN_USE_DATAPARALLEL = _cfg(MODEL_CONFIG, "training.main_use_dataparallel")

# In DataParallel mode, DataLoader multiprocessing can be less stable; override here.
# On 4×A40, a small pool of workers helps keep GPUs fed (XML + PIL can be CPU/I/O bound).
# Tune if needed: try 4/8/12 depending on CPU cores and storage throughput.
MAIN_DATAPARALLEL_NUM_WORKERS = _cfg(MODEL_CONFIG, "training.main_dataparallel_num_workers")

# Optimizer settings.
# Reproducible model initialization, sampling, workers, and augmentations.
TRAINING_SEED = int(_cfg(MODEL_CONFIG, "training.seed"))
# Learning rate for the first training stage.
LEARNING_RATE_STAGE1 = _cfg(MODEL_CONFIG, "training.learning_rate_stage1")
# Learning rate for the second training stage.
LEARNING_RATE_STAGE2 = _cfg(MODEL_CONFIG, "training.learning_rate_stage2")
# Lower rate for the pretrained language encoder and higher rate for the newly
# initialized word pooling/summarization/adapter layers.
LEARNING_RATE_ALEPHBERT = _cfg(MODEL_CONFIG, "training.learning_rate_alephbert")
LEARNING_RATE_NEW_WORD = _cfg(MODEL_CONFIG, "training.learning_rate_new_word")
# AdamW weight decay.
WEIGHT_DECAY = _cfg(MODEL_CONFIG, "training.weight_decay")
# Adam/AdamW beta coefficients.
BETAS = tuple(_cfg(MODEL_CONFIG, "training.betas"))

# Training duration.
NUM_EPOCHS = _cfg(MODEL_CONFIG, "training.num_epochs")

# Learning rate scheduler (CosineAnnealingLR).
SCHEDULER_ETA_MIN = _cfg(MODEL_CONFIG, "training.scheduler_eta_min")

# Transformer Fusion Settings
SYMMETRIC_RELIABILITY_HIDDEN_DIM = int(_cfg(MODEL_CONFIG, "fusion.symmetric_reliability_hidden_dim"))
# Number of layers in the fusion transformer implementation.
TRANSFORMER_NUM_LAYERS = _cfg(MODEL_CONFIG, "fusion.transformer_num_layers")
# Number of heads in the fusion transformer implementation.
TRANSFORMER_NUM_HEADS = _cfg(MODEL_CONFIG, "fusion.transformer_num_heads")
# Feedforward width inside the fusion transformer blocks.
TRANSFORMER_DIM_FEEDFORWARD = _cfg(MODEL_CONFIG, "fusion.transformer_dim_feedforward")
# Dropout inside the fusion transformer blocks.
TRANSFORMER_DROPOUT = _cfg(MODEL_CONFIG, "fusion.transformer_dropout")
# If enabled, TransformerFusion exports a residual blend of the branch-pooled
# token means and the contextual CLS token. This gives retrieval a guaranteed
# branch-geometry path while still letting attention add cross-modal context.
try:
    TRANSFORMER_USE_RESIDUAL_POOL = bool(_cfg(MODEL_CONFIG, "fusion.transformer_use_residual_pool"))
except KeyError:
    TRANSFORMER_USE_RESIDUAL_POOL = False
try:
    TRANSFORMER_RESIDUAL_CLS_GATE = float(_cfg(MODEL_CONFIG, "fusion.transformer_residual_cls_gate"))
except KeyError:
    TRANSFORMER_RESIDUAL_CLS_GATE = 1.0
try:
    TRANSFORMER_RESIDUAL_TILE_WEIGHT = float(_cfg(MODEL_CONFIG, "fusion.transformer_residual_tile_weight"))
except KeyError:
    TRANSFORMER_RESIDUAL_TILE_WEIGHT = 1.0
try:
    TRANSFORMER_RESIDUAL_GLYPH_WEIGHT = float(_cfg(MODEL_CONFIG, "fusion.transformer_residual_glyph_weight"))
except KeyError:
    TRANSFORMER_RESIDUAL_GLYPH_WEIGHT = 1.0
try:
    TRANSFORMER_RESIDUAL_WORD_WEIGHT = float(_cfg(MODEL_CONFIG, "fusion.transformer_residual_word_weight"))
except KeyError:
    TRANSFORMER_RESIDUAL_WORD_WEIGHT = 1.0

# Tile Set Transformer Settings (post-DINOv2, pre-fusion)
# This processes the set of tile tokens to learn spatial relationships.
# Enable an intra-tile transformer before multimodal fusion.
TILE_BRANCH_USE_TRANSFORMER = _cfg(MODEL_CONFIG, "tile.use_transformer")
# Number of layers in the tile-set transformer.
TILE_BRANCH_TRANSFORMER_LAYERS = _cfg(MODEL_CONFIG, "tile.transformer_layers")
# Number of heads in the tile-set transformer.
TILE_BRANCH_TRANSFORMER_HEADS = _cfg(MODEL_CONFIG, "tile.transformer_heads")
# Feedforward width inside the tile-set transformer.
TILE_BRANCH_TRANSFORMER_DIM_FEEDFORWARD = _cfg(MODEL_CONFIG, "tile.transformer_dim_feedforward")
# Dropout inside the tile-set transformer.
TILE_BRANCH_TRANSFORMER_DROPOUT = _cfg(MODEL_CONFIG, "tile.transformer_dropout")
try:
    # Learned summary-query count is independent of the number of input tiles.
    # Cross-attention supports more queries than keys. Every query remains
    # active for a nonempty modality; the source mask excludes padded tiles
    # from the pre-softmax attention scores.
    TILE_NUM_SUMMARY_TOKENS = int(_cfg(MODEL_CONFIG, "tile.num_summary_tokens"))
except KeyError:
    TILE_NUM_SUMMARY_TOKENS = 0
try:
    TILE_MIN_SUMMARY_TOKENS = int(_cfg(MODEL_CONFIG, "tile.min_summary_tokens"))
except KeyError:
    TILE_MIN_SUMMARY_TOKENS = 1
try:
    TILE_SUMMARIZER_CROSS_ATTN_LAYERS = int(
        _cfg(MODEL_CONFIG, "tile.summarizer_cross_attn_layers")
    )
except KeyError:
    TILE_SUMMARIZER_CROSS_ATTN_LAYERS = 1
try:
    TILE_SUMMARIZER_ATTENTION_HEADS = int(
        _cfg(MODEL_CONFIG, "tile.summarizer_attention_heads")
    )
except KeyError:
    TILE_SUMMARIZER_ATTENTION_HEADS = TILE_BRANCH_TRANSFORMER_HEADS

# Loss weights.
# ArcFace loss (replaces CrossEntropy)
ARCFACE_WEIGHT = _cfg(MODEL_CONFIG, "loss.arcface_weight")  # ArcFace directly optimises angular margins (cosine-distance retrieval)
ARCFACE_MARGIN = _cfg(MODEL_CONFIG, "loss.arcface_margin")  # Angular margin in radians (~11.5°). Keep 0.2 for pretrain (many classes). For 80-class finetune, consider 0.10–0.15 for smoother open-set Geniza.
ARCFACE_SCALE = _cfg(MODEL_CONFIG, "loss.arcface_scale")  # Slightly lower than before (30) for gentler early training dynamics.
ARCFACE_MARGIN_WARMUP_EPOCHS = _cfg(MODEL_CONFIG, "loss.arcface_margin_warmup_epochs")  # Shorter warmup (was 3) to reduce prolonged instability in the first epochs.

# CrossEntropy loss (used when ARCFACE_WEIGHT = 0)
CE_WEIGHT = _cfg(MODEL_CONFIG, "loss.ce_weight")  # Disabled — using ArcFace instead

# Geniza memory-bank supervised contrastive training.
GENIZA_CONTRASTIVE_ENABLED = _cfg(MODEL_CONFIG, "loss.geniza_contrastive_enabled")
GENIZA_CONTRASTIVE_TABLE = _cfg(MODEL_CONFIG, "loss.geniza_contrastive_table")
GENIZA_CONTRASTIVE_WEIGHT = _cfg(MODEL_CONFIG, "loss.geniza_contrastive_weight")
GENIZA_CONTRASTIVE_TEMPERATURE = _cfg(MODEL_CONFIG, "loss.geniza_contrastive_temperature")
GENIZA_QUEUE_SIZE = _cfg(MODEL_CONFIG, "loss.geniza_queue_size")
GENIZA_PK_K = _cfg(MODEL_CONFIG, "loss.geniza_pk_k")
GENIZA_DEMO_FRACTION = _cfg(MODEL_CONFIG, "loss.geniza_demo_fraction")
GENIZA_DEMO_MIN_TRAIN_MANUSCRIPTS = _cfg(MODEL_CONFIG, "loss.geniza_demo_min_train_manuscripts")
GENIZA_DEMO_MIN_VAL_MANUSCRIPTS = _cfg(MODEL_CONFIG, "loss.geniza_demo_min_val_manuscripts")
GENIZA_DEMO_MAX_IMAGES_PER_MANUSCRIPT = _cfg(MODEL_CONFIG, "loss.geniza_demo_max_images_per_manuscript")

# Regularization
LATENT_SPARSITY_WEIGHT = _cfg(MODEL_CONFIG, "loss.latent_sparsity_weight")  # L1 on latent can hurt cosine/ArcFace geometry; use 0 (or at most 1e-3) when using ArcFace

# Auxiliary Branch Losses (forces each branch to be discriminative on its own).
# Glyphs get stronger standalone pressure because the tile branch is currently
# the easier route and tends to dominate multimodal fusion.
TILE_AUX_LOSS_WEIGHT =  _cfg(MODEL_CONFIG, "loss.tile_aux_loss_weight")
GLYPH_AUX_LOSS_WEIGHT = _cfg(MODEL_CONFIG, "loss.glyph_aux_loss_weight")
try:
    FUSION_AUX_LOSS_WEIGHT = _cfg(MODEL_CONFIG, "loss.fusion_aux_loss_weight")
except KeyError:
    FUSION_AUX_LOSS_WEIGHT = 0.0
WORD_AUX_LOSS_WEIGHT = _cfg(MODEL_CONFIG, "loss.word_aux_loss_weight")     # Weight for auxiliary loss on word-only embeddings
WORD_AUX_FULL_WEIGHT_EPOCHS = int(_cfg(MODEL_CONFIG, "loss.word_aux_full_weight_epochs"))
WORD_AUX_DECAY_END_EPOCH = int(_cfg(MODEL_CONFIG, "loss.word_aux_decay_end_epoch"))
SHARED_BRANCH_ARCFACE = bool(_cfg(MODEL_CONFIG, "loss.shared_branch_arcface"))
GATE_ENTROPY_WEIGHT = float(_cfg(MODEL_CONFIG, "loss.gate_entropy_weight"))
GATE_ENTROPY_DECAY_EPOCHS = int(_cfg(MODEL_CONFIG, "loss.gate_entropy_decay_epochs"))


# =============================================================================
# 7) Task / label head
# =============================================================================

# What the classifier predicts:
# - "manuscript_id": predict manuscript ID (default)
# - "decade": predict coarse dating bucket (50-year window)
LABEL_HEAD = "manuscript_id"

# Maximum number of classes for the classifier head (depends on dataset).
# This is a safe upper bound used to initialize model heads.
MAX_SELECTED_MANUSCRIPTS = 20000


# =============================================================================
# 8) Experiment tracking (W&B) + run naming
# =============================================================================

PROJECT_NAME = "MultiModal_Manuscript_Classification"


# =============================================================================
# 9) Dating / chronology
# =============================================================================

# Aggregated dating table (per-manuscript), produced by:
#   preprocess/dating/create_manuscript_dating_db.py
# Columns include: manuscript_id, is_decade, decade, ...
DATING_MANUSCRIPT_TABLE = "sfar_data_manuscripts_dating"


# =============================================================================
# 10) Image preprocessing
# =============================================================================

# ImageNet normalization used by the tile transforms in `main.py`.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Disable PIL decompression bomb limit for very large manuscripts.
DISABLE_PIL_LIMIT = True


# =============================================================================
# 11) Memory / perf knobs (chunking)
# =============================================================================

# Controls tile encoding chunk size (larger = faster, more peak memory).
TILE_ENCODE_CHUNK_SIZE = _cfg(MODEL_CONFIG, "performance.tile_encode_chunk_size")

# Controls char encoding chunk size when many glyphs exist per page.
# Smaller = lower peak GPU memory (ConvNeXt forward), larger = faster. Tune if OOM.
# With DataParallel, each GPU processes a split batch, but model is replicated.
CHAR_ENCODE_CHUNK_SIZE = _cfg(MODEL_CONFIG, "performance.char_encode_chunk_size")


# =============================================================================
# 12) Logging / visualization / analysis
# =============================================================================

NUM_FIXED_VALIDATION_SAMPLES = 100
NUM_ATTENTION_IMAGES = 10
NUM_TOP_ERRORS_PER_EPOCH = 100

ATTENTION_IMAGES_DIR = "Results/attention_maps"

# PCA analysis outputs and knobs.
PCA_SAVE_DIR = "Results/latent_space/pca"
ENABLE_PCA_ANALYSIS = True
PCA_EVERY_EPOCHS = 1

# Optional hang watchdog for debugging (prints traceback after N seconds).
ENABLE_TRAIN_HANG_WATCHDOG = False
TRAIN_HANG_WATCHDOG_SECONDS = 15 * 60

# PCA hyperparameters.
PCA_DIMENSION = 512
PCA_MAX_SAMPLES = 1000
PCA_MAX_COMPONENTS = 512
PCA_EXPLAINED_VARIANCE_THRESHOLD = 0.95


# =============================================================================
# 13) Examples (misc demo/visualization)
# =============================================================================

# Demo mode sampling (used by --demo in main.py).
# We sample a small, balanced subset of manuscripts from oriental vs non-oriental
# and then a fixed number of pages per manuscript, to better mimic the real task.
DEMO_MANUSCRIPTS_PER_CLASS = 50       # e.g. 10 oriental + 10 non-oriental manuscripts
DEMO_IMAGES_PER_MANUSCRIPT = 40       # e.g. up to 5 pages per selected manuscript


# =============================================================================
# 14) Training mode + checkpoints
# =============================================================================

TRAINING_MODE = "stage1"  # dataset/run stage: "stage1" | "stage2" | "demo"

# Epoch range (inclusive) when each branch is frozen; applies in both pretrain and finetune.
# Set both to -1 for no freezing. E.g. (0, 0) = freeze only epoch 0; (0, 2) = freeze first 3 epochs; (10, 14) = freeze at end.
# To force tile to matter: freeze GLYPH for first 1–2 epochs (e.g. 0,1) so fusion/head must rely on tiles early; then unfreeze glyph.
# Tile branch (DINOv2 + tile transformer).
TILE_BRANCH_FREEZE_FROM_EPOCH = -1
TILE_BRANCH_FREEZE_UNTIL_EPOCH = -1
# Glyph branch (ConvNeXt/Swin + summarizer).
GLYPH_BRANCH_FREEZE_FROM_EPOCH = -1 #0
GLYPH_BRANCH_FREEZE_UNTIL_EPOCH = -1 #2 
# Word branch (transformer encoder).
WORD_BRANCH_FREEZE_FROM_EPOCH = -1
WORD_BRANCH_FREEZE_UNTIL_EPOCH = -1

# Finetune dataset: oriental-heavy mix. 0.85 = 85% oriental, 15% non-oriental "rehearsal".
# ArcFace starts fresh (not loaded from pretrain); warmup is 1 epoch (ARCFACE_MARGIN_WARMUP_EPOCHS).
FINETUNE_ORIENTAL_RATIO = 0.8

BEST_MODEL_PATH_BASE = "Results/best_model"
def build_best_model_path(training_mode: str | None = None) -> str:
    """
    Build the default checkpoint path for the current training mode.
    Kept as a function because `main.py` updates `system.TRAINING_MODE`
    after importing this module.
    """
    mode = training_mode or TRAINING_MODE
    return (
        f"{BEST_MODEL_PATH_BASE}/best_model_{mode}"
        f"_ce_weight_{CE_WEIGHT}"
        f"_arcface_weight_{ARCFACE_WEIGHT}"
        f"_{LATENT_DIM}.pth"
    )


BEST_MODEL_PATH = build_best_model_path()


def recompute_best_model_path(training_mode: str | None = None) -> None:
    global BEST_MODEL_PATH
    BEST_MODEL_PATH = build_best_model_path(training_mode)


# =============================================================================
# 15) Clustering / export
# =============================================================================

CLUSTERING_IMAGES_ROOT = "/nas/archive/NLI_GNIZA_jpgs/"
CLUSTERING_DB_CONFIG_PATH = "db_config.ini"
CLUSTERING_N_NEIGHBORS = 10

# PCA-reduced dimension used for ANN search.
CLUSTERING_SEARCH_VECTOR_DIM = 256

# DB commit batching for streaming uploads.
CLUSTERING_COMMIT_INTERVAL = 100

# Clustering tile settings (must match training for compatibility by default).
CLUSTERING_TILE_SIZE = TILE_SIZE
CLUSTERING_TILE_STRIDE = TILE_STRIDE
CLUSTERING_MAX_TILES_EVAL = MAX_TILES_EVAL

# Fixed member-level retrieval tests evaluated after every training epoch.
# Paths are resolved relative to the project root when they are not absolute.
CLUSTER_PAIRS_CSV_PATH = "results_analysis/test_set/clusters_images_metadata.csv"
CLUSTER_MEMBERS_XLSX_PATH = "results_analysis/test_set/cluster_members.xlsx"
CLUSTER_MEMBERS_XLSX_SHEET = "dj_v1.1.1_cluster_members"
CLUSTER_TEST_EXPECTED_COUNTS = {
    "clusters_images_metadata": (242, 85),
    "cluster_members": (234, 86),
}
# Both collections are small, so a modest batch fits comfortably in eval memory.
CLUSTER_PAIRS_EVAL_BATCH_SIZE = 8

# Geniza: source table (metadata per image), output tables (with vectors / neighbors), paths and ingestion settings.
GENIZA_IMAGE_INFORMATION_TABLE = "geniza_image_information"
# Table containing the currently active Geniza retrieval embeddings.
GENIZA_IMAGE_LATENTS_TABLE = "geniza_image_latents"
# Standalone interpretability projections, keyed by projection run + image.
INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE = "interpretability_experiment_vectors"
# Paths for geniza_image_information xml_path and image_path (must match update_geniza_image_information_paths).
GENIZA_XML_BASE = "/nas/archive/paris-data/automatic_transcriptions/geniza/all_04_improved_reading_order"
GENIZA_IMAGE_BASE = "/nas/archive/NLI_GNIZA_jpgs"
GENIZA_XML_FILENAME_SUFFIX = "—reco_improved_polys_improved_reading_order.xml"
# Allowed image extensions when scanning manuscript directories (add_geniza_manuscripts_to_db).
GENIZA_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
# Commit every N manuscripts when inserting into geniza_image_information.
GENIZA_COMMIT_EVERY_MANUSCRIPTS = 10
# Single flat table for KNN results (other-manuscript-only neighbors, geniza_top_neighbors.py).
GENIZA_KNN_RESULTS_TABLE = "geniza_knn_results"
# Overall neighbors table INCLUDING inter-manuscript connections (same_manuscript may be true).
GENIZA_OVERALL_NEIGHBORS_TABLE = "geniza_knn_results_including_intermanuscripts"
# Geniza manuscript shelfmark table for same-shelfmark filtering (geniza_top_neighbors_gpu.py).
GENIZA_MANUSCRIPT_SHELFMARK_TABLE = "geniza_manuscript_shelfmark"
# Curated/known Geniza join pairs used for cluster-test evaluation.
GENIZA_JOINS_TABLE = "geniza_joins"


# =============================================================================
# Results analysis (graph visualization, KNN filtering)
# =============================================================================

# Minimum query_num_visual_patches and neighbor_num_visual_patches to include a row.
RESULTS_ANALYSIS_MIN_PATCHES = 1
# Minimum query_num_glyphs and neighbor_num_glyphs to include a row.
RESULTS_ANALYSIS_MIN_GLYPHS = 1


# =============================================================================
# 16) OCR / ALTO feature extraction (stats)
# =============================================================================

OCR_GLYPH_CONFIDENCE_THRESHOLD = 0.92
OCR_STRING_CONFIDENCE_THRESHOLD = 0.96

# Filter characters by aspect ratio: remove characters with aspect ratio (max(width, height) / min(width, height)) > threshold.
# Applied after confidence filtering. Set to None or 0 to disable this filter.
OCR_CHAR_ASPECT_RATIO_THRESHOLD = 2.0

# If set to a string alphabet, fixes vector order for OCR stats (helpful for Hebrew).
OCR_ALPHABET = None  # type: ignore[var-annotated]

OCR_CASE_SENSITIVE = False
OCR_INCLUDE_GLYPH_SIZE_STATS = False
OCR_INCLUDE_NORMALIZED_FREQUENCY = True


# =============================================================================
# 17) Multi-modal XML (glyph + word extraction) and char encoder
# =============================================================================

# Default classification DB table names. Model initialization is controlled separately in main.py;
# stage only selects the default classification table preset and optional classification-data sampling.
# Geniza contrastive data uses GENIZA_CONTRASTIVE_TABLE.
# XML paths come from the DB (xml_path column).
PRETRAIN_TABLE_NAME = "pretrain_finetune_oriental_non_oriental_train_val_test_split"
FINETUNE_TABLE_NAME = PRETRAIN_TABLE_NAME
STAGE1_TABLE_NAME = PRETRAIN_TABLE_NAME
STAGE2_TABLE_NAME = FINETUNE_TABLE_NAME

# Optional fallback root for ALTO XML when a row has no xml_path (e.g. scripts that don't use DB).
# Training uses xml_path from the DB; this is only for find_xml_path_pretrain() fallback. Set via --xml_base_path if needed.
FALLBACK_XML_BASE_PATH = None

# Glyph/character patch extraction.
# Resolution used for each extracted glyph crop.
CHAR_PATCH_SIZE = _cfg(MODEL_CONFIG, "glyph.char_patch_size")
# Hebrew glyph alphabet: 27 entries (22 standard letters + 5 final forms).
HEBREW_ALPHABET = list("אבגדהוזחטיכלמנסעפצקרשתךםןףץ")
# Only these glyph classes are used as character-branch inputs.
GLYPH_INPUT_LETTERS = str(_cfg(MODEL_CONFIG, "glyph.input_letters"))
GLYPH_INPUT_ALPHABET = [ch for ch in GLYPH_INPUT_LETTERS if ch in HEBREW_ALPHABET]
_ensure(
    len(GLYPH_INPUT_ALPHABET) == len(GLYPH_INPUT_LETTERS),
    f"All glyph input letters must exist in HEBREW_ALPHABET, got {GLYPH_INPUT_LETTERS!r}",
)
# Number of glyphs retained for each selected input letter.
GLYPHS_PER_INPUT_CLASS = int(_cfg(MODEL_CONFIG, "glyph.glyphs_per_class"))
# Always cap the character branch to at most this many selected glyphs per image.
GLYPH_MAX_INPUTS_PER_IMAGE = len(GLYPH_INPUT_ALPHABET) * GLYPHS_PER_INPUT_CLASS
# Max glyphs per image (caps collate + GlyphVisualEncoder memory). Lower = less OOM risk.
# With DataParallel, per-GPU batch has fewer images but each can have up to this many glyphs.
MAX_CHARS_PER_IMAGE = min(int(_cfg(MODEL_CONFIG, "glyph.max_chars_per_image")), GLYPH_MAX_INPUTS_PER_IMAGE)
# Optional stricter glyph selection:
# keep only non-edge glyphs inside a word to reduce fragment/background leakage.
GLYPH_ONLY_MIDDLE_LETTERS = _cfg(MODEL_CONFIG, "glyph.only_middle_letters")
# Minimum word length required before middle-letter filtering is applied.
GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE = _cfg(MODEL_CONFIG, "glyph.min_word_length_for_middle")
# Tighten each OCR glyph bbox to the actual ink before resizing. Helps when
# the OCR bbox is loose or partly extends past the parchment edge into
# library backing. Vectorized and cheap (~1 ms per page of glyphs).
GLYPH_TIGHTEN_BBOX_TO_INK = _cfg(MODEL_CONFIG, "glyph.tighten_bbox_to_ink")
# A pixel is treated as ink if its grayscale brightness is darker than this
# percentile of the patch AND its RGB peak-to-peak (color spread) is small
# enough to look neutral. The neutrality check excludes saturated library
# board pixels (e.g. deep blue) that happen to be dark.
GLYPH_TIGHTEN_INK_PERCENTILE = 22.0
# Cap so we never declare bright parchment as ink on patches with no actual
# ink pixels (rare but possible if OCR placed a bbox on bare parchment).
GLYPH_TIGHTEN_INK_MAX_THRESHOLD = 160
# Maximum RGB peak-to-peak for a pixel to count as ink. Manuscript ink is
# carbon black or warm dark brown (spread <= ~50). Library blue boards have
# spread >= ~100 and are correctly excluded.
GLYPH_TIGHTEN_MAX_COLOR_SPREAD = 70
# Minimum ink-pixel count per row/column for it to be kept by the trimmer.
# Filters out a few stray dark specks at the edges.
GLYPH_TIGHTEN_MIN_PIXELS_PER_LINE = 3
# Padding (as fraction of the smaller patch side) added back around the
# tight ink bbox so we don't clip stroke endings.
GLYPH_TIGHTEN_MARGIN_RATIO = 0.10
# Bail out and keep the original bbox if tightening would shrink either
# axis below this fraction of its current length. Defensive against very
# faded ink where the brightness threshold can underestimate the ink area.
GLYPH_TIGHTEN_MIN_SIZE_RATIO = 0.10
# Minimum fraction of patch pixels classified as ink (same darkness + neutral
# color rule as tightening) before the crop is kept. Drops near-empty OCR
# boxes on bare parchment. Set to 0.0 to disable (glyph.min_ink_fraction).
GLYPH_MIN_INK_FRACTION = float(_cfg(MODEL_CONFIG, "glyph.min_ink_fraction"))
_ensure(
    0.0 <= GLYPH_MIN_INK_FRACTION <= 1.0,
    f"glyph.min_ink_fraction must be in [0, 1], got {GLYPH_MIN_INK_FRACTION!r}",
)
NUM_GLYPH_CLASSES = len(GLYPH_INPUT_ALPHABET) + 1  # +1 for unknown/other
_ensure(
    int(MAX_CHARS_PER_IMAGE) == int(GLYPH_MAX_INPUTS_PER_IMAGE),
    "MAX_CHARS_PER_IMAGE must equal GLYPH_MAX_INPUTS_PER_IMAGE for the fixed glyph-slot layout "
    f"({GLYPH_MAX_INPUTS_PER_IMAGE}), got {MAX_CHARS_PER_IMAGE!r}",
)
try:
    GLYPHS_PER_CLASS = _cfg(MODEL_CONFIG, "glyph.glyphs_per_class")
except KeyError:
    GLYPHS_PER_CLASS = GLYPHS_PER_INPUT_CLASS
_ensure(
    int(GLYPHS_PER_CLASS) == int(GLYPHS_PER_INPUT_CLASS),
    f"glyph.glyphs_per_class must be {GLYPHS_PER_INPUT_CLASS} for the fixed glyph-slot layout, got {GLYPHS_PER_CLASS!r}",
)
 
# Character patch transforms (applied to EACH glyph patch; keep lightweight).
# NOTE: Applying heavy augmentations like RandAugment to 150 glyphs/page can dominate runtime.
CHAR_PATCH_APPLY_RANDAUGMENT = _cfg(MODEL_CONFIG, "glyph.char_patch_apply_randaugment") 
# If True, apply ImageNet normalization to glyph patches (recommended for CNN/ViT encoders).
# For VAE training/encoder, we typically keep patches in [0,1] (handled in dataset by encoder type).
CHAR_PATCH_APPLY_IMAGENET_NORM = _cfg(MODEL_CONFIG, "glyph.char_patch_apply_imagenet_norm")

# Color augmentation (applied to tiles AND glyph patches during training).
# Targets parchment tone variation, lighting, and digitization differences across collections.
# Enable color jitter augmentation during training.
AUGMENT_COLOR_JITTER = _cfg(MODEL_CONFIG, "augmentation.color_jitter")
try:
    AUGMENT_APPLY_PROB = _cfg(MODEL_CONFIG, "augmentation.apply_prob")
except KeyError:
    AUGMENT_APPLY_PROB = 1.0
try:
    AUGMENT_COLOR_JITTER_PROB = _cfg(MODEL_CONFIG, "augmentation.color_jitter_prob")
except KeyError:
    AUGMENT_COLOR_JITTER_PROB = 1.0 if AUGMENT_COLOR_JITTER else 0.0
# Brightness jitter amplitude.
AUGMENT_COLOR_JITTER_BRIGHTNESS = _cfg(MODEL_CONFIG, "augmentation.color_jitter_brightness")
# Contrast jitter amplitude.
AUGMENT_COLOR_JITTER_CONTRAST = _cfg(MODEL_CONFIG, "augmentation.color_jitter_contrast")
# Saturation jitter amplitude.
AUGMENT_COLOR_JITTER_SATURATION = _cfg(MODEL_CONFIG, "augmentation.color_jitter_saturation")
# Hue jitter amplitude.
AUGMENT_COLOR_JITTER_HUE = _cfg(MODEL_CONFIG, "augmentation.color_jitter_hue")
# Probability of converting a sample to grayscale.
AUGMENT_RANDOM_GRAYSCALE_PROB = _cfg(MODEL_CONFIG, "augmentation.random_grayscale_prob")
# Probability of applying Gaussian blur.
AUGMENT_GAUSSIAN_BLUR_PROB = _cfg(MODEL_CONFIG, "augmentation.gaussian_blur_prob")
# Kernel size used for Gaussian blur augmentation.
AUGMENT_GAUSSIAN_BLUR_KERNEL = _cfg(MODEL_CONFIG, "augmentation.gaussian_blur_kernel")
# Probability of overlaying a synthetic library/background cue pattern.
AUGMENT_BACKGROUND_PATTERN_PROB = _cfg(MODEL_CONFIG, "augmentation.background_pattern_prob")
# Pattern families are split into two semantic groups:
# - surface patterns (paper/parchment texture cues)
# - library support patterns (mount/board/grid cues)
# Keep backward compatibility with the legacy single list key.
try:
    AUGMENT_BACKGROUND_SURFACE_PATTERN_TYPES = tuple(_cfg(MODEL_CONFIG, "augmentation.background_surface_pattern_types"))
except KeyError:
    AUGMENT_BACKGROUND_SURFACE_PATTERN_TYPES = ()
try:
    AUGMENT_BACKGROUND_LIBRARY_PATTERN_TYPES = tuple(_cfg(MODEL_CONFIG, "augmentation.background_library_pattern_types"))
except KeyError:
    AUGMENT_BACKGROUND_LIBRARY_PATTERN_TYPES = ()
if AUGMENT_BACKGROUND_SURFACE_PATTERN_TYPES or AUGMENT_BACKGROUND_LIBRARY_PATTERN_TYPES:
    AUGMENT_BACKGROUND_PATTERN_TYPES = tuple(AUGMENT_BACKGROUND_SURFACE_PATTERN_TYPES) + tuple(AUGMENT_BACKGROUND_LIBRARY_PATTERN_TYPES)
else:
    # Legacy config path
    try:
        AUGMENT_BACKGROUND_PATTERN_TYPES = tuple(_cfg(MODEL_CONFIG, "augmentation.background_pattern_types"))
    except KeyError:
        AUGMENT_BACKGROUND_PATTERN_TYPES = ("random_library_background",)
try:
    AUGMENT_BACKGROUND_SURFACE_PATTERN_PROB = _cfg(MODEL_CONFIG, "augmentation.background_surface_pattern_prob")
except KeyError:
    # Backward compatible default: if categories are configured, split legacy prob.
    if AUGMENT_BACKGROUND_SURFACE_PATTERN_TYPES or AUGMENT_BACKGROUND_LIBRARY_PATTERN_TYPES:
        AUGMENT_BACKGROUND_SURFACE_PATTERN_PROB = float(AUGMENT_BACKGROUND_PATTERN_PROB) * 0.4
    else:
        AUGMENT_BACKGROUND_SURFACE_PATTERN_PROB = float(AUGMENT_BACKGROUND_PATTERN_PROB)
try:
    AUGMENT_BACKGROUND_LIBRARY_PATTERN_PROB = _cfg(MODEL_CONFIG, "augmentation.background_library_pattern_prob")
except KeyError:
    if AUGMENT_BACKGROUND_SURFACE_PATTERN_TYPES or AUGMENT_BACKGROUND_LIBRARY_PATTERN_TYPES:
        AUGMENT_BACKGROUND_LIBRARY_PATTERN_PROB = float(AUGMENT_BACKGROUND_PATTERN_PROB) * 0.6
    else:
        AUGMENT_BACKGROUND_LIBRARY_PATTERN_PROB = 0.0
try:
    AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB = _cfg(MODEL_CONFIG, "augmentation.background_random_library_prob")
except KeyError:
    AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB = float(AUGMENT_BACKGROUND_PATTERN_PROB)
try:
    AUGMENT_PARCHMENT_STAINS_PROB = _cfg(MODEL_CONFIG, "augmentation.parchment_stains_prob")
except KeyError:
    AUGMENT_PARCHMENT_STAINS_PROB = 0.0
try:
    AUGMENT_WHITE_BACKGROUND_PROB = _cfg(MODEL_CONFIG, "augmentation.white_background_prob")
except KeyError:
    AUGMENT_WHITE_BACKGROUND_PROB = 0.0
try:
    AUGMENT_BACKGROUND_SAMPLING_ALPHA = _cfg(MODEL_CONFIG, "augmentation.background_sampling_alpha")
except KeyError:
    AUGMENT_BACKGROUND_SAMPLING_ALPHA = 0.72
try:
    AUGMENT_BACKGROUND_SAMPLING_FLOOR = _cfg(MODEL_CONFIG, "augmentation.background_sampling_floor")
except KeyError:
    AUGMENT_BACKGROUND_SAMPLING_FLOOR = 80.0
# Blend-strength range for the synthetic background overlay.
AUGMENT_BACKGROUND_PATTERN_ALPHA_RANGE = tuple(_cfg(MODEL_CONFIG, "augmentation.background_pattern_alpha_range"))
# Grid-cell size range for the blue-grid overlay pattern.
AUGMENT_BACKGROUND_PATTERN_GRID_SPACING_RANGE = tuple(_cfg(MODEL_CONFIG, "augmentation.background_pattern_grid_spacing_range"))
# Line width for the blue-grid overlay pattern.
AUGMENT_BACKGROUND_PATTERN_GRID_LINE_WIDTH = _cfg(MODEL_CONFIG, "augmentation.background_pattern_grid_line_width")
# Probability of random erasing.
AUGMENT_RANDOM_ERASING_PROB = _cfg(MODEL_CONFIG, "augmentation.random_erasing_prob")
# Random erasing area range as a fraction of image area.
AUGMENT_RANDOM_ERASING_SCALE = tuple(_cfg(MODEL_CONFIG, "augmentation.random_erasing_scale"))
# Random erasing aspect-ratio range.
AUGMENT_RANDOM_ERASING_RATIO = tuple(_cfg(MODEL_CONFIG, "augmentation.random_erasing_ratio"))
try:
    AUGMENT_LOCAL_TEXTURE_PROB = _cfg(MODEL_CONFIG, "augmentation.local_texture_prob")
except KeyError:
    AUGMENT_LOCAL_TEXTURE_PROB = 0.0
try:
    AUGMENT_LOCAL_TEXTURE_ALPHA_RANGE = tuple(_cfg(MODEL_CONFIG, "augmentation.local_texture_alpha_range"))
except KeyError:
    AUGMENT_LOCAL_TEXTURE_ALPHA_RANGE = (0.06, 0.20)
try:
    AUGMENT_TONE_CONTRAST_PROB = _cfg(MODEL_CONFIG, "augmentation.tone_contrast_prob")
except KeyError:
    AUGMENT_TONE_CONTRAST_PROB = 0.0
try:
    AUGMENT_INK_DEGRADATION_PROB = _cfg(MODEL_CONFIG, "augmentation.ink_degradation_prob")
except KeyError:
    AUGMENT_INK_DEGRADATION_PROB = 0.0
try:
    AUGMENT_SPECKLE_MORPH_PROB = _cfg(MODEL_CONFIG, "augmentation.speckle_morph_prob")
except KeyError:
    AUGMENT_SPECKLE_MORPH_PROB = 0.0
try:
    AUGMENT_BORDER_CROP_PROB = _cfg(MODEL_CONFIG, "augmentation.border_crop_prob")
except KeyError:
    AUGMENT_BORDER_CROP_PROB = 0.0
try:
    AUGMENT_BORDER_MAX_FRAC = _cfg(MODEL_CONFIG, "augmentation.border_max_frac")
except KeyError:
    AUGMENT_BORDER_MAX_FRAC = 0.13
try:
    AUGMENT_EDGE_CROP_MAX_FRAC = _cfg(MODEL_CONFIG, "augmentation.edge_crop_max_frac")
except KeyError:
    AUGMENT_EDGE_CROP_MAX_FRAC = 0.18
try:
    AUGMENT_GAMMA_PROB = _cfg(MODEL_CONFIG, "augmentation.gamma_prob")
except KeyError:
    AUGMENT_GAMMA_PROB = 0.3
try:
    AUGMENT_RESOLUTION_JITTER_PROB = _cfg(MODEL_CONFIG, "augmentation.resolution_jitter_prob")
except KeyError:
    AUGMENT_RESOLUTION_JITTER_PROB = 0.2
try:
    AUGMENT_ZOOM_PROB = _cfg(MODEL_CONFIG, "augmentation.zoom_prob")
except KeyError:
    AUGMENT_ZOOM_PROB = float(AUGMENT_RESOLUTION_JITTER_PROB)
try:
    AUGMENT_ZOOM_SCALE_RANGE = tuple(_cfg(MODEL_CONFIG, "augmentation.zoom_scale_range"))
except KeyError:
    AUGMENT_ZOOM_SCALE_RANGE = (0.82, 1.25)
try:
    AUGMENT_TILT_PROB = _cfg(MODEL_CONFIG, "augmentation.tilt_prob")
except KeyError:
    AUGMENT_TILT_PROB = 0.0
try:
    AUGMENT_TILT_DEGREES = _cfg(MODEL_CONFIG, "augmentation.tilt_degrees")
except KeyError:
    AUGMENT_TILT_DEGREES = 10.0
try:
    AUGMENT_RANDAUGMENT_PROB = _cfg(MODEL_CONFIG, "augmentation.randaugment_prob")
except KeyError:
    AUGMENT_RANDAUGMENT_PROB = 1.0

# Glyph-specific overrides (glyphs are 128x128, much smaller than 560x560 tiles,
# so aggressive augmentations need softer parameters to avoid destroying character shapes).
try:
    GLYPH_GAUSSIAN_BLUR_KERNEL = int(_cfg(MODEL_CONFIG, "augmentation.glyph_gaussian_blur_kernel"))
except KeyError:
    GLYPH_GAUSSIAN_BLUR_KERNEL = int(AUGMENT_GAUSSIAN_BLUR_KERNEL)
try:
    GLYPH_RANDOM_ERASING_SCALE = tuple(_cfg(MODEL_CONFIG, "augmentation.glyph_random_erasing_scale"))
except KeyError:
    GLYPH_RANDOM_ERASING_SCALE = AUGMENT_RANDOM_ERASING_SCALE

# Word extraction limits.
# Maximum words kept per image.
MAX_WORDS_PER_IMAGE = _cfg(MODEL_CONFIG, "word.max_words_per_image")
# If True, require words to pass the Hebrew dictionary check.
USE_HEBREW_DICT_CHECK = _cfg(MODEL_CONFIG, "word.use_hebrew_dict_check")
# Optional deterministic TF-IDF scaling before word-to-line attention.
WORD_BRANCH_USE_TFIDF_GATING = bool(_cfg(MODEL_CONFIG, "word.use_tfidf_gating"))
_word_tfidf_dictionary_path = Path(_cfg(MODEL_CONFIG, "word.tfidf_dictionary_path"))
if not _word_tfidf_dictionary_path.is_absolute():
    _word_tfidf_dictionary_path = _SYSTEM_DIR / _word_tfidf_dictionary_path
WORD_BRANCH_TFIDF_DICTIONARY_PATH = _word_tfidf_dictionary_path.resolve()

# Modality flags (toggle inputs to the fusion transformer).
# Enable the visual/tile branch.
USE_VISUAL_MOD = _cfg(MODEL_CONFIG, "modalities.use_visual")
# Enable the glyph branch.
USE_CHAR_MOD = _cfg(MODEL_CONFIG, "modalities.use_char")
# Enable the word branch.
USE_WORD_MOD = _cfg(MODEL_CONFIG, "modalities.use_word")


# =============================================================================
# 18) New multi-modal branch architecture constants
# =============================================================================

# Tile branch settings.
TILE_BRANCH_ENABLE_POS_ENCODING = _cfg(MODEL_CONFIG, "tile.enable_pos_encoding")  # Tile position on page is irrelevant for manuscript style
# If True, use Fourier features for tile positions.
TILE_BRANCH_USE_FOURIER_POS = _cfg(MODEL_CONFIG, "tile.use_fourier_pos")
# Number of Fourier frequencies for tile position encoding.
TILE_BRANCH_NUM_FREQS = _cfg(MODEL_CONFIG, "tile.num_freqs")

# Glyph branch settings.
GLYPH_ENCODER_TYPE = _cfg(MODEL_CONFIG, "glyph.encoder_type")  # "convnext_tiny" | "swin_tiny"
try:
    # Learned queries are set aggregators, not one-to-one glyph slots, so their
    # count is independent of the maximum number of input glyphs.
    GLYPH_NUM_SUMMARY_TOKENS = int(_cfg(MODEL_CONFIG, "glyph.num_summary_tokens"))
except KeyError:
    GLYPH_NUM_SUMMARY_TOKENS = MAX_CHARS_PER_IMAGE
# A count of zero disables the optional glyph summarizer.
try:
    GLYPH_MIN_SUMMARY_TOKENS = _cfg(MODEL_CONFIG, "glyph.min_summary_tokens")
except KeyError:
    GLYPH_MIN_SUMMARY_TOKENS = 4
# Glyph set summarizer (cross-attention over glyph tokens).
try:
    GLYPH_SUMMARIZER_CROSS_ATTN_LAYERS = _cfg(MODEL_CONFIG, "glyph.summarizer_cross_attn_layers")
except KeyError:
    GLYPH_SUMMARIZER_CROSS_ATTN_LAYERS = 1
try:
    GLYPH_SUMMARIZER_ATTENTION_HEADS = _cfg(MODEL_CONFIG, "glyph.summarizer_attention_heads")
except KeyError:
    GLYPH_SUMMARIZER_ATTENTION_HEADS = 8
# Minimum allowed glyph area before hard filtering.
GLYPH_QUALITY_MIN_AREA = _cfg(MODEL_CONFIG, "glyph.quality_min_area")
# Minimum allowed glyph width before hard filtering.
GLYPH_QUALITY_MIN_WIDTH = _cfg(MODEL_CONFIG, "glyph.quality_min_width")
# Minimum allowed glyph height before hard filtering.
GLYPH_QUALITY_MIN_HEIGHT = _cfg(MODEL_CONFIG, "glyph.quality_min_height")
# Maximum allowed glyph aspect ratio before hard filtering.
GLYPH_QUALITY_MAX_ASPECT_RATIO = _cfg(MODEL_CONFIG, "glyph.quality_max_aspect_ratio")
try:
    GLYPH_AUG_COLOR_JITTER_PROB = _cfg(MODEL_CONFIG, "glyph.aug_color_jitter_prob")
except KeyError:
    GLYPH_AUG_COLOR_JITTER_PROB = AUGMENT_COLOR_JITTER_PROB
try:
    GLYPH_AUG_APPLY_PROB = _cfg(MODEL_CONFIG, "glyph.aug_apply_prob")
except KeyError:
    GLYPH_AUG_APPLY_PROB = AUGMENT_APPLY_PROB
try:
    GLYPH_AUG_RANDOM_GRAYSCALE_PROB = _cfg(MODEL_CONFIG, "glyph.aug_random_grayscale_prob")
except KeyError:
    GLYPH_AUG_RANDOM_GRAYSCALE_PROB = AUGMENT_RANDOM_GRAYSCALE_PROB
try:
    GLYPH_AUG_GAUSSIAN_BLUR_PROB = _cfg(MODEL_CONFIG, "glyph.aug_gaussian_blur_prob")
except KeyError:
    GLYPH_AUG_GAUSSIAN_BLUR_PROB = AUGMENT_GAUSSIAN_BLUR_PROB
try:
    GLYPH_AUG_BACKGROUND_RANDOM_LIBRARY_PROB = _cfg(MODEL_CONFIG, "glyph.aug_background_random_library_prob")
except KeyError:
    GLYPH_AUG_BACKGROUND_RANDOM_LIBRARY_PROB = AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB
try:
    GLYPH_AUG_PARCHMENT_STAINS_PROB = _cfg(MODEL_CONFIG, "glyph.aug_parchment_stains_prob")
except KeyError:
    GLYPH_AUG_PARCHMENT_STAINS_PROB = AUGMENT_PARCHMENT_STAINS_PROB
try:
    GLYPH_AUG_WHITE_BACKGROUND_PROB = _cfg(MODEL_CONFIG, "glyph.aug_white_background_prob")
except KeyError:
    GLYPH_AUG_WHITE_BACKGROUND_PROB = AUGMENT_WHITE_BACKGROUND_PROB
try:
    GLYPH_AUG_LOCAL_TEXTURE_PROB = _cfg(MODEL_CONFIG, "glyph.aug_local_texture_prob")
except KeyError:
    GLYPH_AUG_LOCAL_TEXTURE_PROB = AUGMENT_LOCAL_TEXTURE_PROB
try:
    GLYPH_AUG_TONE_CONTRAST_PROB = _cfg(MODEL_CONFIG, "glyph.aug_tone_contrast_prob")
except KeyError:
    GLYPH_AUG_TONE_CONTRAST_PROB = AUGMENT_TONE_CONTRAST_PROB
try:
    GLYPH_AUG_TILT_PROB = _cfg(MODEL_CONFIG, "glyph.aug_tilt_prob")
except KeyError:
    GLYPH_AUG_TILT_PROB = AUGMENT_TILT_PROB
try:
    GLYPH_AUG_BORDER_CROP_PROB = _cfg(MODEL_CONFIG, "glyph.aug_border_crop_prob")
except KeyError:
    GLYPH_AUG_BORDER_CROP_PROB = AUGMENT_BORDER_CROP_PROB
try:
    GLYPH_AUG_ZOOM_PROB = _cfg(MODEL_CONFIG, "glyph.aug_zoom_prob")
except KeyError:
    GLYPH_AUG_ZOOM_PROB = AUGMENT_ZOOM_PROB
try:
    GLYPH_AUG_RANDAUGMENT_PROB = _cfg(MODEL_CONFIG, "glyph.aug_randaugment_prob")
except KeyError:
    GLYPH_AUG_RANDAUGMENT_PROB = AUGMENT_RANDAUGMENT_PROB
GLYPH_BRANCH_ENABLE_POS_ENCODING = _cfg(MODEL_CONFIG, "glyph.enable_pos_encoding")  # Character shapes are position-invariant for style ID
# If True, use Fourier features for glyph positions.
GLYPH_BRANCH_USE_FOURIER_POS = _cfg(MODEL_CONFIG, "glyph.use_fourier_pos")
# Number of Fourier frequencies for glyph position encoding.
GLYPH_BRANCH_NUM_FREQS = _cfg(MODEL_CONFIG, "glyph.num_freqs")

# Word branch settings.
WORD_BRANCH_FREEZE_ALEPHBERT = _cfg(MODEL_CONFIG, "word.freeze_alephbert")   # True = freeze AlephBERT weights (faster, less VRAM); False = fine-tune
# Maximum tokenized line length sent into AlephBERT.
WORD_BRANCH_LINE_MAX_LENGTH = _cfg(MODEL_CONFIG, "word.line_max_length")
WORD_BRANCH_WORD_POOLING = _cfg(MODEL_CONFIG, "word.word_pooling")  # "mean" | "first" - how to pool subwords per word
WORD_BRANCH_ATTENTION_HEADS = _cfg(MODEL_CONFIG, "word.attention_heads")  # For attention pooling (words → line)
# If True, add line-level positional encodings to word tokens.
WORD_BRANCH_ENABLE_POS_ENCODING = _cfg(MODEL_CONFIG, "word.enable_pos_encoding")
# If True, use Fourier features for word coordinates.
WORD_BRANCH_USE_FOURIER_POS = _cfg(MODEL_CONFIG, "word.use_fourier_pos")
# Number of Fourier frequencies for word position encoding.
WORD_BRANCH_NUM_FREQS = _cfg(MODEL_CONFIG, "word.num_freqs")
WORD_BRANCH_MIN_AREA = _cfg(MODEL_CONFIG, "word.min_area")  # Minimum bbox area for word filtering
WORD_NUM_SUMMARY_TOKENS = int(_cfg(MODEL_CONFIG, "word.num_summary_tokens"))
# A count of zero disables the optional word/line summarizer.
try:
    WORD_MIN_SUMMARY_TOKENS = _cfg(MODEL_CONFIG, "word.min_summary_tokens")
except KeyError:
    WORD_MIN_SUMMARY_TOKENS = 2

# Perceiver-style fusion settings.
PERCEIVER_NUM_LATENTS = _cfg(MODEL_CONFIG, "perceiver.num_latents")  # Number of learnable latent queries
PERCEIVER_NUM_CROSS_ATTN_LAYERS = _cfg(MODEL_CONFIG, "perceiver.num_cross_attn_layers")  # Number of cross-attention layers
PERCEIVER_POOLING = _cfg(MODEL_CONFIG, "perceiver.pooling")  # "mean" | "cls" - how to pool latents for classification

# VLAD-style fusion settings.
VLAD_NUM_CLUSTERS = _cfg(MODEL_CONFIG, "vlad.num_clusters")
VLAD_ASSIGNMENT_TEMPERATURE = _cfg(MODEL_CONFIG, "vlad.assignment_temperature")
VLAD_NORMALIZE_INPUT = _cfg(MODEL_CONFIG, "vlad.normalize_input")

# Modality dropout (training only, for regularization).
# Probability of dropping each modality during training (0.0 = never drop, 1.0 = always drop).
# Drop tiles more often than glyphs so the fusion/head must learn a viable glyph path.
MODALITY_DROPOUT_ENABLED = _cfg(MODEL_CONFIG, "modality_dropout.enabled")  # Enable/disable modality dropout
MODALITY_DROPOUT_PROB_VISUAL = _cfg(MODEL_CONFIG, "modality_dropout.prob_visual")
MODALITY_DROPOUT_PROB_CHAR = _cfg(MODEL_CONFIG, "modality_dropout.prob_char")
MODALITY_DROPOUT_PROB_WORD = _cfg(MODEL_CONFIG, "modality_dropout.prob_word")  # Probability of dropping word modality

# Token subsampling (training only). Randomly keeps a fraction of valid tokens
# per modality per sample. Teaches the model to produce stable representations
# regardless of how many tiles/glyphs/words are available (full page vs fragment).
# Applied independently per sample in the batch, AFTER modality dropout.
TOKEN_SUBSAMPLE_ENABLED = _cfg(MODEL_CONFIG, "token_subsample.enabled")
TOKEN_SUBSAMPLE_PROB = _cfg(MODEL_CONFIG, "token_subsample.prob")          # Probability of applying subsampling to a given sample
TOKEN_SUBSAMPLE_MIN_KEEP_TILES = _cfg(MODEL_CONFIG, "token_subsample.min_keep_tiles")  # Always keep at least this many tiles so tile branch sees richer inputs
TOKEN_SUBSAMPLE_MIN_KEEP_GLYPHS = _cfg(MODEL_CONFIG, "token_subsample.min_keep_glyphs") # Always keep at least this many glyph patches
TOKEN_SUBSAMPLE_MIN_KEEP_WORDS = _cfg(MODEL_CONFIG, "token_subsample.min_keep_words")  # Always keep at least this many word tokens
TOKEN_SUBSAMPLE_FRAC_RANGE = tuple(_cfg(MODEL_CONFIG, "token_subsample.frac_range"))  # Uniform range for keep-fraction


# =============================================================================
# 19) Config validation
# =============================================================================

def validate_runtime_config() -> None:
    # Basic architecture choices
    _ensure(TILE_ENCODER_TYPE in {"dinov2", "convnext"}, f"TILE_ENCODER_TYPE must be 'dinov2' or 'convnext', got {TILE_ENCODER_TYPE!r}")
    _ensure(FUSION_METHOD in {"transformer", "perceiver", "vlad", "symmetric"}, f"Unsupported FUSION_METHOD: {FUSION_METHOD!r}")
    if FUSION_METHOD == "symmetric":
        enabled_count = int(USE_VISUAL_MOD) + int(USE_CHAR_MOD) + int(USE_WORD_MOD)
        _ensure(enabled_count > 0, "symmetric fusion requires at least one enabled modality")
        branch_output_dim = (
            int(SYMMETRIC_BRANCH_DIM)
            if USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
            else int(D_MODEL)
        )
        expected_dim = branch_output_dim * enabled_count
        _ensure(int(LATENT_DIM) == expected_dim,
                f"symmetric fusion output must be {expected_dim}, got {LATENT_DIM}")
    _ensure(GLYPH_ENCODER_TYPE in {"convnext_tiny", "swin_tiny"}, f"GLYPH_ENCODER_TYPE must be 'convnext_tiny' or 'swin_tiny', got {GLYPH_ENCODER_TYPE!r}")
    _ensure(WORD_BRANCH_WORD_POOLING in {"mean", "first"}, f"WORD_BRANCH_WORD_POOLING must be 'mean' or 'first', got {WORD_BRANCH_WORD_POOLING!r}")
    if WORD_BRANCH_USE_TFIDF_GATING:
        _ensure(
            WORD_BRANCH_TFIDF_DICTIONARY_PATH.is_file(),
            "word.use_tfidf_gating requires a valid word.tfidf_dictionary_path; "
            f"not found: {WORD_BRANCH_TFIDF_DICTIONARY_PATH}",
        )
    _ensure(PERCEIVER_POOLING in {"mean", "cls"}, f"PERCEIVER_POOLING must be 'mean' or 'cls', got {PERCEIVER_POOLING!r}")
    _ensure(LABEL_HEAD in {"manuscript_id", "decade"}, f"LABEL_HEAD must be 'manuscript_id' or 'decade', got {LABEL_HEAD!r}")

    # Positive sizes / counts
    for name, value in [
        ("TILE_SIZE", TILE_SIZE),
        ("TILE_STRIDE", TILE_STRIDE),
        ("MAX_TILES_TRAIN", MAX_TILES_TRAIN),
        ("MAX_TILES_EVAL", MAX_TILES_EVAL),
        ("D_MODEL", D_MODEL),
        ("SYMMETRIC_BRANCH_DIM", SYMMETRIC_BRANCH_DIM),
        ("LATENT_DIM", LATENT_DIM),
        ("NUM_LAYERS", NUM_LAYERS),
        ("NUM_HEADS", NUM_HEADS),
        ("TRANSFORMER_NUM_LAYERS", TRANSFORMER_NUM_LAYERS),
        ("TRANSFORMER_NUM_HEADS", TRANSFORMER_NUM_HEADS),
        ("TRANSFORMER_DIM_FEEDFORWARD", TRANSFORMER_DIM_FEEDFORWARD),
        ("TILE_BRANCH_TRANSFORMER_LAYERS", TILE_BRANCH_TRANSFORMER_LAYERS),
        ("TILE_BRANCH_TRANSFORMER_HEADS", TILE_BRANCH_TRANSFORMER_HEADS),
        ("TILE_BRANCH_TRANSFORMER_DIM_FEEDFORWARD", TILE_BRANCH_TRANSFORMER_DIM_FEEDFORWARD),
        ("TILE_SUMMARIZER_CROSS_ATTN_LAYERS", TILE_SUMMARIZER_CROSS_ATTN_LAYERS),
        ("TILE_SUMMARIZER_ATTENTION_HEADS", TILE_SUMMARIZER_ATTENTION_HEADS),
        ("CHAR_PATCH_SIZE", CHAR_PATCH_SIZE),
        ("MAX_CHARS_PER_IMAGE", MAX_CHARS_PER_IMAGE),
        ("MAX_WORDS_PER_IMAGE", MAX_WORDS_PER_IMAGE),
        ("GLYPH_SUMMARIZER_CROSS_ATTN_LAYERS", GLYPH_SUMMARIZER_CROSS_ATTN_LAYERS),
        ("GLYPH_SUMMARIZER_ATTENTION_HEADS", GLYPH_SUMMARIZER_ATTENTION_HEADS),
        ("WORD_BRANCH_LINE_MAX_LENGTH", WORD_BRANCH_LINE_MAX_LENGTH),
        ("WORD_BRANCH_ATTENTION_HEADS", WORD_BRANCH_ATTENTION_HEADS),
        ("PERCEIVER_NUM_LATENTS", PERCEIVER_NUM_LATENTS),
        ("PERCEIVER_NUM_CROSS_ATTN_LAYERS", PERCEIVER_NUM_CROSS_ATTN_LAYERS),
        ("VLAD_NUM_CLUSTERS", VLAD_NUM_CLUSTERS),
        ("PER_GPU_BATCH_SIZE", PER_GPU_BATCH_SIZE),
        ("GRADIENT_ACCUMULATION_STEPS", GRADIENT_ACCUMULATION_STEPS),
        ("NUM_EPOCHS", NUM_EPOCHS),
        ("NUM_WORKERS", NUM_WORKERS),
        ("MAIN_DATAPARALLEL_NUM_WORKERS", MAIN_DATAPARALLEL_NUM_WORKERS),
        ("TILE_ENCODE_CHUNK_SIZE", TILE_ENCODE_CHUNK_SIZE),
        ("CHAR_ENCODE_CHUNK_SIZE", CHAR_ENCODE_CHUNK_SIZE),
        ("MAX_SELECTED_MANUSCRIPTS", MAX_SELECTED_MANUSCRIPTS),
    ]:
        _ensure_positive_int(name, value)

    # Head divisibility assumptions from attention modules
    for name, heads in [
        ("NUM_HEADS", NUM_HEADS),
        ("TRANSFORMER_NUM_HEADS", TRANSFORMER_NUM_HEADS),
        ("TILE_BRANCH_TRANSFORMER_HEADS", TILE_BRANCH_TRANSFORMER_HEADS),
        ("TILE_SUMMARIZER_ATTENTION_HEADS", TILE_SUMMARIZER_ATTENTION_HEADS),
        ("GLYPH_SUMMARIZER_ATTENTION_HEADS", GLYPH_SUMMARIZER_ATTENTION_HEADS),
        ("WORD_BRANCH_ATTENTION_HEADS", WORD_BRANCH_ATTENTION_HEADS),
    ]:
        _ensure(D_MODEL % int(heads) == 0, f"D_MODEL={D_MODEL} must be divisible by {name}={heads}")

    _ensure(int(TILE_NUM_SUMMARY_TOKENS) >= 0,
            f"TILE_NUM_SUMMARY_TOKENS must be >= 0, got {TILE_NUM_SUMMARY_TOKENS!r}")
    if int(TILE_NUM_SUMMARY_TOKENS) > 0:
        _ensure(1 <= int(TILE_MIN_SUMMARY_TOKENS) <= int(TILE_NUM_SUMMARY_TOKENS),
                "TILE_MIN_SUMMARY_TOKENS must be between 1 and TILE_NUM_SUMMARY_TOKENS")
    for prefix, count, minimum in (
        ("GLYPH", GLYPH_NUM_SUMMARY_TOKENS, GLYPH_MIN_SUMMARY_TOKENS),
        ("WORD", WORD_NUM_SUMMARY_TOKENS, WORD_MIN_SUMMARY_TOKENS),
    ):
        _ensure(int(count) >= 0, f"{prefix}_NUM_SUMMARY_TOKENS must be >= 0, got {count!r}")
        if int(count) > 0:
            _ensure(1 <= int(minimum) <= int(count),
                    f"{prefix}_MIN_SUMMARY_TOKENS must be between 1 and "
                    f"{prefix}_NUM_SUMMARY_TOKENS")

    # Optimizer / scheduler basics
    _ensure(len(BETAS) == 2, f"BETAS must have length 2, got {BETAS!r}")
    _ensure(0.0 <= float(BETAS[0]) < 1.0 and 0.0 <= float(BETAS[1]) < 1.0, f"BETAS values must be in [0,1), got {BETAS!r}")
    _ensure(float(LEARNING_RATE_STAGE1) > 0.0, f"LEARNING_RATE_STAGE1 must be > 0, got {LEARNING_RATE_STAGE1!r}")
    _ensure(float(LEARNING_RATE_STAGE2) > 0.0, f"LEARNING_RATE_STAGE2 must be > 0, got {LEARNING_RATE_STAGE2!r}")
    _ensure(float(LEARNING_RATE_ALEPHBERT) > 0.0,
            f"LEARNING_RATE_ALEPHBERT must be > 0, got {LEARNING_RATE_ALEPHBERT!r}")
    _ensure(float(LEARNING_RATE_NEW_WORD) > 0.0,
            f"LEARNING_RATE_NEW_WORD must be > 0, got {LEARNING_RATE_NEW_WORD!r}")
    _ensure(int(TRAINING_SEED) >= 0, f"TRAINING_SEED must be >= 0, got {TRAINING_SEED!r}")
    _ensure(float(WEIGHT_DECAY) >= 0.0, f"WEIGHT_DECAY must be >= 0, got {WEIGHT_DECAY!r}")
    _ensure(float(SCHEDULER_ETA_MIN) >= 0.0, f"SCHEDULER_ETA_MIN must be >= 0, got {SCHEDULER_ETA_MIN!r}")
    _ensure(float(GRADIENT_CLIP_NORM) >= 0.0, f"GRADIENT_CLIP_NORM must be >= 0, got {GRADIENT_CLIP_NORM!r}")
    _ensure(float(VLAD_ASSIGNMENT_TEMPERATURE) > 0.0, f"VLAD_ASSIGNMENT_TEMPERATURE must be > 0, got {VLAD_ASSIGNMENT_TEMPERATURE!r}")

    # Loss consistency
    _ensure(float(ARCFACE_WEIGHT) >= 0.0, f"ARCFACE_WEIGHT must be >= 0, got {ARCFACE_WEIGHT!r}")
    _ensure(float(CE_WEIGHT) >= 0.0, f"CE_WEIGHT must be >= 0, got {CE_WEIGHT!r}")
    _ensure(float(LATENT_SPARSITY_WEIGHT) >= 0.0, f"LATENT_SPARSITY_WEIGHT must be >= 0, got {LATENT_SPARSITY_WEIGHT!r}")
    _ensure(float(TILE_AUX_LOSS_WEIGHT) >= 0.0, f"TILE_AUX_LOSS_WEIGHT must be >= 0, got {TILE_AUX_LOSS_WEIGHT!r}")
    _ensure(float(GLYPH_AUX_LOSS_WEIGHT) >= 0.0, f"GLYPH_AUX_LOSS_WEIGHT must be >= 0, got {GLYPH_AUX_LOSS_WEIGHT!r}")
    _ensure(float(FUSION_AUX_LOSS_WEIGHT) >= 0.0, f"FUSION_AUX_LOSS_WEIGHT must be >= 0, got {FUSION_AUX_LOSS_WEIGHT!r}")
    _ensure(float(WORD_AUX_LOSS_WEIGHT) >= 0.0, f"WORD_AUX_LOSS_WEIGHT must be >= 0, got {WORD_AUX_LOSS_WEIGHT!r}")
    _ensure(int(WORD_AUX_FULL_WEIGHT_EPOCHS) >= 0,
            f"WORD_AUX_FULL_WEIGHT_EPOCHS must be >= 0, got {WORD_AUX_FULL_WEIGHT_EPOCHS!r}")
    _ensure(int(WORD_AUX_DECAY_END_EPOCH) > int(WORD_AUX_FULL_WEIGHT_EPOCHS),
            "WORD_AUX_DECAY_END_EPOCH must be greater than WORD_AUX_FULL_WEIGHT_EPOCHS")
    _ensure(
        float(ARCFACE_WEIGHT) > 0.0 or float(CE_WEIGHT) > 0.0,
        "At least one of ARCFACE_WEIGHT or CE_WEIGHT must be > 0",
    )
    if float(ARCFACE_WEIGHT) > 0.0:
        _ensure(float(ARCFACE_MARGIN) >= 0.0, f"ARCFACE_MARGIN must be >= 0, got {ARCFACE_MARGIN!r}")
        _ensure(float(ARCFACE_SCALE) > 0.0, f"ARCFACE_SCALE must be > 0, got {ARCFACE_SCALE!r}")
        _ensure(int(ARCFACE_MARGIN_WARMUP_EPOCHS) >= 0, f"ARCFACE_MARGIN_WARMUP_EPOCHS must be >= 0, got {ARCFACE_MARGIN_WARMUP_EPOCHS!r}")
    _ensure(float(GENIZA_CONTRASTIVE_WEIGHT) >= 0.0, f"GENIZA_CONTRASTIVE_WEIGHT must be >= 0, got {GENIZA_CONTRASTIVE_WEIGHT!r}")
    _ensure(float(GENIZA_CONTRASTIVE_TEMPERATURE) > 0.0, f"GENIZA_CONTRASTIVE_TEMPERATURE must be > 0, got {GENIZA_CONTRASTIVE_TEMPERATURE!r}")
    _ensure(int(GENIZA_QUEUE_SIZE) > 0, f"GENIZA_QUEUE_SIZE must be > 0, got {GENIZA_QUEUE_SIZE!r}")
    _ensure(int(GENIZA_PK_K) >= 2, f"GENIZA_PK_K must be >= 2, got {GENIZA_PK_K!r}")
    _ensure(0.0 < float(GENIZA_DEMO_FRACTION) <= 1.0, f"GENIZA_DEMO_FRACTION must be in (0, 1], got {GENIZA_DEMO_FRACTION!r}")
    _ensure(int(GENIZA_DEMO_MIN_TRAIN_MANUSCRIPTS) > 0, f"GENIZA_DEMO_MIN_TRAIN_MANUSCRIPTS must be > 0, got {GENIZA_DEMO_MIN_TRAIN_MANUSCRIPTS!r}")
    _ensure(int(GENIZA_DEMO_MIN_VAL_MANUSCRIPTS) > 0, f"GENIZA_DEMO_MIN_VAL_MANUSCRIPTS must be > 0, got {GENIZA_DEMO_MIN_VAL_MANUSCRIPTS!r}")
    _ensure(int(GENIZA_DEMO_MAX_IMAGES_PER_MANUSCRIPT) >= 2, f"GENIZA_DEMO_MAX_IMAGES_PER_MANUSCRIPT must be >= 2, got {GENIZA_DEMO_MAX_IMAGES_PER_MANUSCRIPT!r}")

    # Modality consistency
    _ensure(USE_VISUAL_MOD or USE_CHAR_MOD or USE_WORD_MOD, "At least one modality must be enabled")
    _ensure(not (float(TILE_AUX_LOSS_WEIGHT) > 0.0 and not USE_VISUAL_MOD), "TILE_AUX_LOSS_WEIGHT > 0 requires USE_VISUAL_MOD=True")
    _ensure(not (float(GLYPH_AUX_LOSS_WEIGHT) > 0.0 and not USE_CHAR_MOD), "GLYPH_AUX_LOSS_WEIGHT > 0 requires USE_CHAR_MOD=True")
    _ensure(not (float(WORD_AUX_LOSS_WEIGHT) > 0.0 and not USE_WORD_MOD), "WORD_AUX_LOSS_WEIGHT > 0 requires USE_WORD_MOD=True")
    for name, value in [
        ("TRANSFORMER_RESIDUAL_TILE_WEIGHT", TRANSFORMER_RESIDUAL_TILE_WEIGHT),
        ("TRANSFORMER_RESIDUAL_GLYPH_WEIGHT", TRANSFORMER_RESIDUAL_GLYPH_WEIGHT),
        ("TRANSFORMER_RESIDUAL_WORD_WEIGHT", TRANSFORMER_RESIDUAL_WORD_WEIGHT),
    ]:
        _ensure(float(value) >= 0.0, f"{name} must be >= 0, got {value!r}")
    _ensure(
        (
            float(TRANSFORMER_RESIDUAL_TILE_WEIGHT)
            + float(TRANSFORMER_RESIDUAL_GLYPH_WEIGHT)
            + float(TRANSFORMER_RESIDUAL_WORD_WEIGHT)
        ) > 0.0,
        "At least one transformer residual modality weight must be > 0",
    )

    # Probability / range checks
    for name, value in [
        ("AUGMENT_RANDOM_GRAYSCALE_PROB", AUGMENT_RANDOM_GRAYSCALE_PROB),
        ("AUGMENT_COLOR_JITTER_PROB", AUGMENT_COLOR_JITTER_PROB),
        ("AUGMENT_GAUSSIAN_BLUR_PROB", AUGMENT_GAUSSIAN_BLUR_PROB),
        ("AUGMENT_BACKGROUND_PATTERN_PROB", AUGMENT_BACKGROUND_PATTERN_PROB),
        ("AUGMENT_BACKGROUND_SURFACE_PATTERN_PROB", AUGMENT_BACKGROUND_SURFACE_PATTERN_PROB),
        ("AUGMENT_BACKGROUND_LIBRARY_PATTERN_PROB", AUGMENT_BACKGROUND_LIBRARY_PATTERN_PROB),
        ("AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB", AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB),
        ("AUGMENT_PARCHMENT_STAINS_PROB", AUGMENT_PARCHMENT_STAINS_PROB),
        ("AUGMENT_RANDOM_ERASING_PROB", AUGMENT_RANDOM_ERASING_PROB),
        ("AUGMENT_LOCAL_TEXTURE_PROB", AUGMENT_LOCAL_TEXTURE_PROB),
        ("AUGMENT_TONE_CONTRAST_PROB", AUGMENT_TONE_CONTRAST_PROB),
        ("AUGMENT_INK_DEGRADATION_PROB", AUGMENT_INK_DEGRADATION_PROB),
        ("AUGMENT_SPECKLE_MORPH_PROB", AUGMENT_SPECKLE_MORPH_PROB),
        ("AUGMENT_BORDER_CROP_PROB", AUGMENT_BORDER_CROP_PROB),
        ("AUGMENT_GAMMA_PROB", AUGMENT_GAMMA_PROB),
        ("AUGMENT_ZOOM_PROB", AUGMENT_ZOOM_PROB),
        ("AUGMENT_TILT_PROB", AUGMENT_TILT_PROB),
        ("AUGMENT_RESOLUTION_JITTER_PROB", AUGMENT_RESOLUTION_JITTER_PROB),
        ("AUGMENT_RANDAUGMENT_PROB", AUGMENT_RANDAUGMENT_PROB),
        ("MODALITY_DROPOUT_PROB_VISUAL", MODALITY_DROPOUT_PROB_VISUAL),
        ("MODALITY_DROPOUT_PROB_CHAR", MODALITY_DROPOUT_PROB_CHAR),
        ("MODALITY_DROPOUT_PROB_WORD", MODALITY_DROPOUT_PROB_WORD),
        ("TOKEN_SUBSAMPLE_PROB", TOKEN_SUBSAMPLE_PROB),
        ("DROPOUT", DROPOUT),
        ("TRANSFORMER_DROPOUT", TRANSFORMER_DROPOUT),
        ("TILE_BRANCH_TRANSFORMER_DROPOUT", TILE_BRANCH_TRANSFORMER_DROPOUT),
        ("TRANSFORMER_RESIDUAL_CLS_GATE", TRANSFORMER_RESIDUAL_CLS_GATE),
    ]:
        _ensure_prob(name, float(value))

    _ensure_range("AUGMENT_BACKGROUND_PATTERN_ALPHA_RANGE", AUGMENT_BACKGROUND_PATTERN_ALPHA_RANGE, lo=0.0, hi=1.0)
    _ensure_range("AUGMENT_RANDOM_ERASING_SCALE", AUGMENT_RANDOM_ERASING_SCALE, lo=0.0)
    _ensure_range("AUGMENT_RANDOM_ERASING_RATIO", AUGMENT_RANDOM_ERASING_RATIO, lo=0.0)
    _ensure_range("AUGMENT_LOCAL_TEXTURE_ALPHA_RANGE", AUGMENT_LOCAL_TEXTURE_ALPHA_RANGE, lo=0.0, hi=1.0)
    _ensure_range("TOKEN_SUBSAMPLE_FRAC_RANGE", TOKEN_SUBSAMPLE_FRAC_RANGE, lo=0.0, hi=1.0)
    _ensure(AUGMENT_GAUSSIAN_BLUR_KERNEL > 0 and int(AUGMENT_GAUSSIAN_BLUR_KERNEL) % 2 == 1, f"AUGMENT_GAUSSIAN_BLUR_KERNEL must be a positive odd integer, got {AUGMENT_GAUSSIAN_BLUR_KERNEL!r}")
    _ensure(GLYPH_GAUSSIAN_BLUR_KERNEL > 0 and int(GLYPH_GAUSSIAN_BLUR_KERNEL) % 2 == 1, f"GLYPH_GAUSSIAN_BLUR_KERNEL must be a positive odd integer, got {GLYPH_GAUSSIAN_BLUR_KERNEL!r}")
    _ensure_range("GLYPH_RANDOM_ERASING_SCALE", GLYPH_RANDOM_ERASING_SCALE, lo=0.0)
    _ensure(AUGMENT_BACKGROUND_PATTERN_GRID_LINE_WIDTH > 0, f"AUGMENT_BACKGROUND_PATTERN_GRID_LINE_WIDTH must be > 0, got {AUGMENT_BACKGROUND_PATTERN_GRID_LINE_WIDTH!r}")
    _ensure_range("AUGMENT_BACKGROUND_PATTERN_GRID_SPACING_RANGE", AUGMENT_BACKGROUND_PATTERN_GRID_SPACING_RANGE, lo=1.0)
    _ensure(0.0 < float(AUGMENT_BORDER_MAX_FRAC) <= 0.5, f"AUGMENT_BORDER_MAX_FRAC must be in (0, 0.5], got {AUGMENT_BORDER_MAX_FRAC!r}")
    _ensure(0.0 < float(AUGMENT_EDGE_CROP_MAX_FRAC) <= 0.5, f"AUGMENT_EDGE_CROP_MAX_FRAC must be in (0, 0.5], got {AUGMENT_EDGE_CROP_MAX_FRAC!r}")
    if AUGMENT_BACKGROUND_PATTERN_PROB > 0 or AUGMENT_BACKGROUND_SURFACE_PATTERN_PROB > 0 or AUGMENT_BACKGROUND_LIBRARY_PATTERN_PROB > 0 or AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB > 0:
        _ensure(len(AUGMENT_BACKGROUND_PATTERN_TYPES) > 0, "AUGMENT_BACKGROUND_PATTERN_TYPES must be non-empty when background pattern augmentation is enabled")

    # Token subsampling consistency
    _ensure(int(TOKEN_SUBSAMPLE_MIN_KEEP_TILES) >= 1, f"TOKEN_SUBSAMPLE_MIN_KEEP_TILES must be >= 1, got {TOKEN_SUBSAMPLE_MIN_KEEP_TILES!r}")
    _ensure(int(TOKEN_SUBSAMPLE_MIN_KEEP_GLYPHS) >= 1, f"TOKEN_SUBSAMPLE_MIN_KEEP_GLYPHS must be >= 1, got {TOKEN_SUBSAMPLE_MIN_KEEP_GLYPHS!r}")
    _ensure(int(TOKEN_SUBSAMPLE_MIN_KEEP_WORDS) >= 1, f"TOKEN_SUBSAMPLE_MIN_KEEP_WORDS must be >= 1, got {TOKEN_SUBSAMPLE_MIN_KEEP_WORDS!r}")
    _ensure(float(TOKEN_SUBSAMPLE_FRAC_RANGE[0]) > 0.0, f"TOKEN_SUBSAMPLE_FRAC_RANGE lower bound must be > 0, got {TOKEN_SUBSAMPLE_FRAC_RANGE!r}")

    # Glyph / word extraction consistency
    _ensure(float(OCR_GLYPH_CONFIDENCE_THRESHOLD) >= 0.0, f"OCR_GLYPH_CONFIDENCE_THRESHOLD must be >= 0, got {OCR_GLYPH_CONFIDENCE_THRESHOLD!r}")
    _ensure(float(OCR_STRING_CONFIDENCE_THRESHOLD) >= 0.0, f"OCR_STRING_CONFIDENCE_THRESHOLD must be >= 0, got {OCR_STRING_CONFIDENCE_THRESHOLD!r}")
    _ensure(float(GLYPH_QUALITY_MIN_AREA) >= 0.0, f"GLYPH_QUALITY_MIN_AREA must be >= 0, got {GLYPH_QUALITY_MIN_AREA!r}")
    _ensure(float(GLYPH_QUALITY_MIN_WIDTH) >= 0.0, f"GLYPH_QUALITY_MIN_WIDTH must be >= 0, got {GLYPH_QUALITY_MIN_WIDTH!r}")
    _ensure(float(GLYPH_QUALITY_MIN_HEIGHT) >= 0.0, f"GLYPH_QUALITY_MIN_HEIGHT must be >= 0, got {GLYPH_QUALITY_MIN_HEIGHT!r}")
    _ensure(float(GLYPH_QUALITY_MAX_ASPECT_RATIO) > 0.0, f"GLYPH_QUALITY_MAX_ASPECT_RATIO must be > 0, got {GLYPH_QUALITY_MAX_ASPECT_RATIO!r}")
    _ensure(float(WORD_BRANCH_MIN_AREA) >= 0.0, f"WORD_BRANCH_MIN_AREA must be >= 0, got {WORD_BRANCH_MIN_AREA!r}")
    if GLYPH_ONLY_MIDDLE_LETTERS:
        _ensure(int(GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE) >= 3, f"GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE must be >= 3 when GLYPH_ONLY_MIDDLE_LETTERS=True, got {GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE!r}")


validate_runtime_config()


# =============================================================================
# 20) Derived values (keep at bottom; depends on many constants above)
# =============================================================================

RUN_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def build_run_name(training_mode: str | None = None) -> str:
    """
    Build the wandb/run name.

    Important: `main.py` sets `system.TRAINING_MODE` after importing this
    module, so `main.py` should call `recompute_run_name()` to refresh it.
    """
    mode = training_mode or TRAINING_MODE

    # Build module summary for run name (full names with on/off status)
    mod_parts = []
    if USE_VISUAL_MOD:
        tile_type = str(TILE_ENCODER_TYPE).strip().lower()
        if tile_type == "dinov2":
            tile_type_display = "DINOv2"
        elif tile_type == "convnext":
            tile_type_display = "Convnext"
        else:
            tile_type_display = TILE_ENCODER_TYPE
        mod_parts.append(f"Visual{tile_type_display}_ON")
    else:
        mod_parts.append("Visual_OFF")
    if USE_CHAR_MOD:
        # Use glyph encoder type (new architecture uses GLYPH_ENCODER_TYPE, not CHAR_ENCODER_TYPE)
        glyph_type_display = (
            GLYPH_ENCODER_TYPE.replace("convnext_tiny", "Convnext").replace("swin_tiny", "Swin")
        )
        mod_parts.append(f"Glyph{glyph_type_display}_ON")
    else:
        mod_parts.append("Glyph_OFF")
    if USE_WORD_MOD:
        mod_parts.append("Word_ON")
    else:
        mod_parts.append("Word_OFF")
    mod_str = "_".join(mod_parts)

    return (
        f"{mod_str}_{mode}"
        f"_{PROJECT_NAME}_{RUN_DATE}"
        # Intentionally omit tile chunk size (was: _tile{TILE_ENCODE_CHUNK_SIZE})
        f"_bs{PER_GPU_BATCH_SIZE}x{GRADIENT_ACCUMULATION_STEPS}"
        f"_lat{LATENT_DIM}"
        f"_adapt_sum{int(USE_BRANCH_ADAPTERS_AND_SUMMARIZERS)}"
        f"_layers{NUM_LAYERS}"
        f"_heads{NUM_HEADS}"
    )


RUN_NAME = build_run_name()


def recompute_run_name(training_mode: str | None = None) -> None:
    global RUN_NAME
    RUN_NAME = build_run_name(training_mode)
