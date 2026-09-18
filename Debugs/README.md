# Debugs

Standalone diagnostic scripts and checked-in output snapshots.

This folder is not imported by the main training or projection pipeline. Scripts
here are useful for understanding failures, validating augmentations, inspecting
background leakage, and analyzing branch behavior. Output subfolders are snapshots
from past runs; they should not be treated as source data.

Rule of thumb:
- run commands from the repo root
- each script writes by default to an `outputs/` folder under its own subfolder
- DB-backed scripts require a working `db_config.ini` and access to the image/XML paths stored in the DB

Role in the page workflow:

- Visualization scripts inspect how one page becomes tiles, glyphs, words, masks,
  and attention/diagnostic artifacts.
- Augmentation scripts inspect train-time perturbations applied after extraction.
- Background leakage scripts check whether selected tiles/glyphs still encode
  collection-specific backgrounds.
- Impact/test-result scripts inspect how latent similarities and branch scores
  create false positives/false negatives in clustering or fixed-pair evaluation.

## Letter appearance and confidence

### `Debugs/LetterAppearance/analyze_geniza_letters.py`

Samples Geniza ALTO pages evenly across manuscripts and compares aleph and mem
with the full Hebrew alphabet. It exports unfiltered letter counts and raw
OCR-confidence quantile intervals for every letter (plus a secondary box plot). It can also consume a fixed CSV of
`manuscript_id,xml_path` rows for a fully reproducible or DB-free run. See
`Debugs/LetterAppearance/README.md` for usage and output details.
Use `--apply-confidence-filter` to compare against the configured training GC
cutoff; filtered and unfiltered runs are written separately by default.

## ConvNeXt

### `Debugs/ConvNeXt/convnext_blur_sensitivity.py`

What it does:
- extracts real manuscript tiles and/or glyphs
- applies blur and downsample/upsample degradation
- measures how much the ConvNeXt embedding changes

Input:
- required: `--image-path`
- optional: `--xml-path` if you want glyphs

Run:

```bash
python Debugs/ConvNeXt/convnext_blur_sensitivity.py \
  --image-path path/to/page.jpg \
  --xml-path path/to/page.xml
```

Output:
- default folder: `Debugs/ConvNeXt/outputs/convnext_blur_sensitivity/`
- main files:
  - `summary.md`
  - `measurements.csv`
  - `aggregate_plots/*.png`
  - `degradation_overviews/*.png`
  - `tile_convnext/.../previews/*.png`
  - `glyph_convnext/.../previews/*.png`

### `Debugs/ConvNeXt/convnext_latent_restoration.py`

What it does:
- inverts ConvNeXt pooled features back into image space
- can use synthetic probes or real tiles/glyphs

Input:
- none required for synthetic mode
- optional real-input mode: `--image-path`, `--xml-path`

Run:

```bash
python Debugs/ConvNeXt/convnext_latent_restoration.py
```

Real manuscript mode:

```bash
python Debugs/ConvNeXt/convnext_latent_restoration.py \
  --image-path path/to/page.jpg \
  --xml-path path/to/page.xml
```

Output:
- default folder: `Debugs/ConvNeXt/outputs/convnext_latent_restoration/`
- main files:
  - `summary.md`
  - `summary.csv`
  - `tile_convnext/.../*.png`
  - `glyph_convnext/.../*.png`

### `Debugs/ConvNeXt/convnext_tile_glyph_histograms.py`

What it does:
- computes image-level ConvNeXt tile and glyph embeddings
- samples same-manuscript and cross-manuscript pairs
- writes separate cosine histograms for tiles and glyphs

Input:
- either DB-backed split mode with `--split`
- or fixed CSV mode with `--test-set-csv`

Run:

```bash
python Debugs/ConvNeXt/convnext_tile_glyph_histograms.py \
  --split val \
  --n-same 200 \
  --n-cross 500
```

Fixed CSV mode:

```bash
python Debugs/ConvNeXt/convnext_tile_glyph_histograms.py \
  --test-set-csv results_analysis/test_set/clusters_images_metadata.csv
```

Output:
- default folder: `Debugs/ConvNeXt/outputs/convnext_tile_glyph_histograms/`
- main files:
  - `tile_convnext_histogram.png`
  - `glyph_convnext_histogram.png`
  - `pair_cosines.csv`
  - `summary.md`

## DINOv2

### `Debugs/DINOv2/dinov2_tile_glyph_histograms.py`

What it does:
- same idea as the ConvNeXt histogram script
- uses DINOv2 features for both tiles and glyphs

Input:
- either DB-backed split mode with `--split`
- or fixed CSV mode with `--test-set-csv`

Run:

```bash
python Debugs/DINOv2/dinov2_tile_glyph_histograms.py \
  --split val \
  --n-same 200 \
  --n-cross 500
```

Output:
- default folder: `Debugs/DINOv2/outputs/dinov2_tile_glyph_histograms/`
- main files:
  - `tile_dinov2_histogram.png`
  - `glyph_dinov2_histogram.png`
  - `pair_cosines.csv`
  - `summary.md`

## Augmentations

### `Debugs/Augmentations/branch_augmentation_label_flips.py`

What it does:
- loads a trained checkpoint
- runs tile-only or glyph-only prediction through the full model
- applies one augmentation to all tiles together or all glyphs together
- records cases where the prediction flips

Input:
- required: `--checkpoint`
- optional: `--split`, `--branches`, `--save-flip-samples`

Run:

```bash
python Debugs/Augmentations/branch_augmentation_label_flips.py \
  --checkpoint Results/best_model/your_checkpoint.pth \
  --split val \
  --batch-size 16 \
  --save-flip-samples 10
```

Output:
- default folder: `Debugs/Augmentations/outputs/branch_augmentation_label_flips/`
- main files:
  - `label_flips_<split>.csv`
  - `summary_<split>.txt`
  - `flip_samples/<branch>/<augmentation>/*.png`
  - `flip_samples_manifest_<split>.csv`

### `Debugs/Augmentations/sample_geniza_fragment_and_visualize_augmentations.py`

What it does:
- picks one Geniza fragment from `geniza_image_latents`
- extracts one tile and one glyph
- renders grids showing the current training augmentations for each modality

Input:
- DB access to `geniza_image_latents`
- no explicit image path needed

Run:

```bash
python Debugs/Augmentations/sample_geniza_fragment_and_visualize_augmentations.py --random
```

Output:
- default folder: `Debugs/Augmentations/outputs/sample_geniza_fragment_and_visualize_augmentations/`
- main files:
  - `tile_aug_grid.png`
  - `glyph_aug_grid.png`
  - `selected_geniza_row.json`

## Background Leakage

### `Debugs/BackgroundLeakage/inspect_geniza_tiles_by_library_background.py`

What it does:
- samples Geniza images from the DB
- joins `geniza_manuscript_shelfmark` to get `normalized_library`
- compares selected XML-guided tiles against library background templates

Input:
- DB access
- `Backgrounds/` folder with library template images

Run:

```bash
python Debugs/BackgroundLeakage/inspect_geniza_tiles_by_library_background.py \
  --limit 40 \
  --min-text-coverage 0.85 \
  --max-tiles 8
```

Output:
- default folder: `Debugs/BackgroundLeakage/outputs/geniza_tile_background_inspection/`
- main files:
  - `tile_background_metrics.csv`
  - `grids/<library>/*.png`

### `Debugs/BackgroundLeakage/inspect_geniza_glyphs_by_library_background.py`

What it does:
- samples Geniza images from the DB
- extracts glyphs from XML
- compares glyph crops against library background templates
- can restrict to middle letters only

Input:
- DB access
- `Backgrounds/` folder with library template images

Run:

```bash
python Debugs/BackgroundLeakage/inspect_geniza_glyphs_by_library_background.py \
  --limit 40 \
  --max-glyphs 24 \
  --only-middle-glyphs
```

Output:
- default folder: `Debugs/BackgroundLeakage/outputs/geniza_glyph_background_inspection/`
- main files:
  - `glyph_background_metrics.csv`
  - `grids/<library>/*.png`

## Impact Analysis

### `Debugs/impact_analysis/discriminability_calibration.py`

What it does:
- loads a trained checkpoint
- computes branch-level and fusion-subset embeddings
- compares same-manuscript vs cross-manuscript cosine distributions

Input:
- required: `--checkpoint`
- optional: `--split` or `--test-set-csv`

Run:

```bash
python Debugs/impact_analysis/discriminability_calibration.py \
  --checkpoint Results/best_model/your_checkpoint.pth \
  --split val
```

Output:
- default folder: `Debugs/impact_analysis/outputs/discriminability_calibration/`
- main files:
  - CSV summaries
  - histogram / comparison plots
  - text summary files

Note:
- exact filenames depend on the selected modes and pair type

### `Debugs/impact_analysis/swap_sensitivity.py`

What it does:
- loads a trained checkpoint
- swaps glyphs or tiles between two images
- measures how much logits / embeddings move
- can also run full subset swaps over modality combinations

Input:
- required: `--checkpoint`
- optional: pair selection via `--index-a`, `--index-b`, `--pair-type`, `--num-tests`

Run:

```bash
python Debugs/impact_analysis/swap_sensitivity.py \
  --checkpoint Results/best_model/your_checkpoint.pth \
  --pair-type on \
  --num-tests 5
```

Output:
- default folder: `Debugs/impact_analysis/outputs/swap_sensitivity/`
- main files:
  - per-test visualization PNGs
  - printed console summary of embedding/logit deltas

## Notes

- Most DB-backed scripts assume `psycopg2` is installed in the current environment.
- Many scripts rely on image/XML paths that live on `/nas/...`.
- For heavy model scripts, use `--device cpu` if CUDA is unavailable, but some runs will be slow.
