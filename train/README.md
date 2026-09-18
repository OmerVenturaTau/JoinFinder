# train

Training-time data and optimization code.

Important files:

- `dataset.py`: `ManuscriptDataset`, XML-guided tile/glyph/word loading, transforms,
  padding collate, and XML warning collection.
- `split_data.py`: DB-table split construction, including use of `dataset_split`
  when present.
- `db_loader.py`: DB connection and table loading helpers.
- `trainer.py`: epoch loop, evaluation, checkpointing, auxiliary loss handling,
  PCA diagnostics, attention logging, and W&B reporting.
- `metric_learning.py`: memory-bank supervised contrastive queue/loss utilities.
- `pk_sampler.py`: P-K sampler for balanced metric-learning batches.

The main executable is still `main.py` at the repository root. This folder should
not contain older `pretrain/` or `finetune/` training trees; those empty legacy
directories were removed to avoid path ambiguity.

Runtime flow:

1. `main.py` selects a stage/table and asks `split_data.py` for DB-backed splits.
2. `dataset.py` receives paths/labels, loads PIL images, resolves XML paths, and
   extracts modality inputs.
3. `tile_collate_with_padding` pads variable numbers of tiles/glyphs/words into
   masks consumed by `models.MultiModal`.
4. `trainer.py` computes joint and auxiliary losses, logs diagnostics, and saves
   checkpoints under `Results/`.

Page-to-batch details:

- A dataset item begins with one `image_path`, one label, and optional `xml_path`.
- XML is resolved only when a modality needs it: tile XML extraction, glyphs, or
  words.
- The visual stream returns tile tensors `[N, 3, TILE_SIZE, TILE_SIZE]`, normalized
  tile center coordinates `[N, 2]`, and tile page segments.
- The glyph stream returns glyph tensors `[M, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE]`,
  metadata with normalized boxes, Hebrew character class IDs, and page segments.
- The enabled word stream returns Python lists because word counts and line
  grouping are ragged.
- `tile_collate_with_padding` pads tiles and glyphs across the batch, creates
  `tile_valid_mask` and `char_valid_mask`, and emits fully masked dummy tokens
  when an entire batch has zero real tokens. That dummy-token behavior is
  intentional: it avoids zero-sized tensors while still telling the model that no
  valid tile/glyph exists.

Training-only regularization enters in two places:

- Tile and glyph image augmentations are applied inside the dataset transforms for
  the `train` split. Eval/test/projection use deterministic tensor conversion and
  normalization.
- Modality dropout and token subsampling are applied later in `models.MultiModal`,
  not in the dataset. The masks from this folder are what make that subsampling
  safe.

Clustering dependency: `aftertune/project_geniza_to_latent.py` reuses
`ManuscriptDataset` and `tile_collate_with_padding`, so extraction or mask changes
here alter exported Geniza latents and downstream neighbor graphs.

Run tests with the project conda environment:

```bash
conda run -n NN pytest tests/test_dataset_coordinates.py tests/test_sampling_balance.py
```
