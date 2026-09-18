# config

JSON configuration fragments loaded by `system.py`.

`system.py` deep-merges these files in this order:

1. `model_architecture.json`
2. `model_regularization.json`
3. `training.json`
4. `loss.json`

The merged config is validated at import time by
`system.validate_runtime_config()`. A bad value should fail early before training
or projection starts.

Current files:

- `model_architecture.json`: tile, glyph, word, fusion, and modality settings.
  Current defaults enable tile, glyph, and word modalities, use ConvNeXt
  tile/glyph encoders, symmetric fusion, optional alternative fusion settings,
  active learned-query summarizers and 512-dimensional branch adapters, and
  1536-dimensional latents (three adapted modality blocks).
- `model_regularization.json`: augmentation probabilities, modality dropout,
  glyph filtering, and token subsampling settings.
- `training.json`: dataloader, optimizer, scheduler, chunk-size, DataParallel,
  and batch/accumulation settings.
- `loss.json`: ArcFace/CE weights, auxiliary branch losses, latent regularization,
  and optional Geniza contrastive settings.

Do not put credentials, `db_config.ini` content, NAS paths that differ by machine,
or production secrets here. Put stable defaults here only when later code should
read them through `system.py`.

Workflow-sensitive config groups:

- `fusion.use_branch_adapters_and_summarizers` is the single architecture switch
  for the optional learned-query branch summarizers and symmetric-fusion
  projection adapters. It is currently `true`; the configured
  tile/glyph/word query counts are 8/8/24, respectively, and each pooled 768-d
  branch is adapted to `fusion.symmetric_branch_dim` (currently 512). The
  symmetric latent width is derived automatically from this choice and the
  enabled modality count.
- `tile.*` controls how tile tensors are encoded after `train/dataset.py` extracts
  XML-guided crops. `num_summary_tokens` sets the query count used when the
  combined architecture switch is enabled.
- `glyph.*` controls glyph crop size, cap, encoder, and optional set-summarizer
  settings.
- `word.*` controls the enabled word branch, including optional dictionary and
  TF-IDF filtering/gating and optional line-set summarization. The optional gate uses term frequency within each OCR
  line and the fixed book-level IDF values in `bhsa_normalized_dict.json`, so it
  is independent of batch composition. It remains disabled by default. The OCR
  input budget is applied to complete lines so the language encoder retains
  useful context.
- `modalities.*` decides which streams are active in both training and projection.
- `augmentation.*` and `glyph.*` regularization keys shape train-time crop
  perturbations; projection should not use these train-time perturbations.
- `modality_dropout.*` and `token_subsample.*` are model-side training
  regularizers that operate on masks/tokens after collation.
- `training.*` also defines the global seed and separate learning rates for the
  pretrained AlephBERT encoder and newly initialized word-branch layers.
- `loss.*` determines how the page latent is supervised before it is reused for
  Geniza clustering, including the word auxiliary-loss fade-out schedule.
