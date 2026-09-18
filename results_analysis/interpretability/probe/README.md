# Interpretability probes

This folder owns latent projection and downstream probes. Projection is not part
of training: choose a finished checkpoint, populate a named projection run, and
then evaluate that immutable run in each available latent space.

## 1. Create the vectors table

The table supports projection runs with different enabled modalities and vector
dimensions. The command is a dry run unless `--apply` is present.

```bash
conda run -n NN python Drafts/create_interpretability_experiment_vectors_table.py --apply
```

## 2. Project manuscripts

With no manuscript selector, every image in
`pretrain_finetune_oriental_non_oriental_train_val_test_split` is projected.

```bash
conda run -n NN python results_analysis/interpretability/probe/project_manuscripts.py \
  --checkpoint Results/best_model/best_model_stage2_V_C_W_20260824_012240.pth \
  --run-name stage2_vcw
```

Restrict a run with repeated IDs or a file. A CSV file must contain a
`manuscript_id` column; a text file contains one ID per line.

```bash
conda run -n NN python results_analysis/interpretability/probe/project_manuscripts.py \
  --checkpoint Results/best_model/best_model_stage2_V_C_W_20260824_012240.pth \
  --run-name selected_stage2 \
  --manuscript-list manuscripts.csv
```

Rows are upserted by `(projection_run, image_path)`, so an interrupted run can be
rerun with the same arguments. Tile, glyph, or word vectors are null when the
checkpoint disables that modality or the image has no usable evidence for it.

Use a distinct `--run-name` for every checkpoint/configuration you want to
compare. For example, project the current stage-1 and stage-2 checkpoints as two
independent runs:

```bash
conda run -n NN python results_analysis/interpretability/probe/project_manuscripts.py \
  --checkpoint Results/best_model/best_model_stage1_V_C_W_20260905_001355.pth \
  --run-name stage1_vcw_20260905

conda run -n NN python results_analysis/interpretability/probe/project_manuscripts.py \
  --checkpoint Results/best_model/best_model_stage2_V_C_W_20260906_082447.pth \
  --run-name stage2_vcw_20260906
```

## 3. Run probes

```bash
conda run -n NN python results_analysis/interpretability/probe/run_dating_probe.py \
  --projection-run stage2_vcw

conda run -n NN python results_analysis/interpretability/probe/run_dating_linear_regression.py \
  --projection-run stage2_vcw

conda run -n NN python results_analysis/interpretability/probe/run_oriental_probe.py \
  --projection-run stage2_vcw
```

The dating MLP, dating linear regression, and Oriental logistic probe compare
tile, glyph, word, and shared vectors by default. Use, for
example, `--spaces tile shared` to select a subset.

Repeat the probe commands for each projection run you created, changing only
`--projection-run`. The 11 known excluded manuscript IDs declared in
`common.py` are filtered while vectors are loaded, before manuscript splitting,
cross-validation, scaling, probe training, and evaluation. Their vectors can
remain in the projection table for other analyses. Every probe's `config.json`
records the exclusion list used for that run.

Splits are created from unique manuscripts, not from the source table's page
split. Seventy percent of eligible manuscripts form the training partition and
30 percent are untouched final test manuscripts. Five-fold stratified tuning is
performed only inside the 70 percent. Dating buckets represented by fewer than
10 manuscripts are recorded in the split artifact as excluded and are not used
to fit or evaluate the probe.

Outputs are written below `outputs/<experiment>/<projection-run>/` and include:

- the exact split and CV-fold assignment for every manuscript;
- run configuration and cross-validation summaries;
- image- and manuscript-level predictions and probabilities;
- per-space metrics and confusion matrices.

All pages from an eligible manuscript are retained. Both probes give every
target class equal total training weight and every manuscript equal weight
within its class, so manuscripts with many images cannot dominate training.
Final manuscript probabilities are the mean of their page probabilities.
