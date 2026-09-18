# Debugs/ConvNeXt

ConvNeXt branch diagnostics.

- `train_convnext_ce.py`: trains a whole-page manuscript-ID classifier with
  cross-entropy on the repository's stage-1 pretraining or stage-2 fine-tuning
  dataset; `--backbone` selects ConvNeXt or DINOv2.
- `../DINOv2/train_dinov2_ce.py`: runs the identical protocol with the
  pretrained DINOv2 ViT-B/14 backbone and its native CLS-token representation.
- `convnext_blur_sensitivity.py`: measures embedding sensitivity to blur and
  down/up-sampling for tiles or glyphs.
- `convnext_latent_restoration.py`: inverts ConvNeXt pooled features back into
  image space for interpretability.
- `convnext_tile_glyph_histograms.py`: compares same-manuscript and
  cross-manuscript cosine distributions for ConvNeXt embeddings.

Generated output normally lives under `Debugs/ConvNeXt/outputs/` or `Results/`.

## Whole-page CE baseline

Stage 1 starts from the pretrained timm ConvNeXt-Tiny checkpoint and trains the
entire backbone and linear manuscript classifier with cross-entropy. Complete
pages are resized with preserved aspect ratio and padded to the ConvNeXt input
size; no page content is center-cropped away. After every epoch the script prints train/validation CE loss and
accuracy, followed by mAP and KNN@1/5/10 on the canonical 242-page Geniza
cluster test (`results_analysis/test_set/clusters_images_metadata.csv`):

By default, CUDA training, validation, and Geniza evaluation use every GPU
exposed to the process through `DataParallel`. For example,
`CUDA_VISIBLE_DEVICES=2,3` exposes physical GPUs 2 and 3 as local IDs 0 and 1,
and both are selected automatically. Use `--gpu-ids` only to select a subset of
the already-visible devices.

Batching and optimizer defaults match `main.py`: 4 pages per exposed GPU and
four accumulation steps (with two GPUs: global microbatch 8, effective batch
32), AdamW betas
0.9/0.999, weight decay 0.05, gradient clipping at 1.0, 12 epochs, and cosine
decay to 1e-6. Stage 1 uses LR 2e-5; stage 2 initialized from a checkpoint uses
the actual `main.py` checkpoint LR of 1e-4.

Training, validation, and Geniza evaluation have live `tqdm` progress bars.
Whole-page JPEGs use decoder-side downsampling before letterboxing, and the
loader defaults to four workers with prefetch factor two to avoid flooding the
NAS with large concurrent reads.

```bash
CUDA_VISIBLE_DEVICES=2,3 \
/home/omerv/anaconda3/envs/DeepEnv/bin/python \
  Debugs/ConvNeXt/train_convnext_ce.py \
  --stage stage1
```

Stage 2 initializes from the best stage-1 checkpoint while preserving its
manuscript-ID output mapping:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
/home/omerv/anaconda3/envs/DeepEnv/bin/python \
  Debugs/ConvNeXt/train_convnext_ce.py \
  --stage stage2 \
  --init-checkpoint Debugs/ConvNeXt/outputs/convnext_ce/stage1/best.pth
```

Use `--resume .../last.pth` to continue an interrupted run. Outputs include
`best.pth`, `last.pth`, `history.csv`, `run_config.json`, and test metrics under
`Debugs/ConvNeXt/outputs/convnext_ce/<stage>/`.

## DINOv2 version

The DINOv2 entry point keeps the complete page through the same aspect-ratio
preserving letterbox transform, uses the checkpoint's 518x518 input, and trains
the full `vit_base_patch14_dinov2.lvd142m` model with CE. Retrieval uses its
768-D CLS representation.

```bash
/home/omerv/anaconda3/envs/DeepEnv/bin/python \
  Debugs/DINOv2/train_dinov2_ce.py \
  --stage stage1
```

```bash
/home/omerv/anaconda3/envs/DeepEnv/bin/python \
  Debugs/DINOv2/train_dinov2_ce.py \
  --stage stage2 \
  --init-checkpoint Debugs/ConvNeXt/outputs/dinov2_ce/stage1/best.pth
```

The same behavior is available from the main script with `--backbone dinov2`.
Both backbones use the same main-pipeline batch settings for a controlled
comparison; `--batch-size` can override the global microbatch.

## Center-crop experiment

Pass `--preprocessing center-crop` to train, validate, and evaluate Geniza with
the backbone checkpoint's native resize-plus-center-crop geometry. Training
uses the same deterministic center crop plus color jitter; it does not switch
to a random resized crop. Center-crop outputs are isolated under
`outputs/<backbone>_ce_center_crop/`, so the full-page checkpoints are not
overwritten.

```bash
CUDA_VISIBLE_DEVICES=2,3 \
/home/omerv/anaconda3/envs/DeepEnv/bin/python -u \
  Debugs/ConvNeXt/train_convnext_ce.py \
  --stage stage1 \
  --preprocessing center-crop
```

For stage 2, initialize from the corresponding center-crop stage-1 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
/home/omerv/anaconda3/envs/DeepEnv/bin/python -u \
  Debugs/ConvNeXt/train_convnext_ce.py \
  --stage stage2 \
  --preprocessing center-crop \
  --init-checkpoint Debugs/ConvNeXt/outputs/convnext_ce_center_crop/stage1/best.pth
```
