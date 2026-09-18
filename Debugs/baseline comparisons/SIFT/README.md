# Whole-page SIFT CE baseline

`train_sift_ce.py` extracts and caches one 256-D vector per complete page by
mean/std pooling RootSIFT descriptors. A small learned 256-D projection and
linear manuscript classifier are trained with cross-entropy; the learned
projection is evaluated on the canonical Geniza test with mAP and KNN@1/5/10.

The stage splits, effective batch, optimizer settings, progress reporting, and
checkpoint behavior match the ConvNeXt/DINOv2 baseline scripts.
Every GPU exposed through `CUDA_VISIBLE_DEVICES` is selected automatically.
Exposed physical GPUs are locally renumbered from zero, so no `--gpu-ids` flag
is needed in the usual case.

```bash
CUDA_VISIBLE_DEVICES=2,3 \
/home/omerv/anaconda3/envs/DeepEnv/bin/python -u \
  Debugs/SIFT/train_sift_ce.py \
  --stage stage1
```

Feature extraction is CPU/NAS-bound and is performed only once. Cached vectors
are stored under `Debugs/SIFT/outputs/sift_ce/<stage>/feature_cache/`.

Stage 2:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
/home/omerv/anaconda3/envs/DeepEnv/bin/python -u \
  Debugs/SIFT/train_sift_ce.py \
  --stage stage2 \
  --init-checkpoint Debugs/SIFT/outputs/sift_ce/stage1/best.pth
```
