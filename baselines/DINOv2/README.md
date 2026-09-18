# debug_tools/DINOv2

DINOv2 embedding diagnostics.

`train_dinov2_ce.py` is the whole-page DINOv2 ViT-B/14 CE baseline. It uses
the same stage-1/stage-2 classification splits and canonical Geniza retrieval
test as the ConvNeXt version, printing train/validation loss and accuracy plus
test mAP and KNN@1/5/10 after each epoch. CUDA runs automatically use every GPU
exposed to the process. For example, `CUDA_VISIBLE_DEVICES=2,3` uses both
physical GPUs through local IDs 0 and 1. Like `main.py`, the default is 4 pages
per exposed GPU with 4-step gradient accumulation (effective batch 32 across
two GPUs).

```bash
CUDA_VISIBLE_DEVICES=2,3 \
/home/omerv/anaconda3/envs/DeepEnv/bin/python \
  debug_tools/DINOv2/train_dinov2_ce.py --stage stage1
```

Center-crop training and Geniza evaluation use DINOv2's native 518px transform
and write to a separate output directory:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
/home/omerv/anaconda3/envs/DeepEnv/bin/python -u \
  debug_tools/DINOv2/train_dinov2_ce.py \
  --stage stage1 \
  --preprocessing center-crop
```

`dinov2_tile_glyph_histograms.py` computes same-manuscript and cross-manuscript
cosine histograms for DINOv2 features. This is diagnostic only; the current
default tile encoder in config is ConvNeXt.
