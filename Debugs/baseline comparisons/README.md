# SIFT Baseline Latents

Create fixed-size latent vectors from classical SIFT features for baseline comparisons.

Examples:

```bash
/home/ventura/miniconda3/envs/NN/bin/python "Debugs/baseline comparisons/sift_latents.py" \
  --image-dir /path/to/images \
  --recursive \
  --output "Debugs/baseline comparisons/outputs/sift_latents.npz"
```

```bash
/home/ventura/miniconda3/envs/NN/bin/python "Debugs/baseline comparisons/sift_latents.py" \
  --image-list image_paths.txt \
  --output "Debugs/baseline comparisons/outputs/sift_latents.npz" \
  --csv-output "Debugs/baseline comparisons/outputs/sift_latents.csv"
```

The NPZ contains:

- `paths`: image paths.
- `latents`: `[N, D]` float32 latent matrix.
- `num_keypoints`: number of SIFT keypoints detected per image.
- `metadata`: JSON string with extraction settings and skipped-image details.

Default pooling is `mean_std`, which creates a 256-dimensional vector from pooled RootSIFT descriptors.

## Pairwise Cluster Comparison

To create an Excel comparison file on the same test-set metadata used by
`results_analysis/test_set/compare_clusters_pairs.py`:

```bash
/home/ventura/miniconda3/envs/NN/bin/python "Debugs/baseline comparisons/compare_clusters_pairs_sift.py"
```

This defaults to:

- input: `results_analysis/test_set/clusters_images_metadata.csv`
- output: `Debugs/baseline comparisons/outputs/clusters_images_metadata_pairs_sift.xlsx`

The Excel contains a `pairs` sheet and a `metrics` sheet with same/different
pair separation metrics and image retrieval metrics including `mAP`,
`recall@k`, and `hit@k`.

You can also save and reuse the SIFT latents:

```bash
/home/ventura/miniconda3/envs/NN/bin/python "Debugs/baseline comparisons/compare_clusters_pairs_sift.py" \
  --latents-output "Debugs/baseline comparisons/outputs/sift_latents_test_set.npz"
```

```bash
/home/ventura/miniconda3/envs/NN/bin/python "Debugs/baseline comparisons/compare_clusters_pairs_sift.py" \
  --load-latents "Debugs/baseline comparisons/outputs/sift_latents_test_set.npz"
```

## Frozen ConvNeXt and DINOv2 baselines

`compare_clusters_pairs_backbone.py` evaluates off-the-shelf visual features
without loading a trained JoinsFinder checkpoint or its learned 512-D adapter.
By default, it passes each complete image through the model's native pretrained
evaluation transform and uses the untouched backbone output directly. It runs:

- `convnext_tiny.fb_in22k_ft_in1k`
- `vit_base_patch14_dinov2.lvd142m`

Each encoder produces one native 768-dimensional descriptor per evaluation
image. ConvNeXt and DINOv2 encode the image pixels. AlephBERT uses the paired
`xml_path`, restores Hebrew OCR reading order, and mean-pools native non-special
token states across as many 512-token chunks as the complete page requires.
It does not use a learned adapter or discard text after the first context
window. Vectors are L2-normalized and scored with the same cosine,
pair-separation, mAP, Recall@K, and Hit@K metrics as the SIFT baseline.
Pages with no Hebrew OCR surviving the configured confidence threshold cannot
receive a text embedding; they are excluded from AlephBERT scoring and recorded
in the output metadata. Compare models on a common covered subset when quoting
their metrics side by side.

```bash
/home/omerv/anaconda3/envs/DeepEnv/bin/python \
  "Debugs/baseline comparisons/compare_clusters_pairs_backbone.py"
```

Run only one backbone:

```bash
/home/omerv/anaconda3/envs/DeepEnv/bin/python \
  "Debugs/baseline comparisons/compare_clusters_pairs_backbone.py" \
  --backbone dinov2
```

Run only the AlephBERT OCR baseline:

```bash
/home/omerv/anaconda3/envs/DeepEnv/bin/python \
  "Debugs/baseline comparisons/compare_clusters_pairs_backbone.py" \
  --backbone alephbert
```

The output directory receives one Excel pair table, metrics CSV, and latent NPZ
per encoder. The default files end in `_native` and contain a native 768-D
representation. To explicitly test local visual tile descriptors instead,
use `--input-mode tiles`; in that optional mode, `--pooling mean_std` creates a
1536-D pooled representation without a learned adapter. `--input-mode` does
not affect AlephBERT because its source is the OCR XML rather than image pixels.
