# aftertune

Active post-training Geniza projection and nearest-neighbor jobs.

- `project_geniza_to_latent.py`: loads a trained checkpoint, projects Geniza images
  into model latent space, and writes `geniza_image_latents`.
- `recompute_geniza_pca.py`: recomputes PCA/search vectors from stored latents.
- `geniza_top_neighbors_gpu.py`: GPU KNN search over Geniza latent vectors and DB
  upload of neighbor tables.
- `launch_project_geniza_multi_gpu.py`: shards projection work across GPUs by
  manuscript ID.

These scripts depend on DB tables and paths configured in `system.py`. They should
use active model/utilities code, not `Drafts/AfterTune`.

Typical order:

```bash
conda run -n NN python aftertune/project_geniza_to_latent.py \
  --checkpoint Results/best_model/example.pth

conda run -n NN python aftertune/recompute_geniza_pca.py
conda run -n NN python aftertune/geniza_top_neighbors_gpu.py
```

`project_geniza_to_latent.py` must use the same architecture flags as the
checkpoint. Checkpoint compatibility helpers in `utilities/checkpoint_utils.py`
patch known modality/summary-token differences for older checkpoints.

Page-to-clustering workflow:

1. `project_geniza_to_latent.py` reads rows from
   `system.GENIZA_IMAGE_INFORMATION_TABLE`.
2. For each page, it builds the same `ManuscriptDataset` item shape used during
   training: XML-guided tiles, filtered glyphs, optional words, masks, and page
   metadata.
3. It calls the model's latent path rather than using classification logits as the
   output product.
4. It writes one row per image to `system.GENIZA_IMAGE_LATENTS_TABLE`, including
   `latent_vector`, `latent_vector_search`, `num_visual_patches`, `num_glyphs`,
   and `num_words`.
5. `recompute_geniza_pca.py` refreshes search vectors after projection batches are
   complete.
6. `geniza_top_neighbors_gpu.py` computes cosine KNN from stored vectors and writes
   other-manuscript and overall-neighbor tables.

This folder is therefore downstream of extraction, masking, augmentation choices
from training, and model architecture. Projection should use eval-style transforms
only; introducing train-time augmentation here would make clustering unstable.
