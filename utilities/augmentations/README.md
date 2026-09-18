# utilities/augmentations

Training and diagnostic augmentations for manuscript images.

- `background_overlays.py`: random library/background template overlay used to
  reduce collection-background leakage.
- `manuscript_augmentations.py`: resolution, zoom, tilt, local texture, tone,
  border, white-background cleanup, and weighted augmentation wrappers.

The probability and strength settings live in `config/model_regularization.json`
and are exposed through `system.py`.

Where augmentation sits in the page pipeline:

- Augmentation is applied after page-level extraction creates tile/glyph crops.
  The full page is not globally augmented before extraction.
- Tile augmentation is assembled in `main.py` as an 80% apply gate around a
  normalized `WeightedOneOf` transform over tile crops. The weighted pool can
  perturb color, grayscale, blur, library/background texture, parchment stains,
  white-background cleanup, local texture, tone/contrast, tilt, borders, zoom,
  resolution, or RandAugment. Tensor random erasing and normalization happen
  after the PIL transform.
- Glyph augmentation is assembled inside `train/ManuscriptDataset` with separate
  glyph-specific probabilities and smaller blur/erasing settings because 128x128
  glyph crops are fragile. It uses the same 80% apply gate plus normalized
  glyph-specific weights.
- Eval, test, and projection should stay deterministic: tensor conversion and
  normalization only.

The goal of these augmentations is to reduce reliance on collection/background
artifacts while keeping letter and handwriting evidence usable for the latent
space that later feeds clustering.
