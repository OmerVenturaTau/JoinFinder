# losses

Loss modules used by the active training loop.

- `arcface_loss.py`: ArcFace classifier/loss implementation for angular-margin
  manuscript classification.
- `combined_loss.py`: combines joint ArcFace/CE loss, optional branch auxiliary
  losses, and latent regularization. Geniza contrastive loss is orchestrated from
  `main.py`/`train/trainer.py` when enabled in config.

Current default in `config/loss.json`: ArcFace is enabled, CE is disabled,
auxiliary tile/glyph losses are enabled, word auxiliary loss is zero because the
word modality is disabled by default.

Role in the page-to-clustering pipeline:

- The loss does not run during clustering, but it shapes the latent space that
  clustering uses later.
- ArcFace trains the fused page latent to separate manuscript labels angularly.
- Auxiliary tile/glyph losses use branch latents so individual branches remain
  discriminative instead of relying only on fusion.
- Optional Geniza contrastive loss is orchestrated by `main.py`/`trainer.py` and
  can further structure same-manuscript or related-fragment neighborhoods when
  enabled.
