# Debugs/impact_analysis

Impact-analysis scripts for branch contribution and similarity behavior.

- `diagnose_similarity_issue.py`: case-level similarity diagnosis.
- `swap_sensitivity.py`: tests how swapping branch inputs affects embeddings;
  `--source classification` samples a configured train/validation/test split,
  while `--source geniza` joins `geniza_image_information` with
  `geniza_image_latents` and samples only pages that fill the configured tile,
  glyph, and word input capacities.
- `discriminability_calibration.py`: computes calibration/discriminability
  statistics for enabled modality subsets.
- `diagnosis/` and `outputs/`: generated reports from previous investigations.

These are DB/checkpoint-oriented diagnostics, not part of the training loop.
