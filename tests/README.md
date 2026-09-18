# tests

Unit and regression tests for active code paths.

Coverage is focused on:

- multimodal model and fusion shapes,
- tile/glyph/word branches,
- coordinate and ALTO extraction behavior,
- sampling and metric-learning utilities,
- combined loss and auxiliary losses,
- background overlay behavior.

Run the suite with:

```bash
conda run -n NN pytest
```

Some tests instantiate model components but should not require DB access.

Use focused tests while editing:

```bash
conda run -n NN pytest tests/test_multimodal_model.py
conda run -n NN pytest tests/test_dataset_coordinates.py tests/test_pipeline_regressions.py
conda run -n NN pytest tests/test_combined_loss.py tests/test_combined_loss_aux.py
```

Workflow-sensitive tests:

- Dataset/coordinate tests protect page-to-token geometry and masking behavior.
- Branch tests protect tile/glyph/word tensor contracts.
- Multimodal/fusion tests protect how masks and modality tokens become page
  latents.
- Loss tests protect the supervision that shapes latents before clustering.
