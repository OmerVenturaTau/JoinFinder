# models

Active neural network modules for JoinsFinder.

Current model flow:

1. `tile_branch.py` extracts tile embeddings with the configured tile backbone and
   tile-set transformer, then summarizes them with 8 learned queries.
2. `glyph_branch.py` filters, encodes, positions, and summarizes OCR glyph tokens
   with 8 learned queries.
3. `word_branch.py` encodes OCR words/lines with AlephBERT and summarizes the line
   set with 24 learned queries.
4. `symmetric_fusion.py` is the current default fusion module;
   it optionally restores learned branch adapters together with the branch
   summarizers. `transformer_fusion.py` remains available as an alternative.
5. `vlad_fusion.py` provides a NetVLAD-style aggregation option for local
   descriptors.
6. `perceiver_fusion.py` still provides Perceiver fusion and the shared head
   implementation used by the multimodal model.
7. `multimodal_model.py` wires branches, modality masks/dropout, fusion, and
   classification/latent outputs.

Page representation flow:

```text
Padded batch from train/dataset.py
  tiles + tile_coords + tile_valid_mask
  glyph_patches + glyph_coords + glyph_valid_mask + char_class_ids
  optional words + word metadata
        |
        v
MultiModal.forward
  training only: modality dropout
  training only: token subsampling by editing valid masks/lists
        |
        +--> TileBranch
        |      encode only valid tiles in chunks
        |      zero padded positions
        |      add optional position features / tile-set transformer
        |      8 queries attend to valid tiles -> [B, 8, 768]
        |
        +--> GlyphBranch
        |      encode valid glyph crops in chunks
        |      add coordinate/class position information
        |      8 queries attend to valid glyphs -> [B, 8, 768]
        |
        +--> WordBranch
               24 queries attend to valid line tokens -> [B, 24, 768]
        |
        v
SymmetricRetrievalFusion, TransformerFusion, PerceiverFusion, or VLADFusion
  active symmetric path: learned queries attend only to valid source tokens;
                  all query outputs remain active for nonempty modalities
                  -> mask-mean -> project each branch from 768-d to 512-d
                  -> reliability-weight -> concatenate
        |
        v
SymmetricRetrievalHead for the active symmetric path; PerceiverHead otherwise
  logits for training
  Retrieval latent for clustering/projection. Symmetric fusion exports
  three adapted modality blocks (1536-d for tile + glyph + word)
  and the aftertune PCA pipeline derives the
  existing 256-d ANN search vector.
```

Mask contract:

- Branches must respect `*_valid_mask` and must not treat padded/dummy tokens as
  content.
- Tile and glyph branches may receive a batch with one dummy token and an all-false
  mask. This represents "no extracted tokens", not a real blank page.
- Fusion receives modality token tensors plus modality masks. A modality with no
  valid tokens should not influence the fused page representation.
- Learned summary queries are set aggregators rather than source-token slots.
  Invalid source tokens are masked in cross-attention; query outputs are not
  truncated according to the number of source tokens.
- The pre-summary source mask also supplies a separate evidence fraction to the
  symmetric reliability scorer. It informs the modality weight without masking
  individual summary vectors.

Training vs clustering:

- `forward(...)` returns logits, latent, and optional auxiliary branch latents for
  training.
- `forward_features(...)` is the projection/clustering path; it returns the same
  page latent without using classification logits as the final product.
- `forward_with_attention(...)` is diagnostic-only and intentionally avoids
  dropout/subsampling for stable visualizations. Tile and glyph overlays compose
  the final fusion contribution through their respective query→input maps.

`models/__init__.py` exports only the active branch-based architecture. Historical
model names were removed from this package; do not restore old imports unless the
actual source files are intentionally migrated back.

Important interfaces:

- `MultiModal.forward(...)` is called by `train/trainer.py`, projection scripts,
  and tests. Keep backward-compatible return keys unless all consumers are updated.
- Tile/glyph/word branches receive already-extracted tensors plus masks from
  `train/dataset.py`; they should not open image or XML files.
- Fusion modules operate on modality tokens and masks only. Keep modality-specific
  preprocessing in the branch modules or dataset utilities.

Current defaults come from `config/model_architecture.json`, not hardcoded values
inside this folder. After architecture edits, run:

```bash
conda run -n NN pytest tests/test_multimodal_model.py tests/test_tile_branch.py tests/test_glyph_branch.py
```
