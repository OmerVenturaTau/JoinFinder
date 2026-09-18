# utilities

Shared parsing, extraction, augmentation, and checkpoint helper code.

Subfolders:

- `VisionModule/`: ALTO parsing plus XML-guided tile and glyph extraction.
- `ContextModule/`: ALTO word extraction and AlephBERT text encoding.
- `CharacterModule/`: OCR character/statistical feature helpers.
- `HebrewDict/`: local Hebrew wordset checker and bundled wordset JSON files.
- `augmentations/`: manuscript-specific image augmentations used by training and
  diagnostics.

Top-level helpers:

- `xml_loader.py`: fallback XML path resolution.
- `page_rotation.py`: page rotation detection/correction.
- `checkpoint_utils.py`: checkpoint inspection and compatibility loading.

Keep this package importable by active code. Exploratory visualizations belong in
`Debugs/` or `Drafts/`, not here.

Extraction-sensitive modules:

- Tile extraction changes in `VisionModule/xml_patch_extraction.py` affect
  training, projection, and many diagnostics.
- Glyph extraction changes in `VisionModule/xml_character_extraction.py` affect
  the glyph branch, background leakage checks, and checkpoint comparability.
- Word extraction changes in `ContextModule/xml_word_extraction.py` affect only
  runs with `word.use_word = true`.

Role in the page workflow:

- `train/dataset.py` owns orchestration, but this package owns most of the actual
  page parsing and crop geometry.
- `VisionModule` turns full-page image/XML pairs into tile crops and glyph crops.
- `ContextModule` turns XML text into word/line structures for the optional word
  branch.
- `augmentations` defines train-only perturbations used after crops are created.
- `page_rotation.py` can adjust images/coordinates before XML-guided tile
  extraction when rotated pages are detected.
- `checkpoint_utils.py` keeps projection/evaluation compatible with older
  checkpoints whose modality flags or summary-token counts differ from current
  config.

Downstream effect: anything that changes crop coordinates, filtering thresholds,
page rotation, or augmentation probabilities can change both supervised training
and the Geniza latent vectors used for clustering.

After editing extraction code, run the relevant tests through `NN` and inspect a
debug visualization if DB/image access is available.
