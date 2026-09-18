# Database Tables

This folder contains metadata tables used by JoinFinder. The attached
spreadsheets describe the training/evaluation files and include manuscript-level
metadata such as manuscript IDs, shelfmarks, and dating information used by the
probing experiments.

- `pretrain_finetune_oriental_non_oriental_train_val_test_split.xlsx`: metadata
  for the training/validation/test split.
- `geniza_manuscript_shelfmark.xlsx`: manuscript ID to shelfmark metadata.
- `sfar_data_images_dating.xlsx`: dating metadata used for dating probes.

The image datasets themselves are not included here. Many manuscript images are
copyrighted or otherwise controlled by their holding institutions, so they cannot
be uploaded with this repository. Some manuscript IDs can still be searched and
viewed through public manuscript platforms, including
[Transcriptus](https://transcriptus.org/), depending on availability and access
rights.

The full KNN dataset is also too large to upload. Results from the test run are
available under `results_analysis/test_set/`.
