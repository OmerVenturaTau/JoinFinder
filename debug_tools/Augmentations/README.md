# debug_tools/Augmentations

Diagnostics for augmentation behavior and branch robustness.

- `branch_augmentation_label_flips.py`: runs a checkpoint under controlled
  augmentations and records tile/glyph prediction flips.
- `sample_geniza_fragment_and_visualize_augmentations.py`: samples one Geniza
  fragment and renders tile/glyph augmentation grids.
- `test_background_sampling_probabilities.py`: checks random-library background
  sampling probabilities.
- `outputs/`: generated snapshots from previous runs.

These scripts may require DB access, checkpoint paths, and the `Backgrounds/`
template images.
