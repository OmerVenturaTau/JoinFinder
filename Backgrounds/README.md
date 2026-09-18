# Backgrounds

Library/background template images used by random background overlay augmentation
and background-leakage diagnostics.

The PNG filenames are normalized library/source names. `background_appearances.xlsx`
tracks observed background/source appearances. Training uses these templates through
`utilities/augmentations/background_overlays.py`, with probabilities configured in
`config/model_regularization.json`.

Do not treat these as manuscript inputs. They are augmentation/support artifacts
intended to make the model less dependent on collection-specific backgrounds.

In the page workflow, these files are sampled only by train-time background
overlay augmentation for tile/glyph crops. They should not be used during eval or
Geniza projection, because projection needs deterministic page latents for stable
nearest-neighbor clustering.
