# Debugs/BackgroundLeakage

Scripts for checking whether model inputs contain library/background leakage.

- `inspect_geniza_tiles_by_library_background.py`: compares selected Geniza tiles
  with library background templates.
- `inspect_geniza_glyphs_by_library_background.py`: performs the same type of
  inspection for glyph crops.

Use these when tuning tile/glyph extraction thresholds or background augmentation.
They require DB access and local image/XML paths.
