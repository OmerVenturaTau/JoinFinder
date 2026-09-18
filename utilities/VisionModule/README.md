# utilities/VisionModule

ALTO/XML parsing and visual extraction utilities.

- `alto_parser.py`: common ALTO structures for text lines, strings, and glyphs.
- `xml_patch_extraction.py`: extracts text-focused image tiles from ALTO regions,
  including reading-order and ink-density ranking.
- `xml_character_extraction.py`: extracts, filters, balances, tightens, and resizes
  glyph crops.
- `extract_patch_coordinates.py`: helper for patch coordinate extraction workflows.

Page workflow responsibilities:

- Tile extraction starts with a full RGB page and ALTO text regions. It proposes
  tile centers around text, applies reading-order logic, filters/ranks candidates
  by text coverage and ink density, and returns PIL tile crops plus normalized
  coordinates and page segments.
- Glyph extraction starts with the same page/XML pair. It crops individual glyphs,
  can tighten boxes to ink, stores normalized `(x, y, w, h)` metadata, and assigns
  character class IDs used by the glyph branch.
- These functions should return geometry aligned with the image actually cropped.
  If rotation correction is applied, image and coordinates must stay in the same
  rotated frame.

Masking is not done here. This folder returns variable-length per-page lists.
`train/tile_collate_with_padding` pads them and builds the masks consumed by the
model.

These functions are used by `train/dataset.py`, debug visualizations, and Geniza
projection. Changes here can alter both training data and exported latent vectors,
so run extraction/model tests after edits.
