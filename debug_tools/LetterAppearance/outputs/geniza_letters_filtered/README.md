# Letter counts and OCR confidence

Source: **geniza**. Random-sampling seed: **8472917007555711145**.
Analyzed **500** pages containing recognized Hebrew glyphs from **264** manuscripts and **148,898** Hebrew glyphs.
Mean glyphs per analyzed page: **297.8**. Mean count across letter classes: **5,514.7**.
Applied the glyph-confidence filter **GC >= 0.92**.
Final forms were kept as separate letters.

## Target letters

| letter | count | glyph probability | pages present | mean GC | median GC | frequency rank | GC rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| א | 13828 | 9.287% | 90.000% | 0.993 | 1.000 | 3 | 25 |
| מ | 7621 | 5.118% | 85.000% | 0.993 | 1.000 | 8 | 18 |

## Interpretation notes

- Counts describe OCR output, not ground-truth linguistic frequency: OCR substitutions can change both counts and confidence.
- DB sampling caps each manuscript at the requested pages-per-manuscript value and draws replacement manuscripts until every page slot is filled. Glyph-level probability still gives more weight to text-heavy pages; use `page_presence_probability` as its page-level companion.

## Files

- `letter_summary.csv`: counts, probabilities, and raw-confidence summaries.
- `letter_counts.png`: confidence-filtered recognized-letter counts.
- `confidence_by_letter.png`: confidence quantile intervals for the confidence-filtered glyphs, displayed from their global mean minus three SD to their maximum.
- `confidence_boxplot_by_letter.png`: secondary box-plot view on the same scale.
- `glyph_measurements.csv`: auditable raw measurements used by the summaries.
- `sampled_pages.csv`: the final complete page sample with capped per-manuscript contribution.
- `rejected_candidates.csv`: replacement-pool manuscripts that were incomplete, unusable, or not needed.
