# results_analysis

Analysis tools for Geniza nearest-neighbor results and fixed test-set evaluation.

This folder contains post-training analysis. Most tools read DB tables or fixed
CSV/XLSX files produced after projection and KNN search; the standalone
`interpretability/probe/` workflow also projects selected manuscripts from an
explicit finished checkpoint. Nothing here trains the base neural model.

Where this sits in the page pipeline:

1. A page image is converted to a latent vector by the training/projection path.
2. `aftertune/geniza_top_neighbors_gpu.py` turns those vectors into neighbor rows.
3. This folder turns neighbor rows into human-readable evaluation tables, coverage
   reports, image-level graphs, manuscript-level graphs, and fixed test-set pair
   diagnostics.

The clustering signal here is not computed from raw pixels. It is computed from
model latents that already include the effects of XML tile selection, glyph
filtering, masking, model fusion, and PCA/search-vector choices.

Subfolders:

- `clusters/`: manuscript/image graph builders and interactive HTML exporters.
- `interpretability/probe/`: standalone tile/glyph/word/shared projection plus
  manuscript-disjoint dating and oriental-status probes.
- `test_set/`: fixed pair metadata, pair-score comparison, and DB coverage reports.

Most scripts expect the same DB config path as the rest of the project
(`system.CLUSTERING_DB_CONFIG_PATH`, default `db_config.ini`) unless overridden on
the command line.

Quick examples:

```bash
python results_analysis/knn_same_manuscript_shelfmark_stats.py

python -m results_analysis.clusters.geniza_manuscript_graph \
  --min_similarity 0.7 \
  --min_ms_edge_weight 5 \
  --interactive_ms_html manuscript_graph.html

python results_analysis/test_set/report_db_coverage_stats.py
```

`knn_same_manuscript_shelfmark_stats.py` reads
`geniza_knn_results_including_intermanuscripts` and reports query-level success@K
(at least one same-manuscript or same-shelfmark result) and suggestion-level match
rates for top 1, 5, and 10 by default.

Generated `.html`, `.png`, and metric CSV/XLSX files in this tree are analysis
snapshots. Keep reusable graph logic in Python modules, not embedded in generated
HTML.
