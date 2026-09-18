# results_analysis/test_set

Fixed test-set metadata and evaluation scripts for Geniza cluster pairs.

- `clusters_images_metadata.csv`: canonical 242-image, 85-cluster retrieval test.
- `cluster_members.xlsx`: canonical 234-image, 86-cluster retrieval test from
  sheet `dj_v1.1.1_cluster_members`; training resolves its image/XML paths in
  memory and does not add similarity-score columns to the workbook.
- `clusters_images_metadata_pairs*.xlsx`: pair lists and variants.
- `compare_clusters_pairs.py`: evaluates pair scores/plots.
- `report_db_coverage_stats.py`: summarizes DB coverage and top-K neighbor stats.

Use this folder when you need repeatable analysis independent of random DB sampling.
During training, both canonical member collections run after every validation
epoch and log only fusion/tile/glyph/word mAP.
