# NeoMME Geniza inference baseline

`evaluate_neomme_geniza.py` evaluates the frozen
`Hcompany/NeoMME-800M-Retriever` dense representation on the canonical Geniza
test set. NeoMME-Retriever does not permit image and text in the same encoded
conversation, so every page receives two inference passes: the image masked to
the ALTO `TextBlock` fragment polygon (plus a 3% margin), and raw OCR text in ALTO
XML order. No manuscript ID, cluster ID, file name, shelfmark, page number, or
other metadata is passed to either input. The report includes image-only,
OCR-only, and equal-weight late-fusion retrieval metrics. The fused vector is a
concatenation of normalized image and OCR vectors, so its cosine similarity is
exactly the mean of the two modality-specific cosine similarities.

The required NeoMME support is installed in:

```text
/home/omerv/anaconda3/envs/DeepPy313
```

Run on one exposed GPU; batch size is intentionally one because page token
counts vary with dynamic image resolution:

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/omerv/anaconda3/envs/DeepPy313/bin/python -u \
  Debugs/NeoMME/evaluate_neomme_geniza.py
```

The script saves resumable embeddings and `metrics.json` under
`Debugs/NeoMME/outputs/neomme_800m_ocr_crop/`. Use `--resume` after an
interrupted run. For a one-page model and memory check, pass `--limit 1` and a
separate `--output-dir`.
