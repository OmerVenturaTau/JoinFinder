# utilities/HebrewDict

Local Hebrew wordset builder and checker.

This folder contains:

- `unified_word_check.py`: CLI/library for building and checking Hebrew wordsets.
- `bhsa_wordset.json`, `bhsa_books_index.json`, `bhsa_normalized_dict.json`:
  Biblical Hebrew data derived from BHSA/Text-Fabric.
- `mh_wordset.json`: Modern Hebrew wordset derived from `wordfreq`.

The active word extraction path can use this checker when
`word.use_hebrew_dict_check` is true in `config/model_architecture.json`.
When `word.use_tfidf_gating` is true, the word branch uses line-local term
frequency with book-level IDF from `bhsa_normalized_dict.json`. This is a soft
embedding gate, separate from the hard dictionary check, and is deterministic
across batch compositions.

Common commands:

```bash
python utilities/HebrewDict/unified_word_check.py check ויאמר
python utilities/HebrewDict/unified_word_check.py check בֵ֣ית --strict
python utilities/HebrewDict/unified_word_check.py build --with-stats
```

Building from scratch requires `text-fabric`, `wordfreq`, and local BHSA data. The
bundled JSON files allow normal checks without rebuilding.
