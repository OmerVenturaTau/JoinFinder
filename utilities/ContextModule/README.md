# utilities/ContextModule

Word/text extraction and encoding utilities.

- `alto_text_extractor.py`: extracts textual content from ALTO XML.
- `xml_word_extraction.py`: extracts word boxes, confidences, line IDs, and page
  positions used by the word branch.
- `text_encoder.py`: AlephBERT-based text encoder utilities.

The word modality is implemented but disabled by default in
`config/model_architecture.json`. Keep APIs stable because tests still cover word
branch behavior.

Role in the page workflow:

- Words are extracted from ALTO XML, not from the image tensor.
- `xml_word_extraction.py` filters by OCR confidence and optional Hebrew dictionary
  checks, keeps box/line/page metadata, and caps the word count.
- `models.word_branch` consumes the extracted word lists and metadata, groups or
  encodes text at line/word level, and returns word tokens plus a valid mask.
- Because the word branch is disabled by default, changes here normally affect
  only explicit word-modality experiments and tests, not the current default
  Geniza clustering path.
