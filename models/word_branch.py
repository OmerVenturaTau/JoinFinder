"""
Word Branch Architecture - Simplified Implementation

This module implements the word branch according to the architecture:
1. Group ALTO words by line
2. Filter low-quality words (confidence, junk)
3. Build line text by joining remaining words (RTL sort if possible)
4. Run AlephBERT on line → get subword vectors
5. Map subwords back to ALTO words (each word = 1+ subwords)
6. Create one embedding per ALTO word by pooling its subword vectors (mean or first-subword)
7. Optionally apply Bible-corpus TF-IDF gating to those ALTO-word embeddings
8. Pool words → line token (attention pooling), then fuse line tokens directly
"""

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Mapping, Any
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from system import (
    D_MODEL,
    XML_PATCH_READING_DIRECTION_RTL,
    OCR_STRING_CONFIDENCE_THRESHOLD,
    DROPOUT,
    USE_HEBREW_DICT_CHECK,
    WORD_BRANCH_LINE_MAX_LENGTH,
    WORD_BRANCH_WORD_POOLING,
    WORD_BRANCH_ATTENTION_HEADS,
    WORD_BRANCH_ENABLE_POS_ENCODING,
    WORD_BRANCH_USE_FOURIER_POS,
    WORD_BRANCH_NUM_FREQS,
    WORD_BRANCH_MIN_AREA,
    WORD_BRANCH_USE_TFIDF_GATING,
    WORD_BRANCH_TFIDF_DICTIONARY_PATH,
    WORD_NUM_SUMMARY_TOKENS,
    WORD_MIN_SUMMARY_TOKENS,
    USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
)
from utilities.ContextModule.text_encoder import (
    FAST_TOKENIZER_CALL_LOCK,
    HebrewTextEncoder,
    TextEncoderConfig,
)
from utilities.VisionModule.alto_parser import HebrewDictChecker
from utilities.HebrewDict.unified_word_check import (
    PREFIXES,
    TWO_PREFIXES,
    normalize_hebrew_consonants,
    normalize_query,
)


def _metadata_float(meta: Dict, keys: Tuple[str, ...], default: float = 0.0) -> float:
    """Read the first present numeric metadata value without treating 0 as missing."""
    for key in keys:
        if key in meta and meta[key] is not None:
            try:
                value = float(meta[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return float(default)


def _metadata_int(meta: Dict, keys: Tuple[str, ...], default: int = 0) -> int:
    for key in keys:
        if key in meta and meta[key] is not None:
            try:
                value = float(meta[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return int(value)
    return int(default)


def _coerce_page_segment(value) -> int:
    try:
        segment_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WordBranch page_segments entries must be 0 or 1, got {value!r}.") from exc
    if not math.isfinite(segment_float) or segment_float not in (0.0, 1.0):
        raise ValueError(f"WordBranch page_segments entries must be 0 or 1, got {value!r}.")
    return int(segment_float)


def _line_cluster_id(center_y: float) -> str:
    # Normalized ALTO coordinates are clustered in page-relative bands. Raw
    # pixel coordinates need a coarser tolerance so small OCR jitter in vpos
    # does not split one physical text line into many singleton lines.
    if -2.0 <= center_y <= 2.0:
        return f"y_cluster_{int(round(center_y * 20))}"
    return f"y_cluster_px_{int(center_y / 40.0)}"


def _validate_word_batch_inputs(
    words: List[List[str]],
    word_metadata: Optional[List[List[Dict]]],
    page_segments: Optional[List[List[int]]] = None,
) -> None:
    if word_metadata is None:
        raise ValueError("WordBranch requires word_metadata when words are provided.")

    if len(word_metadata) != len(words):
        raise ValueError(
            f"WordBranch expected word_metadata batch size {len(words)}, got {len(word_metadata)}."
        )

    if page_segments is not None and len(page_segments) != len(words):
        raise ValueError(
            f"WordBranch expected page_segments batch size {len(words)}, got {len(page_segments)}."
        )

    for b, (img_words, img_metadata) in enumerate(zip(words, word_metadata)):
        if len(img_metadata) != len(img_words):
            raise ValueError(
                f"WordBranch words/metadata length mismatch at batch index {b}: "
                f"{len(img_words)} words vs {len(img_metadata)} metadata entries."
            )
        for j, meta in enumerate(img_metadata):
            if not isinstance(meta, dict):
                raise ValueError(
                    f"WordBranch metadata entries must be dicts; got {type(meta).__name__} "
                    f"at batch index {b}, word index {j}."
                )
        if page_segments is not None and len(page_segments[b]) != len(img_words):
            raise ValueError(
                f"WordBranch words/page_segments length mismatch at batch index {b}: "
                f"{len(img_words)} words vs {len(page_segments[b])} page segments."
            )
        if page_segments is not None:
            for segment in page_segments[b]:
                _coerce_page_segment(segment)


def group_words_by_lines(
    words: List[str],
    word_metadata: List[Dict],
    rtl: bool = XML_PATCH_READING_DIRECTION_RTL,
) -> Dict[str, List[Tuple[int, str, Dict]]]:
    """
    Group words by line_id (best) or fallback to y-cluster into lines.
    
    Args:
        words: List of word strings
        word_metadata: List of metadata dicts with 'line_id', 'center_y', etc.
        rtl: If True, sort words right-to-left within each line
        
    Returns:
        Dict mapping line_id (or cluster_id) to list of (word_idx, word, metadata) tuples
    """
    line_groups: Dict[str, List[Tuple[int, str, Dict]]] = defaultdict(list)
    
    for idx, (word, meta) in enumerate(zip(words, word_metadata)):
        # Try to use line_id from metadata (best case)
        line_id = meta.get('line_id')
        if line_id is not None:
            line_id = str(line_id)
        else:
            # Fallback: use y-cluster (simple binning by y-coordinate)
            center_y = _metadata_float(
                meta,
                ('normalized_center_y', 'center_y', 'vpos'),
                default=0.0,
            )
            line_id = _line_cluster_id(center_y)
        
        line_groups[line_id].append((idx, word, meta))
    
    # Sort words within each line by x-coordinate (RTL: right to left, LTR: left to right)
    for line_id in line_groups:
        words_in_line = line_groups[line_id]
        if rtl:
            # RTL: sort by x descending (rightmost first)
            words_in_line.sort(key=lambda t: -_metadata_float(
                t[2],
                ('normalized_center_x', 'center_x', 'hpos'),
                default=0.0,
            ))
        else:
            # LTR: sort by x ascending (leftmost first)
            words_in_line.sort(key=lambda t: _metadata_float(
                t[2],
                ('normalized_center_x', 'center_x', 'hpos'),
                default=0.0,
            ))
        line_groups[line_id] = words_in_line
    
    return line_groups


class WordHardQualityFilter:
    """
    Hard quality filter for words within each line.
    
    Hard-drops only obvious garbage:
    - Low confidence words (conf < conf_min)
    - Punctuation-only words
    - Tiny bbox (area < min_area)
    """
    def __init__(
        self,
        conf_min: float = OCR_STRING_CONFIDENCE_THRESHOLD,
        min_area: float = WORD_BRANCH_MIN_AREA,
    ):
        self.conf_min = conf_min
        self.min_area = min_area
    
    def is_punctuation_only(self, word: str) -> bool:
        """Check if word contains only punctuation."""
        if word is None:
            return True
        word = str(word)
        return bool(re.match(r'^[\W_]+$', word.strip()))
    
    def filter_words_in_line(
        self,
        line_words: List[Tuple[int, str, Dict]],
    ) -> List[Tuple[int, str, Dict]]:
        """
        Filter words in a line based on quality criteria.
        
        Args:
            line_words: List of (word_idx, word, metadata) tuples
            
        Returns:
            filtered_words: List of (word_idx, word, metadata) tuples
        """
        filtered = []
        
        for word_idx, word, meta in line_words:
            word_text = "" if word is None else str(word)

            # Check empty/junk before punctuation, since OCR glitches can
            # surface as None or non-string values in defensive callers.
            if not word_text or not word_text.strip():
                continue

            # Check confidence
            conf = _metadata_float(meta, ('wc', 'confidence'), default=0.0)
            if conf < self.conf_min:
                continue
            
            # Check punctuation-only
            if self.is_punctuation_only(word_text):
                continue
            
            # Check bbox sanity (tiny bbox)
            width = _metadata_float(meta, ('width', 'normalized_width'), default=0.0)
            height = _metadata_float(meta, ('height', 'normalized_height'), default=0.0)
            area = width * height
            if area < self.min_area:
                continue
            
            filtered.append((word_idx, word_text, meta))
        
        return filtered


def determine_line_reading_order(
    line_groups: Dict[str, List[Tuple[int, str, Dict]]],
    word_metadata: List[Dict],
    rtl: bool = XML_PATCH_READING_DIRECTION_RTL,
) -> List[str]:
    """
    Determine reading order for lines.
    
    Uses page_segment and y-coordinate to sort lines:
    - First by page_segment (0 then 1)
    - Then by y-coordinate (top to bottom)
    
    Args:
        line_groups: Dict mapping line_id to list of (word_idx, word, metadata) tuples
        word_metadata: List of all word metadata (for page_segment lookup)
        rtl: If True, reading order is RTL
        
    Returns:
        List of line_ids in reading order
    """
    line_info = []
    
    for line_id, line_words in line_groups.items():
        if not line_words:
            continue
        
        page_values = [
            _metadata_int(meta, ('page_segment',), default=0)
            for _, _, meta in line_words
        ]
        y_values = [
            _metadata_float(
                meta,
                ('normalized_center_y', 'center_y', 'vpos'),
                default=0.0,
            )
            for _, _, meta in line_words
        ]
        page_segment = int(round(sum(page_values) / max(1, len(page_values))))
        center_y = sum(y_values) / max(1, len(y_values))
        
        line_info.append((line_id, page_segment, center_y))
    
    # Sort: first by page_segment, then by y-coordinate
    line_info.sort(key=lambda x: (x[1], x[2]))  # (page_segment, y)
    
    return [line_id for line_id, _, _ in line_info]


class BibleTfidfDictionary:
    """Read-only lookup over the precomputed BHSA Bible TF/IDF dictionary."""

    def __init__(
        self,
        entries: Mapping[str, Mapping[str, Any]],
        idf_by_word: Mapping[str, float],
        num_documents: int,
    ):
        if num_documents <= 0:
            raise ValueError("Bible TF-IDF dictionary must contain a positive document count.")
        self.entries = entries
        self.idf_by_word = idf_by_word
        self.num_documents = int(num_documents)

    @classmethod
    def from_json(cls, path: Path | str) -> "BibleTfidfDictionary":
        dictionary_path = Path(path)
        if not dictionary_path.is_file():
            raise FileNotFoundError(f"Bible TF-IDF dictionary not found: {dictionary_path}")

        with dictionary_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        entries = data.get("dictionary")
        idf_by_word = data.get("idf")
        stats = data.get("stats", {})
        if not isinstance(entries, dict) or not isinstance(idf_by_word, dict):
            raise ValueError(
                f"Bible TF-IDF dictionary has an invalid schema: {dictionary_path}"
            )
        if stats.get("idf_kind") != "books":
            raise ValueError(
                "Bible TF-IDF dictionary must use books as IDF documents; "
                f"got {stats.get('idf_kind')!r}."
            )
        num_documents = int(stats.get("idf_num_docs", 0) or 0)
        return cls(entries, idf_by_word, num_documents)

    def _lookup_candidates(self, word: str) -> List[str]:
        normalized = normalize_query(str(word))
        if not normalized:
            return []

        consonants = normalize_hebrew_consonants(normalized)
        candidates = [normalized]
        if consonants != normalized:
            candidates.append(consonants)

        # Prefer a real surface-form statistic. Only fall back to stripping a
        # common BHSA clitic if neither exact normalized form is present.
        if any(candidate in self.idf_by_word for candidate in candidates):
            return candidates

        for prefix in (*TWO_PREFIXES, *PREFIXES):
            if consonants.startswith(prefix) and len(consonants) > len(prefix) + 1:
                candidate = consonants[len(prefix):]
                if candidate not in candidates:
                    candidates.append(candidate)
        return candidates

    def lookup_idf(self, word: str) -> Optional[float]:
        for candidate in self._lookup_candidates(word):
            value = self.idf_by_word.get(candidate)
            if value is not None:
                value = float(value)
                if math.isfinite(value) and value > 0.0:
                    return value
        return None


@lru_cache(maxsize=4)
def load_bible_tfidf_dictionary(path: Path | str) -> BibleTfidfDictionary:
    """Load and cache a Bible TF-IDF dictionary by its resolved path."""
    return BibleTfidfDictionary.from_json(Path(path).resolve())


def compute_tfidf_weights(
    words: List[str],
    bible_tfidf: BibleTfidfDictionary,
) -> torch.Tensor:
    """
    Compute deterministic line-local TF x Bible-corpus IDF weights.
    
    Args:
        words: Words in the current OCR line.
        bible_tfidf: Precomputed BHSA IDF lookup, where Bible books are documents.
        
    Returns:
        tfidf_weights: [len(words)] tensor of TF-IDF weights
    """
    if not words:
        return torch.tensor([], dtype=torch.float32)
    
    normalized_words = [normalize_hebrew_consonants(normalize_query(str(word))) for word in words]

    # TF is local to the current line. IDF is global and fixed, so the same line
    # receives the same weights regardless of batch composition.
    word_counts = Counter(normalized_words)
    max_count = max(word_counts.values()) if word_counts else 1.0
    tf_weights = torch.tensor(
        [word_counts.get(word, 0) / max_count for word in normalized_words],
        dtype=torch.float32,
    )

    # A missing token is usually OCR noise. Give it the neutral/minimum smooth
    # IDF of 1.0 rather than treating it as maximally rare and amplifying it.
    idf_weights = torch.tensor(
        [bible_tfidf.lookup_idf(word) or 1.0 for word in words],
        dtype=torch.float32,
    )
    tfidf = tf_weights * idf_weights
    
    # Normalize to [0, 1]
    if tfidf.max() > 0:
        tfidf = tfidf / tfidf.max()
    
    return tfidf


def compute_legacy_batch_tfidf_weights(
    words: List[str],
    all_words_in_batch: Optional[List[str]] = None,
) -> torch.Tensor:
    """Historical 2026-08 TF-IDF gate retained for checkpoint evaluation."""
    if not words:
        return torch.tensor([], dtype=torch.float32)
    word_counts = Counter(words)
    max_count = max(word_counts.values()) if word_counts else 1.0
    tf_weights = torch.tensor(
        [word_counts.get(word, 0) / max_count for word in words],
        dtype=torch.float32,
    )
    if all_words_in_batch:
        doc_freq = Counter(all_words_in_batch)
        num_docs = len(all_words_in_batch)
        idf_weights = torch.tensor([
            math.log((num_docs + 1) / (doc_freq.get(word, 0) + 1)) + 1.0
            for word in words
        ], dtype=torch.float32)
        tfidf = tf_weights * idf_weights
    else:
        tfidf = tf_weights
    if tfidf.numel() and tfidf.max() > 0:
        tfidf = tfidf / tfidf.max()
    return tfidf


class AlephBERTLineEncoder(nn.Module):
    """
    AlephBERT encoder that processes one line at a time.
    
    Builds line text from words (RTL sorted if possible), runs AlephBERT once per line,
    maps subwords back to ALTO words, and pools subwords per word.
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        line_max_length: int = WORD_BRANCH_LINE_MAX_LENGTH,
        dropout: float = DROPOUT,
        word_pooling: str = WORD_BRANCH_WORD_POOLING,
    ):
        super().__init__()
        self.d_model = d_model
        self.line_max_length = line_max_length
        self.word_pooling = word_pooling
        
        from system import WORD_BRANCH_FREEZE_ALEPHBERT
        self.encoder = HebrewTextEncoder(
            config=TextEncoderConfig(
                model_name="onlplab/alephbert-base",
                max_length=line_max_length,
                pooling="none",  # We need token-level outputs, not pooled
                dropout=dropout,
                freeze_encoder=WORD_BRANCH_FREEZE_ALEPHBERT,
            )
        )
        
        # Get encoder feature dimension
        word_feat_dim = self.encoder.transformer.config.hidden_size  # Typically 768
        
        # Projection to d_model
        if word_feat_dim != d_model:
            self.proj = nn.Linear(word_feat_dim, d_model)
        else:
            self.proj = nn.Identity()
    
    def encode_line_with_word_mapping(
        self,
        line_words: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Encode a line of words using AlephBERT and map subwords back to words.
        
        Args:
            line_words: List of word strings in the line (already sorted by reading order)
            device: torch.device
            
        Returns:
            Tuple of:
            - word_embeddings: [num_words, d_model] - one embedding per ALTO word
            - word_to_token_map: List mapping word index to token indices in the line
        """
        if not line_words:
            return torch.zeros(0, self.d_model, device=device), []
        
        # Move to device if needed (avoid next(parameters()) — empty module → StopIteration)
        p0 = next(iter(self.encoder.parameters()), None)
        b0 = next(iter(self.encoder.buffers()), None) if p0 is None else None
        enc_dev = p0.device if p0 is not None else (b0.device if b0 is not None else None)
        if enc_dev is None or enc_dev != device:
            self.encoder.to(device)
        
        tokenizer = self.encoder.tokenizer
        model = self.encoder.transformer
        
        # Tokenize with word alignment
        # Use is_split_into_words=True to get word_ids mapping
        with FAST_TOKENIZER_CALL_LOCK:
            # First tokenize to get word_ids
            tokenized = tokenizer(
                line_words,  # Pass as list of words
                is_split_into_words=True,
                padding=False,
                truncation=True,
                max_length=self.line_max_length,
                return_tensors=None,
            )
            # Now tokenize again with return_tensors for model input
            encoded = tokenizer(
                line_words,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=self.line_max_length,
                return_tensors="pt",
            )
        
        # Extract word_ids if available
        word_ids = None
        if isinstance(tokenized, dict) and "word_ids" in tokenized:
            word_ids = tokenized["word_ids"]
        elif hasattr(tokenized, "word_ids"):
            word_ids = tokenized.word_ids()
        
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # If word_ids not available, try to get from encoded
        if word_ids is None:
            # Some tokenizers return word_ids in the encoded dict
            if "word_ids" in encoded:
                word_ids = encoded["word_ids"][0].cpu().tolist() if isinstance(encoded["word_ids"], torch.Tensor) else encoded["word_ids"]
        
        # Encode
        trainable = any(p.requires_grad for p in model.parameters())
        with torch.set_grad_enabled(trainable):
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state  # [1, T, H] - all subword token embeddings
        
        # Map subwords back to ALTO words using word_ids
        word_embeddings = []
        word_to_token_map = []
        
        if word_ids is not None:
            # Use word_ids for alignment
            n_words = len(line_words)
            
            for word_idx in range(n_words):
                # Find all subword tokens that belong to this word
                subword_indices = []
                for tok_idx, wid in enumerate(word_ids):
                    if wid == word_idx:
                        subword_indices.append(tok_idx)
                
                if subword_indices:
                    # Pool subwords per word
                    subword_embs = hidden[0, subword_indices]  # [num_subwords, H]
                    
                    if self.word_pooling == "mean":
                        word_emb = subword_embs.mean(dim=0)  # [H]
                    elif self.word_pooling == "first":
                        word_emb = subword_embs[0]  # [H] - first subword
                    else:
                        word_emb = subword_embs.mean(dim=0)  # Default to mean
                    
                    word_embeddings.append(word_emb)
                    word_to_token_map.append(subword_indices[0])  # Store first token index
                else:
                    # Word has no tokens (shouldn't happen, but safety)
                    word_embeddings.append(hidden[0, 0])  # Use CLS token
                    word_to_token_map.append(0)
        else:
            # Fallback: use a simple heuristic to map words to tokens
            # This is approximate - assumes words are tokenized roughly in order
            # Skip CLS token (index 0)
            tokens_per_word = max(1, (hidden.shape[1] - 1) // len(line_words)) if line_words else 1
            
            for word_idx in range(len(line_words)):
                # Approximate: assume tokens are roughly distributed across words
                token_start = 1 + word_idx * tokens_per_word  # +1 to skip CLS
                token_end = min(token_start + tokens_per_word, hidden.shape[1])
                token_indices = list(range(token_start, token_end))
                
                if token_indices:
                    subword_embs = hidden[0, token_indices]  # [num_subwords, H]
                    if self.word_pooling == "mean":
                        word_emb = subword_embs.mean(dim=0)  # [H]
                    elif self.word_pooling == "first":
                        word_emb = subword_embs[0]  # [H]
                    else:
                        word_emb = subword_embs.mean(dim=0)
                    word_embeddings.append(word_emb)
                    word_to_token_map.append(token_indices[0])
                else:
                    word_embeddings.append(hidden[0, 0])  # Use CLS token
                    word_to_token_map.append(0)
        
        if word_embeddings:
            word_embeddings = torch.stack(word_embeddings)  # [num_words, H]
            word_embeddings = self.proj(word_embeddings)  # [num_words, d_model]
        else:
            word_embeddings = torch.zeros(0, self.d_model, device=device)
        
        return word_embeddings, word_to_token_map


class PositionalEncoding4D(nn.Module):
    """
    4D Positional encoding for line bounding boxes.
    
    Maps (x, y, w, h) coordinates to d_model-dimensional embeddings.
    Supports two-page manuscripts.
    
    Note: Coordinates represent actual spatial positions (left=0, right=1).
    For RTL, x is transformed only inside positional encoding so right-side
    positions are encoded as earlier in reading order.
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        use_fourier: bool = WORD_BRANCH_USE_FOURIER_POS,
        num_freqs: int = WORD_BRANCH_NUM_FREQS,
        rtl: bool = None,  # Deprecated: kept for backward compatibility
    ):
        super().__init__()
        self.d_model = d_model
        self.use_fourier = use_fourier
        # Keep an explicit flag so older checkpoints that passed rtl don't break.
        # Default is False (LTR); if rtl was provided, cast to bool.
        self.rtl = bool(rtl) if rtl is not None else False
        
        if use_fourier:
            self.num_freqs = num_freqs
            freqs = torch.linspace(0.0, 1.0, num_freqs)
            self.register_buffer('freqs', freqs)
            fourier_dim = 4 * 2 * num_freqs
            self.fourier_proj = nn.Linear(fourier_dim, d_model)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(4, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode 4D coordinates to d_model-dimensional embeddings.
        
        Note: For RTL, x-coordinate is transformed for positional encoding so that
        large x (right side) is treated as "earlier" in reading order. This only
        affects the positional encoding, not the actual coordinates.
        """
        # For RTL reading order, transform x-coordinate for positional encoding
        # Large x (right side) should be treated as "earlier" in reading order
        coords_encoded = coords.clone()
        if self.rtl:
            coords_encoded[..., 0] = 1.0 - coords_encoded[..., 0]
        
        if self.use_fourier:
            x, y, w, h = coords_encoded[..., 0:1], coords_encoded[..., 1:2], coords_encoded[..., 2:3], coords_encoded[..., 3:4]
            x_sin = torch.sin(2 * math.pi * self.freqs * x)
            x_cos = torch.cos(2 * math.pi * self.freqs * x)
            y_sin = torch.sin(2 * math.pi * self.freqs * y)
            y_cos = torch.cos(2 * math.pi * self.freqs * y)
            w_sin = torch.sin(2 * math.pi * self.freqs * w)
            w_cos = torch.cos(2 * math.pi * self.freqs * w)
            h_sin = torch.sin(2 * math.pi * self.freqs * h)
            h_cos = torch.cos(2 * math.pi * self.freqs * h)
            fourier_feat = torch.cat([x_sin, x_cos, y_sin, y_cos, w_sin, w_cos, h_sin, h_cos], dim=-1)
            pos_emb = self.fourier_proj(fourier_feat)
        else:
            pos_emb = self.mlp(coords_encoded)
        return pos_emb


class WordSetSummarizer(nn.Module):
    """
    Summarize a variable number of line tokens into a fixed number of
    word/line summary tokens via cross-attention (mirrors GlyphSetSummarizer).

    Input:
      - line_tokens: [B, L, d_model]
      - line_valid_mask: [B, L] bool (True=valid, False=padding)
    Output:
      - summary_tokens: [B, WORD_NUM_SUMMARY_TOKENS, d_model]
      - summary_valid_mask: [B, WORD_NUM_SUMMARY_TOKENS] bool
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_queries: int = WORD_NUM_SUMMARY_TOKENS,
        num_heads: int = WORD_BRANCH_ATTENTION_HEADS,
        dropout: float = DROPOUT,
        min_active_queries: int = WORD_MIN_SUMMARY_TOKENS,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        # Retained for checkpoint/API compatibility. Query validity is no
        # longer tied to source-token count.
        self.min_active_queries = max(1, min(int(min_active_queries), num_queries))

        # Learnable query tokens that attend to all valid line tokens.
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)

    def _build_summary_valid_mask(
        self,
        line_valid_mask: torch.Tensor,  # [B, L] bool
    ) -> torch.Tensor:
        """Keep every learned query when the word modality has evidence."""
        has_evidence = line_valid_mask.any(dim=1, keepdim=True)
        return has_evidence.expand(-1, self.num_queries).clone()

    def forward(
        self,
        line_tokens: torch.Tensor,  # [B, L, d_model]
        line_valid_mask: torch.Tensor,  # [B, L] bool (True=valid)
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B = line_tokens.shape[0]
        L = line_tokens.shape[1]
        device = line_tokens.device

        # No lines at all: return an all-padding summary representation.
        if L == 0:
            summary_tokens = torch.zeros(B, self.num_queries, self.d_model, device=device, dtype=line_tokens.dtype)
            summary_valid_mask = torch.zeros(B, self.num_queries, dtype=torch.bool, device=device)
            summary_to_line_attn = None
            if return_attention:
                summary_to_line_attn = torch.zeros(B, self.num_queries, 0, device=device, dtype=line_tokens.dtype)
            return summary_tokens, summary_valid_mask, summary_to_line_attn

        any_valid = line_valid_mask.any(dim=1)  # [B]

        summary_tokens = torch.zeros(B, self.num_queries, self.d_model, device=device, dtype=line_tokens.dtype)
        summary_to_line_attn = None
        if return_attention:
            # [B, Q, L]
            summary_to_line_attn = torch.zeros(B, self.num_queries, L, device=device, dtype=line_tokens.dtype)
        if any_valid.any():
            idx = torch.nonzero(any_valid, as_tuple=False).squeeze(1)
            line_tokens_v = line_tokens.index_select(0, idx)         # [Bv, L, d_model]
            line_valid_mask_v = line_valid_mask.index_select(0, idx) # [Bv, L]
            queries_v = self.query_tokens.expand(line_tokens_v.shape[0], -1, -1)  # [Bv, Q, d_model]

            # key_padding_mask: True = ignore
            attn_mask_v = ~line_valid_mask_v
            summary_v, attn_weights = self.cross_attn(
                query=queries_v,
                key=line_tokens_v,
                value=line_tokens_v,
                key_padding_mask=attn_mask_v,
                need_weights=return_attention,
                average_attn_weights=True,
            )
            summary_tokens_v = self.norm(summary_v)
            summary_tokens.index_copy_(0, idx, summary_tokens_v)
            if return_attention and attn_weights is not None:
                summary_to_line_attn.index_copy_(0, idx, attn_weights)

        # Every query summarizes the complete valid line set. Only samples
        # with no valid lines receive an all-False summary mask.
        summary_valid_mask = self._build_summary_valid_mask(line_valid_mask)
        return summary_tokens, summary_valid_mask, summary_to_line_attn


class WordBranch(nn.Module):
    """
    Complete Word Branch: Line Grouping + Quality Filter + AlephBERT Encoding + 
    Subword-to-Word Mapping + Optional Bible TF-IDF Gating + Attention Pooling.
    
    Architecture:
    1. Group ALTO words by line
    2. Filter low-quality words (confidence, junk)
    3. Build line text by joining remaining words (RTL sort if possible)
    4. Run AlephBERT on line → get subword vectors
    5. Map subwords back to ALTO words (each word = 1+ subwords)
    6. Create one embedding per ALTO word by pooling its subword vectors (mean or first-subword)
    7. Optionally apply Bible-corpus TF-IDF gating to those ALTO-word embeddings
    8. Pool words → line token (attention pooling)
    9. Add pos_line (bbox) to the line token
    10. Output line tokens in reading order (ready for fusion)
    
    Supports two-page manuscripts: y-coordinates can be in [0, 2] (second page: y += 1.0)
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        line_max_length: int = WORD_BRANCH_LINE_MAX_LENGTH,
        enable_pos_encoding: bool = WORD_BRANCH_ENABLE_POS_ENCODING,
        use_fourier_pos: bool = WORD_BRANCH_USE_FOURIER_POS,
        num_freqs: int = WORD_BRANCH_NUM_FREQS,
        rtl: bool = XML_PATCH_READING_DIRECTION_RTL,
        dropout: float = DROPOUT,
        word_pooling: str = WORD_BRANCH_WORD_POOLING,
        num_heads: int = WORD_BRANCH_ATTENTION_HEADS,
        use_tfidf_gating: bool = WORD_BRANCH_USE_TFIDF_GATING,
        tfidf_dictionary_path: Path | str = WORD_BRANCH_TFIDF_DICTIONARY_PATH,
        bible_tfidf: Optional[BibleTfidfDictionary] = None,
        legacy_batch_tfidf_gating: bool = False,
        num_summary_tokens: int = WORD_NUM_SUMMARY_TOKENS,
        min_summary_tokens: int = WORD_MIN_SUMMARY_TOKENS,
        enable_summarizer: bool = USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
        # Quality filter parameters
        conf_min: float = OCR_STRING_CONFIDENCE_THRESHOLD,
        min_area: float = WORD_BRANCH_MIN_AREA,
    ):
        super().__init__()
        self.d_model = d_model
        self.rtl = rtl
        self.enable_pos_encoding = bool(enable_pos_encoding)
        self.use_tfidf_gating = bool(use_tfidf_gating)
        self.legacy_batch_tfidf_gating = bool(legacy_batch_tfidf_gating)
        if self.use_tfidf_gating and self.legacy_batch_tfidf_gating:
            raise ValueError("Current and legacy TF-IDF gating cannot both be enabled.")
        self.bible_tfidf = bible_tfidf
        if self.use_tfidf_gating and self.bible_tfidf is None:
            self.bible_tfidf = load_bible_tfidf_dictionary(tfidf_dictionary_path)
        
        # Quality filter (non-trainable)
        self.quality_filter = WordHardQualityFilter(
            conf_min=conf_min,
            min_area=min_area,
        )
        
        # AlephBERT line encoder (with subword-to-word mapping). Freezing only via system.py epoch ranges.
        self.line_encoder = AlephBERTLineEncoder(
            d_model=d_model,
            line_max_length=line_max_length,
            dropout=dropout,
            word_pooling=word_pooling,
        )
        
        # Attention pooling for words → line token
        self.word_to_line_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Learnable query for line-level pooling
        self.line_query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.trunc_normal_(self.line_query, std=0.02)
        
        # Line positional encoding
        # Positional encoding for line coordinates (x, y, w, h)
        # RTL-aware: transforms x for encoding so large x (right) is treated as "earlier"
        self.line_pos_encoder = PositionalEncoding4D(
            d_model=d_model,
            use_fourier=use_fourier_pos,
            num_freqs=num_freqs,
            rtl=rtl,
        )
        
        # Kept for checkpoint compatibility; modality-type embeddings
        # are applied in the fusion module.
        self.type_embed = nn.Embedding(1, d_model)
        
        self.norm = nn.LayerNorm(d_model)

        self.word_set_summarizer = (
            WordSetSummarizer(
                d_model=d_model,
                num_queries=num_summary_tokens,
                num_heads=num_heads,
                dropout=dropout,
                min_active_queries=min_summary_tokens,
            )
            if bool(enable_summarizer) and int(num_summary_tokens) > 0
            else None
        )
    
    def forward(
        self,
        words: List[List[str]],  # [B] list of word string lists
        word_metadata: List[List[Dict]],  # [B] list of metadata dict lists
        device: torch.device,
        page_segments: Optional[List[List[int]]] = None,  # [B] list of page_segment lists
        return_attention: bool = False,
        return_line_to_word_attn: bool = False,
        return_summary_to_line_attn: bool = False,
        return_source_evidence: bool = False,
    ):
        """
        Forward pass through the word branch.
        
        Args:
            words: [B] list of word string lists
            word_metadata: [B] list of metadata dict lists with keys: 'wc', 'center_x', 'center_y', 
                          'normalized_width', 'normalized_height', 'line_id', 'page_segment', etc.
            device: torch.device
            page_segments: [B] list of page_segment lists (0 or 1) for two-page manuscripts
            return_attention: If True, return word-level attention weights
            
        Returns:
            Tuple of:
            - word_tokens: `[B, L_max, d_model]` line tokens when summarization is
              disabled, otherwise fixed summary tokens
            - word_valid_mask: matching bool mask for line or summary tokens
            - word_attention: [B, W_max+1, W_max+1] - word attention matrix with CLS token (if return_attention=True)
            - line_to_word_attn: [B, L_max, W_max] (if return_line_to_word_attn=True)
            - summary_to_line_attn: [B, Q, L_max] (if return_summary_to_line_attn=True)
            - source_evidence_fraction: `[B]`, appended only when
              `return_source_evidence=True`; computed from the valid line count
              used by summarizer attention, normalized by summary-query capacity
        """
        B = len(words)
        _validate_word_batch_inputs(words, word_metadata, page_segments)

        def with_source_evidence(result, evidence_fraction):
            if return_source_evidence:
                return (*result, evidence_fraction)
            return result

        if B == 0 or all(len(w) == 0 for w in words):
            source_evidence_fraction = torch.zeros(B, device=device)
            # Return empty line tokens
            if return_attention:
                base = (
                    torch.zeros(B, 0, self.d_model, device=device),
                    torch.zeros(B, 0, dtype=torch.bool, device=device),
                    None,
                )
                if return_line_to_word_attn or return_summary_to_line_attn:
                    return with_source_evidence(
                        (*base, None, None), source_evidence_fraction
                    )
                return with_source_evidence(base, source_evidence_fraction)
            base = (
                torch.zeros(B, 0, self.d_model, device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )
            return with_source_evidence(base, source_evidence_fraction)
        
        all_line_tokens = []  # Will collect line tokens per image
        all_line_valid_masks = []  # Will collect line valid masks per image
        all_word_attention = [] if return_attention else None  # Will collect word attention per image
        all_line_word_attn = [] if return_line_to_word_attn else None  # [B, L, W]
        all_words_flat = (
            [str(word) for image_words in words for word in image_words if word is not None]
            if self.legacy_batch_tfidf_gating else []
        )
        
        for b in range(B):
            img_words = words[b]
            img_metadata = [dict(meta) for meta in word_metadata[b]]
            img_page_segments = page_segments[b] if page_segments is not None else None
            if img_page_segments is not None:
                for meta, page_segment in zip(img_metadata, img_page_segments):
                    meta['page_segment'] = _coerce_page_segment(page_segment)
            num_words = len(img_words)
            
            if not img_words:
                # Empty image
                all_line_tokens.append(torch.zeros(0, self.d_model, device=device))
                all_line_valid_masks.append(torch.zeros(0, dtype=torch.bool, device=device))
                if return_attention:
                    all_word_attention.append(torch.zeros(num_words + 1, num_words + 1, device=device))
                if return_line_to_word_attn:
                    all_line_word_attn.append(torch.zeros(0, num_words, device=device))
                continue
            
            # 1. Group words by lines
            line_groups = group_words_by_lines(img_words, img_metadata, rtl=self.rtl)
            
            # 2. Filter words in each line (hard-drop only obvious garbage)
            filtered_line_groups = {}
            for line_id, line_words in line_groups.items():
                filtered = self.quality_filter.filter_words_in_line(line_words)
                if filtered:
                    filtered_line_groups[line_id] = filtered
            
            if not filtered_line_groups:
                # No valid words after filtering
                all_line_tokens.append(torch.zeros(0, self.d_model, device=device))
                all_line_valid_masks.append(torch.zeros(0, dtype=torch.bool, device=device))
                if return_attention:
                    all_word_attention.append(torch.zeros(num_words + 1, num_words + 1, device=device))
                if return_line_to_word_attn:
                    all_line_word_attn.append(torch.zeros(0, num_words, device=device))
                continue
            
            # 3. Determine reading order for lines
            line_order = determine_line_reading_order(filtered_line_groups, img_metadata, rtl=self.rtl)
            
            # 4. Process each line in reading order
            line_tokens_list = []
            word_attention_list = [] if return_attention else None  # Collect word attention per line for this image
            line_word_attn_list = [] if return_line_to_word_attn else None  # [num_words] per line
            
            for line_id in line_order:
                if line_id not in filtered_line_groups:
                    continue
                
                filtered_words = filtered_line_groups[line_id]
                line_word_strings = [w for _, w, _ in filtered_words]
                word_indices = [idx for idx, _, _ in filtered_words]
                
                # 5. Encode line with AlephBERT and map subwords to words
                word_embs, _ = self.line_encoder.encode_line_with_word_mapping(
                    line_word_strings, device
                )  # [num_words, d_model]
                
                if word_embs.shape[0] == 0:
                    continue
                
                # 6. Optional deterministic TF-IDF gate: line-local TF and
                # Bible-wide book IDF from the precomputed BHSA dictionary.
                if self.use_tfidf_gating:
                    if self.bible_tfidf is None:
                        raise RuntimeError("Bible TF-IDF gating is enabled without a dictionary.")
                    tfidf_weights = compute_tfidf_weights(line_word_strings, self.bible_tfidf)
                    if len(tfidf_weights) == word_embs.shape[0]:
                        tfidf_weights = tfidf_weights.to(device)
                        word_embs = word_embs * tfidf_weights.unsqueeze(-1)  # [num_words, d_model]
                elif self.legacy_batch_tfidf_gating:
                    tfidf_weights = compute_legacy_batch_tfidf_weights(
                        line_word_strings, all_words_flat
                    )
                    if len(tfidf_weights) == word_embs.shape[0]:
                        word_embs = word_embs * tfidf_weights.to(device).unsqueeze(-1)
                
                # 7. Attention pool words → line token
                word_embs_batch = word_embs.unsqueeze(0)  # [1, num_words, d_model]
                line_query_batch = self.line_query.expand(1, -1, -1)  # [1, 1, d_model]
                
                line_tok, attn_weights = self.word_to_line_pool(
                    query=line_query_batch,  # [1, 1, d_model]
                    key=word_embs_batch,  # [1, num_words, d_model]
                    value=word_embs_batch,  # [1, num_words, d_model]
                    average_attn_weights=False,  # Return per-head attention
                )  # [1, 1, d_model], [1, 1, num_words] or [num_heads, 1, 1, num_words]
                
                # attn_weights layout depends on PyTorch version, but in the common
                # case (batch_first=True) it's either:
                #   [B, num_heads, tgt_len(=1), src_len(=num_words)]
                #   [num_heads, B, tgt_len(=1), src_len(=num_words)]
                # We average over heads, then squeeze to [num_words] so downstream
                # visualization sums correctly per word.
                if attn_weights is not None and attn_weights.dim() == 4:
                    if attn_weights.shape[0] == 1:
                        # [B=1, H, 1, W] -> [1, 1, W] -> [W]
                        attn_weights = attn_weights.mean(dim=1).squeeze(0).squeeze(0)
                    elif attn_weights.shape[1] == 1:
                        # [H, B=1, 1, W] -> [1, 1, W] -> [W]
                        attn_weights = attn_weights.mean(dim=0).squeeze(0).squeeze(0)
                    else:
                        # Fallback: assume [B, H, 1, W] and average heads.
                        attn_weights = attn_weights.mean(dim=1).squeeze(0).squeeze(0)
                elif attn_weights is not None:
                    attn_weights = attn_weights.squeeze(0).squeeze(0)  # [num_words]
                
                # Store word attention with word indices for later aggregation
                if return_attention:
                    word_attention_list.append((word_indices, attn_weights))

                # Also store an explicit line->word attention vector for fusion back-propagation.
                if return_line_to_word_attn:
                    vec = torch.zeros(num_words, device=device, dtype=attn_weights.dtype)
                    for k, word_idx in enumerate(word_indices):
                        if word_idx < num_words and k < attn_weights.numel():
                            vec[word_idx] = attn_weights.reshape(-1)[k]
                    line_word_attn_list.append(vec)
                
                line_tok = line_tok.squeeze(0).squeeze(0)  # [d_model]
                
                # 8. Compute line bbox coordinates and add positional embedding
                line_x_coords = []
                line_y_coords = []
                line_widths = []
                line_heights = []
                
                for word_idx in word_indices:
                    if word_idx < len(img_metadata):
                        meta = img_metadata[word_idx]
                        center_x = _metadata_float(
                            meta,
                            ('normalized_center_x', 'center_x'),
                            default=0.5,
                        )
                        center_y = _metadata_float(
                            meta,
                            ('normalized_center_y', 'center_y'),
                            default=0.5,
                        )
                        width = _metadata_float(
                            meta,
                            ('normalized_width', 'width'),
                            default=0.0,
                        )
                        height = _metadata_float(
                            meta,
                            ('normalized_height', 'height'),
                            default=0.0,
                        )
                        
                        # Apply page_segment shift if provided
                        page_segment = _metadata_int(meta, ('page_segment',), default=0)
                        if page_segment == 1:
                            center_y = center_y + 1.0
                        
                        line_x_coords.append(center_x)
                        line_y_coords.append(center_y)
                        line_widths.append(width)
                        line_heights.append(height)
                
                if line_x_coords:
                    line_x = sum(line_x_coords) / len(line_x_coords)
                    line_y = sum(line_y_coords) / len(line_y_coords)
                    line_w = max(line_x_coords) - min(line_x_coords) + (sum(line_widths) / len(line_widths) if line_widths else 0.0)
                    line_h = sum(line_heights) / len(line_heights) if line_heights else 0.0
                    line_coords = [line_x, line_y, line_w, line_h]
                else:
                    line_coords = [0.5, 0.5, 0.0, 0.0]
                
                # Note: modality-type embeddings are added in the fusion module.
                # The branch optionally adds positional information for the line.
                line_token = line_tok
                if self.enable_pos_encoding:
                    line_coords_tensor = torch.tensor([line_coords], device=device)  # [1, 4]
                    line_pos = self.line_pos_encoder(line_coords_tensor).squeeze(0)  # [d_model]
                    line_token = line_token + line_pos
                line_token = self.norm(line_token)  # [d_model]
                
                line_tokens_list.append(line_token)
            
            if line_tokens_list:
                line_tokens_tensor = torch.stack(line_tokens_list)  # [L, d_model]
                line_valid_mask = torch.ones(len(line_tokens_list), dtype=torch.bool, device=device)
            else:
                line_tokens_tensor = torch.zeros(0, self.d_model, device=device)
                line_valid_mask = torch.zeros(0, dtype=torch.bool, device=device)
            
            all_line_tokens.append(line_tokens_tensor)
            all_line_valid_masks.append(line_valid_mask)
            if return_line_to_word_attn:
                if line_word_attn_list:
                    all_line_word_attn.append(torch.stack(line_word_attn_list))  # [L, W]
                else:
                    all_line_word_attn.append(torch.zeros(0, num_words, device=device))
            
            # Aggregate word attention for this image
            if return_attention and word_attention_list:
                num_words = len(img_words)
                # Create attention matrix [W+1, W+1] where first token is CLS
                word_attn_matrix = torch.zeros(num_words + 1, num_words + 1, device=device)
                
                # CLS token attends to all words (uniform or based on line attention)
                cls_attn = torch.zeros(num_words, device=device)
                for word_indices, line_attn in word_attention_list:
                    la = line_attn.reshape(-1)
                    for i, word_idx in enumerate(word_indices):
                        if word_idx < num_words and i < la.numel():
                            cls_attn[word_idx] += la[i].to(dtype=cls_attn.dtype)
                # Normalize CLS attention
                if cls_attn.sum() > 0:
                    cls_attn = cls_attn / cls_attn.sum()
                word_attn_matrix[0, 1:] = cls_attn  # CLS attends to words
                word_attn_matrix[0, 0] = 0.0  # CLS doesn't attend to itself
                
                # Word-to-word attention: words in same line attend to each other
                # For simplicity, we use identity + line-based attention
                for word_indices, line_attn in word_attention_list:
                    la = line_attn.reshape(-1)
                    for i, word_idx_i in enumerate(word_indices):
                        if word_idx_i >= num_words or i >= la.numel():
                            continue
                        for j, word_idx_j in enumerate(word_indices):
                            if word_idx_j >= num_words or j >= la.numel():
                                continue
                            word_attn_matrix[word_idx_i + 1, word_idx_j + 1] = (
                                la[i] * la[j]
                            ).to(word_attn_matrix.dtype)
                
                # Normalize word-to-word attention rows
                for i in range(1, num_words + 1):
                    row_sum = word_attn_matrix[i, 1:].sum()
                    if row_sum > 0:
                        word_attn_matrix[i, 1:] = word_attn_matrix[i, 1:] / row_sum
                
                all_word_attention.append(word_attn_matrix)
            elif return_attention:
                # No words, create empty attention matrix
                all_word_attention.append(torch.zeros(1, 1, device=device))
        
        # Pad line tokens to the same length across the batch. The active path
        # returns these directly; legacy configurations may summarize them.
        max_lines = max(lt.shape[0] for lt in all_line_tokens) if all_line_tokens else 0
        if max_lines == 0:
            source_evidence_fraction = torch.zeros(B, device=device)
            empty_token_count = (
                int(self.word_set_summarizer.num_queries)
                if self.word_set_summarizer is not None
                else 0
            )
            if return_attention:
                # Pad word attention
                max_words = max(wa.shape[0] - 1 for wa in all_word_attention) if all_word_attention else 0
                if max_words == 0:
                    base = (
                        torch.zeros(B, empty_token_count, self.d_model, device=device),
                        torch.zeros(B, empty_token_count, dtype=torch.bool, device=device),
                        None,
                    )
                    if return_line_to_word_attn or return_summary_to_line_attn:
                        return with_source_evidence(
                            (*base, None, None), source_evidence_fraction
                        )
                    return with_source_evidence(base, source_evidence_fraction)
                padded_word_attn = torch.zeros(B, max_words + 1, max_words + 1, device=device)
                for b in range(B):
                    if b < len(all_word_attention):
                        W = all_word_attention[b].shape[0] - 1
                        padded_word_attn[b, :W+1, :W+1] = all_word_attention[b]
                base = (
                    torch.zeros(B, empty_token_count, self.d_model, device=device),
                    torch.zeros(B, empty_token_count, dtype=torch.bool, device=device),
                    padded_word_attn,
                )
                if return_line_to_word_attn or return_summary_to_line_attn:
                    return with_source_evidence(
                        (*base, None, None), source_evidence_fraction
                    )
                return with_source_evidence(base, source_evidence_fraction)
            return with_source_evidence(
                (
                    torch.zeros(B, empty_token_count, self.d_model, device=device),
                    torch.zeros(B, empty_token_count, dtype=torch.bool, device=device),
                ),
                source_evidence_fraction,
            )
        
        padded_line_tokens = torch.zeros(B, max_lines, self.d_model, device=device)
        padded_line_valid_mask = torch.zeros(B, max_lines, dtype=torch.bool, device=device)
        
        for b, (lt, lvm) in enumerate(zip(all_line_tokens, all_line_valid_masks)):
            L = lt.shape[0]
            if L > 0:
                padded_line_tokens[b, :L] = lt
                padded_line_valid_mask[b, :L] = lvm

        source_capacity = (
            int(self.word_set_summarizer.num_queries)
            if self.word_set_summarizer is not None
            else max_lines
        )
        source_count = padded_line_valid_mask.to(padded_line_tokens.dtype).sum(dim=1)
        source_evidence_fraction = source_count.clamp_max(
            float(source_capacity)
        ) / float(source_capacity)

        padded_line_to_word_attn = None
        if return_line_to_word_attn:
            # Pad explicit line->word attention vectors.
            max_words = 0
            if all_line_word_attn is not None:
                max_words = max(t.shape[1] for t in all_line_word_attn) if all_line_word_attn else 0
            padded_line_to_word_attn = torch.zeros(
                B, max_lines, max_words, device=device,
                dtype=torch.float32,
            )
            if max_words > 0 and all_line_word_attn is not None:
                for b, lw in enumerate(all_line_word_attn):
                    L = lw.shape[0]
                    W = lw.shape[1]
                    if L > 0 and W > 0:
                        padded_line_to_word_attn[b, :L, :W] = lw.to(device=device, dtype=padded_line_to_word_attn.dtype)

        if self.word_set_summarizer is None:
            output_tokens = padded_line_tokens
            output_valid_mask = padded_line_valid_mask
            summary_to_line_attn = None
        else:
            output_tokens, output_valid_mask, summary_to_line_attn = self.word_set_summarizer(
                padded_line_tokens,
                padded_line_valid_mask,
                return_attention=return_summary_to_line_attn,
            )
        
        if return_attention:
            # Pad word attention to same size across batch
            max_words = max(wa.shape[0] - 1 for wa in all_word_attention) if all_word_attention else 0
            if max_words == 0:
                padded_word_attn = None
            else:
                padded_word_attn = torch.zeros(B, max_words + 1, max_words + 1, device=device)
                for b in range(B):
                    if b < len(all_word_attention):
                        W = all_word_attention[b].shape[0] - 1
                        padded_word_attn[b, :W+1, :W+1] = all_word_attention[b]
            if return_line_to_word_attn or return_summary_to_line_attn:
                return with_source_evidence(
                    (
                        output_tokens,
                        output_valid_mask,
                        padded_word_attn,
                        padded_line_to_word_attn,
                        summary_to_line_attn,
                    ),
                    source_evidence_fraction,
                )
            return with_source_evidence(
                (output_tokens, output_valid_mask, padded_word_attn),
                source_evidence_fraction,
            )
        
        return with_source_evidence(
            (output_tokens, output_valid_mask), source_evidence_fraction
        )
