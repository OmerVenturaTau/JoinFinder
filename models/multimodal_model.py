"""
Multi-modal Vision Transformer - New Architecture.

Uses the new branch architecture:
- TileBranch: High-res tiles → enriched tile tokens
- GlyphBranch: Glyphs → enriched glyph tokens
- WordBranch: Words → line tokens
- SymmetricRetrievalFusion: masked mean + native-width concatenation
- PerceiverFusion: optional cross-attention fusion of all modalities
- VLADFusion: NetVLAD-style residual aggregation of all modality tokens
- PerceiverHead: Classification head
"""

import inspect
import math
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from system import (
    TILE_SIZE,
    ENCODER_MODEL_NAME,
    D_MODEL,
    CHAR_PATCH_SIZE,
    USE_VISUAL_MOD,
    USE_CHAR_MOD,
    USE_WORD_MOD,
    LATENT_DIM,
    MODALITY_DROPOUT_ENABLED,
    MODALITY_DROPOUT_PROB_VISUAL,
    MODALITY_DROPOUT_PROB_CHAR,
    MODALITY_DROPOUT_PROB_WORD,
    FUSION_METHOD,
    TOKEN_SUBSAMPLE_ENABLED,
    TOKEN_SUBSAMPLE_PROB,
    TOKEN_SUBSAMPLE_MIN_KEEP_TILES,
    TOKEN_SUBSAMPLE_MIN_KEEP_GLYPHS,
    TOKEN_SUBSAMPLE_MIN_KEEP_WORDS,
    TOKEN_SUBSAMPLE_FRAC_RANGE,
    SYMMETRIC_RELIABILITY_HIDDEN_DIM,
    SYMMETRIC_BRANCH_DIM,
    USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
    NUM_GLYPH_CLASSES,
)

from .tile_branch import TileBranch
from .glyph_branch import GlyphBranch
from .word_branch import WordBranch
from .perceiver_fusion import PerceiverFusion, PerceiverHead
from .transformer_fusion import TransformerFusion
from .vlad_fusion import VLADFusion
from .symmetric_fusion import SymmetricRetrievalFusion, SymmetricRetrievalHead


def _slice_list_by_batch_indices(values: Optional[List], idx: List[int], name: str) -> Optional[List]:
    if values is None:
        return None
    if any(i < 0 or i >= len(values) for i in idx):
        raise ValueError(
            f"batch_element_indices contains an index outside {name} length {len(values)}: {idx}."
        )
    return [values[i] for i in idx]


def _apply_batch_element_indices_to_lists(
    batch_element_indices: Optional[torch.Tensor],
    words: Optional[List],
    word_metadata: Optional[List],
    word_page_segments: Optional[List],
    paths: Optional[List],
) -> Tuple[Optional[List], Optional[List], Optional[List], Optional[List]]:
    """
    Align list-shaped inputs with this replica's batch shard.

    DataParallel splits tensors on dim=0 but duplicates plain Python lists on
    every GPU. Pass ``torch.arange(global_B, dtype=torch.long, device=...)`` so
    scatter sends each device its row indices; then slice words/metadata/paths.
    """
    if batch_element_indices is None:
        return words, word_metadata, word_page_segments, paths
    idx = batch_element_indices.detach().cpu().tolist()
    words = _slice_list_by_batch_indices(words, idx, "words")
    word_metadata = _slice_list_by_batch_indices(word_metadata, idx, "word_metadata")
    word_page_segments = _slice_list_by_batch_indices(word_page_segments, idx, "word_page_segments")
    paths = _slice_list_by_batch_indices(paths, idx, "paths")
    return words, word_metadata, word_page_segments, paths


def _parse_word_page_segment(value) -> int:
    try:
        segment_float = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(segment_float) or segment_float not in (0.0, 1.0):
        return int(segment_float) if segment_float.is_integer() else 0
    return int(segment_float)


def _word_page_segments_from_metadata(
    word_metadata: Optional[List[List[Dict]]],
) -> Optional[List[List[int]]]:
    """Build word page-segment lists from metadata when collate does not emit them separately."""
    if word_metadata is None:
        return None
    segments: List[List[int]] = []
    for image_metadata in word_metadata:
        image_segments = []
        for meta in image_metadata or []:
            try:
                image_segments.append(_parse_word_page_segment(meta.get("page_segment", 0)))
            except Exception:
                image_segments.append(0)
        segments.append(image_segments)
    return segments


def _validate_list_batch_size(values: Optional[List], batch_size: int, name: str) -> None:
    if values is not None and len(values) != batch_size:
        raise ValueError(
            f"MultiModal expected {name} batch size {batch_size}, got {len(values)}."
        )


class MultiModal(nn.Module):
    """
    Multi-modal model using branch architecture and configurable fusion.
    
    Architecture:
    1. TileBranch: tiles → contextualized tile tokens [B, Nt, d_model]
    2. GlyphBranch: glyphs → enriched glyph tokens [B, Mg, d_model]
    3. WordBranch: words → line tokens [B, L, d_model]
    4. Fusion (FUSION_METHOD):
       - "perceiver": cross-attention fusion → [B, num_latents, d_model]
       - "transformer": self-attention fusion → [B, 1, d_model]
       - "vlad": NetVLAD-style residual aggregation → [B, 1, d_model]
    5. PerceiverHead: pool → project → classify → [B, num_classes]
    """
    def __init__(
        self,
        num_classes: int,
        tile_size: int = TILE_SIZE,
        encoder_model_name: str = ENCODER_MODEL_NAME,
        d_model: int = D_MODEL,
        encoder_pretrained: bool = True,
        num_glyph_classes: Optional[int] = None,
        symmetric_modalities_from_instance: bool = False,
        use_visual_mod: Optional[bool] = None,
        use_char_mod: Optional[bool] = None,
        use_word_mod: Optional[bool] = None,
        word_legacy_batch_tfidf: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_branch_adapters_and_summarizers = bool(
            USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
        )
        
        # Modality usage flags
        self.use_visual_mod = USE_VISUAL_MOD if use_visual_mod is None else bool(use_visual_mod)
        self.use_char_mod = USE_CHAR_MOD if use_char_mod is None else bool(use_char_mod)
        self.use_word_mod = USE_WORD_MOD if use_word_mod is None else bool(use_word_mod)
        
        # Initialize branches
        if self.use_visual_mod:
            self.tile_branch = TileBranch(
                tile_size=tile_size,
                encoder_model_name=encoder_model_name,
                d_model=d_model,
                pretrained=encoder_pretrained,
                enable_summarizer=self.use_branch_adapters_and_summarizers,
            )
        
        if self.use_char_mod:
            self.glyph_branch = GlyphBranch(
                char_patch_size=CHAR_PATCH_SIZE,
                d_model=d_model,
                pretrained=encoder_pretrained,
                num_char_classes=(
                    NUM_GLYPH_CLASSES
                    if num_glyph_classes is None
                    else int(num_glyph_classes)
                ),
                enable_summarizer=self.use_branch_adapters_and_summarizers,
            )
        
        if self.use_word_mod:
            self.word_branch = WordBranch(
                d_model=d_model,
                legacy_batch_tfidf_gating=word_legacy_batch_tfidf,
                enable_summarizer=self.use_branch_adapters_and_summarizers,
            )
        
        # Fusion Module
        if FUSION_METHOD == "perceiver":
            self.fusion = PerceiverFusion(d_model=d_model)
        elif FUSION_METHOD == "transformer":
            self.fusion = TransformerFusion(d_model=d_model)
        elif FUSION_METHOD == "vlad":
            self.fusion = VLADFusion(d_model=d_model)
        elif FUSION_METHOD == "symmetric":
            # Keep the configured retrieval schema fixed even when callers use
            # runtime modality overrides for ablations. Missing streams become
            # masked zero blocks, so vector dimensions remain DB-compatible.
            schema_flags = (
                (self.use_visual_mod, self.use_char_mod, self.use_word_mod)
                if symmetric_modalities_from_instance
                else (USE_VISUAL_MOD, USE_CHAR_MOD, USE_WORD_MOD)
            )
            enabled = tuple(name for name, active in (
                ("tile", schema_flags[0]),
                ("glyph", schema_flags[1]),
                ("word", schema_flags[2]),
            ) if active)
            self.fusion = SymmetricRetrievalFusion(
                d_model=d_model,
                enabled_modalities=enabled,
                reliability_hidden_dim=SYMMETRIC_RELIABILITY_HIDDEN_DIM,
                use_adapters=self.use_branch_adapters_and_summarizers,
                branch_dim=SYMMETRIC_BRANCH_DIM,
            )
        else:
            raise ValueError(f"Unknown FUSION_METHOD: {FUSION_METHOD}")
        
        # Classification head
        if FUSION_METHOD == "symmetric":
            self.head = SymmetricRetrievalHead(self.fusion.output_dim, num_classes)
        else:
            self.head = PerceiverHead(d_model=d_model, num_classes=num_classes)
        
        # Modality dropout settings (for training regularization)
        self.token_subsample_enabled = TOKEN_SUBSAMPLE_ENABLED
        self.token_subsample_prob = TOKEN_SUBSAMPLE_PROB
        self.token_subsample_min_tiles = TOKEN_SUBSAMPLE_MIN_KEEP_TILES
        self.token_subsample_min_glyphs = TOKEN_SUBSAMPLE_MIN_KEEP_GLYPHS
        self.token_subsample_min_words = TOKEN_SUBSAMPLE_MIN_KEEP_WORDS
        self.token_subsample_frac_range = TOKEN_SUBSAMPLE_FRAC_RANGE
        self.modality_dropout_enabled = MODALITY_DROPOUT_ENABLED
        self.modality_dropout_prob_visual = MODALITY_DROPOUT_PROB_VISUAL
        self.modality_dropout_prob_char = MODALITY_DROPOUT_PROB_CHAR
        self.modality_dropout_prob_word = MODALITY_DROPOUT_PROB_WORD

    @staticmethod
    def _infer_batch_size(
        tiles: Optional[torch.Tensor],
        glyph_patches: Optional[torch.Tensor],
        words: Optional[List[List[str]]],
        device: torch.device,
    ) -> int:
        """Best-effort batch-size inference from whichever input is available."""
        if tiles is not None:
            return int(tiles.shape[0])
        if glyph_patches is not None:
            return int(glyph_patches.shape[0])
        if words is not None:
            return int(len(words))
        return 0

    @staticmethod
    def _valid_fraction(
        valid_mask: Optional[torch.Tensor],
        capacity: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Return a stable per-sample source-evidence fraction for scoring.

        Summarized branches normalize source count by query capacity (and cap at
        one), reproducing the old count signal without masking query outputs.
        Direct-token branches fall back to the mask width.
        """
        if valid_mask is None:
            return None
        if valid_mask.ndim != 2:
            raise ValueError(
                f"valid_mask must be rank 2, got shape {tuple(valid_mask.shape)}"
            )
        denominator = int(capacity) if capacity is not None else valid_mask.shape[1]
        if denominator <= 0:
            return torch.zeros(
                valid_mask.shape[0], device=valid_mask.device, dtype=torch.float32
            )
        count = valid_mask.to(dtype=torch.float32).sum(dim=1)
        return count.clamp_max(float(denominator)) / float(denominator)

    def _subsample_mask(
        self,
        valid_mask: torch.Tensor,  # [B, S] bool
        min_keep: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Vectorized per-sample subsampling of valid tokens.

        For each row independently, with probability ``token_subsample_prob``
        and only when ``n_valid > min_keep``, keep a random
        ``max(min_keep, round(n_valid * keep_frac))`` tokens (where
        ``keep_frac`` ∼ Uniform(frac_range)). Padding stays padding; per-sample
        decisions are computed on-device with a single RNG call per stage —
        no Python-level CUDA syncs.
        """
        B, S = valid_mask.shape
        if S == 0 or B == 0:
            return valid_mask.clone()

        lo, hi = self.token_subsample_frac_range
        n_valid = valid_mask.sum(dim=1)  # [B]

        # Per-sample: subsample only if RNG hits AND there's room above min_keep.
        do_sub = (torch.rand(B, device=device) < self.token_subsample_prob) & (
            n_valid > min_keep
        )

        # Per-sample keep count.
        keep_frac = lo + (hi - lo) * torch.rand(B, device=device)  # [B]
        n_keep = (n_valid.float() * keep_frac).round().long().clamp_min(min_keep)
        n_keep = torch.minimum(n_keep, n_valid)  # never exceed valid count

        # Random scores; padded positions are pinned below all valid scores so
        # they never get into the "kept" set even with lots of resampling.
        scores = torch.rand(B, S, device=device)
        scores = scores.masked_fill(~valid_mask, -1.0)

        # Rank of each position within its row (descending). The kept set is
        # the top-`n_keep[b]` positions per row.
        sorted_indices = scores.argsort(dim=1, descending=True)  # [B, S]
        rank = torch.empty_like(sorted_indices)
        arange_S = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        rank.scatter_(1, sorted_indices, arange_S)

        keep = (rank < n_keep.unsqueeze(1)) & valid_mask  # [B, S]
        return torch.where(do_sub.unsqueeze(1), keep, valid_mask)

    def _subsample_words(
        self,
        words: List[List[str]],
        word_metadata: Optional[List[List[Dict]]],
        min_keep: int,
        word_page_segments: Optional[List[List[int]]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Dict]]], Optional[List[List[int]]]]:
        """Randomly drop words per sample. Returns new lists (never modifies in-place).

        Subsamples word_metadata and word_page_segments with the same indices
        so they stay aligned. RNG calls are batched (one ``torch.rand`` per
        stage and at most one ``randperm`` per row) so we don't pay a
        CPU↔GPU sync per iteration."""
        lo, hi = self.token_subsample_frac_range
        B = len(words)
        new_words: List[List[str]] = []
        new_meta: Optional[List[List[Dict]]] = [] if word_metadata is not None else None
        new_segments: Optional[List[List[int]]] = [] if word_page_segments is not None else None

        if B == 0:
            return new_words, new_meta, new_segments

        rng_device = device if device is not None else torch.device("cpu")
        # Batch all per-sample randomness up front; one .tolist() call each.
        do_sub_all = (
            torch.rand(B, device=rng_device) < self.token_subsample_prob
        ).cpu().tolist()
        keep_fracs = (
            lo + (hi - lo) * torch.rand(B, device=rng_device)
        ).cpu().tolist()

        for b in range(B):
            w = words[b]
            m = word_metadata[b] if (word_metadata is not None and b < len(word_metadata)) else []
            seg = word_page_segments[b] if (word_page_segments is not None and b < len(word_page_segments)) else []
            if not do_sub_all[b] or len(w) <= min_keep:
                new_words.append(w)
                if new_meta is not None:
                    new_meta.append(m)
                if new_segments is not None:
                    new_segments.append(seg)
                continue
            n_keep = max(min_keep, int(round(len(w) * keep_fracs[b])))
            if n_keep >= len(w):
                new_words.append(w)
                if new_meta is not None:
                    new_meta.append(m)
                if new_segments is not None:
                    new_segments.append(seg)
                continue
            perm = torch.randperm(len(w), device=rng_device)
            indices = sorted(perm[:n_keep].tolist())
            new_words.append([w[i] for i in indices])
            if new_meta is not None:
                new_meta.append([m[i] for i in indices if i < len(m)])
            if new_segments is not None:
                new_segments.append([seg[i] for i in indices if i < len(seg)])
        return new_words, new_meta, new_segments

    def _word_sample_has_usable_entries(
        self,
        sample_words: List[str],
        sample_metadata: Optional[List[Dict]],
    ) -> bool:
        if not sample_words:
            return False
        if sample_metadata is None:
            return bool(sample_words)

        quality_filter = getattr(getattr(self, "word_branch", None), "quality_filter", None)
        if quality_filter is None or not hasattr(quality_filter, "filter_words_in_line"):
            return bool(sample_words)

        try:
            line_words = [
                (idx, word, meta)
                for idx, (word, meta) in enumerate(zip(sample_words, sample_metadata))
            ]
            return bool(quality_filter.filter_words_in_line(line_words))
        except Exception:
            # Preserve the existing fail-fast behavior in WordBranch for malformed
            # metadata instead of hiding input errors in dropout bookkeeping.
            return bool(sample_words)
    
    def forward(
        self,
        tiles: Optional[torch.Tensor] = None,  # [B, N, 3, H, W]
        tile_coords: Optional[torch.Tensor] = None,  # [B, N, 2]
        tile_valid_mask: Optional[torch.Tensor] = None,  # [B, N] bool
        tile_page_segments: Optional[torch.Tensor] = None,  # [B, N] long
        glyph_patches: Optional[torch.Tensor] = None,  # [B, M, 3, H, W]
        glyph_coords: Optional[torch.Tensor] = None,  # [B, M, 4]
        glyph_valid_mask: Optional[torch.Tensor] = None,  # [B, M] bool
        glyph_page_segments: Optional[torch.Tensor] = None,  # [B, M] long
        char_class_ids: Optional[torch.Tensor] = None,  # [B, M] long — Hebrew letter class
        words: Optional[List[List[str]]] = None,
        word_metadata: Optional[List[List[Dict]]] = None,
        word_page_segments: Optional[List[List[int]]] = None,
        paths: Optional[List[str]] = None,
        batch_element_indices: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        return_aux_latents: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the new architecture.
        
        Args:
            tiles: [B, N, 3, H, W] - visual tile patches
            tile_coords: [B, N, 2] - normalized tile coordinates (x, y)
            tile_valid_mask: [B, N] bool - mask for valid tiles
            tile_page_segments: [B, N] long (optional) - page segment IDs for two-page manuscripts
            glyph_patches: [B, M, 3, H, W] - glyph/character patches
            glyph_coords: [B, M, 4] (optional) - glyph coordinates (x, y, w, h)
            glyph_valid_mask: [B, M] bool - mask for valid glyphs
            glyph_page_segments: [B, M] long (optional) - page segment IDs for glyphs
            words: List[List[str]] - words per image
            word_metadata: List[List[Dict]] - word metadata per image
            word_page_segments: List[List[int]] (optional) - page segment IDs for words
            batch_element_indices: Optional [B_local] long tensor of indices into the
                collated batch (e.g. ``torch.arange(B, device=...)``). Required for
                correct DataParallel behavior with list kwargs (words, paths, ...).
            device: torch.device (optional) - device to use
        
        Returns:
            - logits: [B, num_classes]
            - latent: [B, LATENT_DIM]
            - tile_latent: [B, d_model] or None (if return_tile_latent=True)
        """
        # Determine device (robust for word-only or empty batches)
        if device is None:
            for tensor in [tiles, glyph_patches]:
                if tensor is not None:
                    device = tensor.device
                    break
            if device is None:
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = torch.device("cpu")
        
        words, word_metadata, word_page_segments, paths = _apply_batch_element_indices_to_lists(
            batch_element_indices, words, word_metadata, word_page_segments, paths
        )

        use_visual = self.use_visual_mod
        use_char = self.use_char_mod
        use_word = self.use_word_mod

        tensor_batch_size = None
        for tensor in (tiles, glyph_patches):
            if tensor is not None:
                tensor_batch_size = int(tensor.shape[0])
                break
        if tensor_batch_size is not None and use_word:
            _validate_list_batch_size(words, tensor_batch_size, "words")
            _validate_list_batch_size(word_metadata, tensor_batch_size, "word_metadata")
            _validate_list_batch_size(word_page_segments, tensor_batch_size, "word_page_segments")

        if use_visual and tiles is not None:
            if tile_valid_mask is None:
                tile_valid_mask = torch.ones(
                    tiles.shape[:2], dtype=torch.bool, device=tiles.device
                )
            else:
                if tile_valid_mask.ndim != 2 or tuple(tile_valid_mask.shape) != tuple(tiles.shape[:2]):
                    raise ValueError(
                        f"MultiModal.forward: tile_valid_mask must have shape {tuple(tiles.shape[:2])} "
                        f"to match tiles; got {tuple(tile_valid_mask.shape)}."
                    )
                tile_valid_mask = tile_valid_mask.to(device=tiles.device, dtype=torch.bool)

        if use_char and glyph_patches is not None:
            if glyph_valid_mask is None:
                glyph_valid_mask = torch.ones(
                    glyph_patches.shape[:2], dtype=torch.bool, device=glyph_patches.device
                )
            else:
                if glyph_valid_mask.ndim != 2 or tuple(glyph_valid_mask.shape) != tuple(glyph_patches.shape[:2]):
                    raise ValueError(
                        f"MultiModal.forward: glyph_valid_mask must have shape {tuple(glyph_patches.shape[:2])} "
                        f"to match glyph_patches; got {tuple(glyph_valid_mask.shape)}."
                    )
                glyph_valid_mask = glyph_valid_mask.to(device=glyph_patches.device, dtype=torch.bool)

        # ------------------------------------------------------------------
        # Per-sample modality dropout (training only).
        # ------------------------------------------------------------------
        # Unlike the previous batch-wide coin flip, each sample independently
        # decides which modalities are kept. We implement this by zeroing out
        # the per-sample valid masks (and clearing the words list) for "dropped"
        # samples; the branches still run for the rest of the batch.
        # If a sample's only available modalities all drop, we randomly rescue
        # one to avoid relying purely on the all-empty null-token fallback.
        if self.training and self.modality_dropout_enabled:
            B_local = self._infer_batch_size(tiles, glyph_patches, words, device)
            if B_local > 0:
                visual_avail = self.use_visual_mod and tiles is not None
                char_avail = (
                    self.use_char_mod
                    and glyph_patches is not None
                    and glyph_coords is not None
                )
                word_avail = (
                    self.use_word_mod
                    and words is not None
                    and word_metadata is not None
                )

                if visual_avail and tile_valid_mask is not None:
                    visual_avail_b = tile_valid_mask.any(dim=1).to(device=device)
                else:
                    visual_avail_b = torch.zeros(B_local, dtype=torch.bool, device=device)
                if char_avail and glyph_valid_mask is not None:
                    char_avail_b = glyph_valid_mask.any(dim=1).to(device=device)
                else:
                    char_avail_b = torch.zeros(B_local, dtype=torch.bool, device=device)
                if word_avail:
                    word_avail_b = torch.tensor(
                        [
                            self._word_sample_has_usable_entries(
                                w,
                                word_metadata[b] if b < len(word_metadata) else None,
                            )
                            for b, w in enumerate(words)
                        ],
                        dtype=torch.bool,
                        device=device,
                    )
                else:
                    word_avail_b = torch.zeros(B_local, dtype=torch.bool, device=device)

                keep_v_b = visual_avail_b.clone()
                keep_c_b = char_avail_b.clone()
                keep_w_b = word_avail_b.clone()

                if visual_avail:
                    keep_v_b = keep_v_b & (
                        torch.rand(B_local, device=device) >= self.modality_dropout_prob_visual
                    )
                if char_avail:
                    keep_c_b = keep_c_b & (
                        torch.rand(B_local, device=device) >= self.modality_dropout_prob_char
                    )
                if word_avail:
                    keep_w_b = keep_w_b & (
                        torch.rand(B_local, device=device) >= self.modality_dropout_prob_word
                    )

                # Rescue only from modalities that are actually present for
                # that sample. Batch-level availability is not enough: a batch
                # can contain word lists while an individual sample has no
                # usable words, or padded tile/glyph streams with zero valid
                # tokens. Rescuing an empty modality would force the fusion
                # stack to train on its null-token fallback.
                any_avail_b = visual_avail_b | char_avail_b | word_avail_b
                none_kept = any_avail_b & ~(keep_v_b | keep_c_b | keep_w_b)
                if none_kept.any():
                    rescue_scores = torch.rand(B_local, 3, device=device)
                    avail_stack = torch.stack(
                        [visual_avail_b, char_avail_b, word_avail_b],
                        dim=1,
                    )
                    rescue_scores = rescue_scores.masked_fill(~avail_stack, -1.0)
                    choice = rescue_scores.argmax(dim=1)
                    keep_v_b = keep_v_b | (none_kept & (choice == 0))
                    keep_c_b = keep_c_b | (none_kept & (choice == 1))
                    keep_w_b = keep_w_b | (none_kept & (choice == 2))

                # Apply to per-sample masks.
                if visual_avail and tile_valid_mask is not None:
                    tile_valid_mask = tile_valid_mask & keep_v_b.unsqueeze(1)
                if char_avail and glyph_valid_mask is not None:
                    glyph_valid_mask = glyph_valid_mask & keep_c_b.unsqueeze(1)
                if word_avail:
                    keep_w_list = keep_w_b.detach().cpu().tolist()
                    words = [
                        w if (b < len(keep_w_list) and keep_w_list[b]) else []
                        for b, w in enumerate(words)
                    ]
                    if word_metadata is not None:
                        word_metadata = [
                            m if (b < len(keep_w_list) and keep_w_list[b]) else []
                            for b, m in enumerate(word_metadata)
                        ]
                    if word_page_segments is not None:
                        word_page_segments = [
                            s if (b < len(keep_w_list) and keep_w_list[b]) else []
                            for b, s in enumerate(word_page_segments)
                        ]

        # Token subsampling (training only): randomly mask out a fraction of
        # valid tokens per modality per sample so the model learns to produce
        # stable representations regardless of tile/glyph count (full page vs fragment).
        if self.training and self.token_subsample_enabled:
            if tile_valid_mask is not None and use_visual:
                tile_valid_mask = self._subsample_mask(
                    tile_valid_mask, self.token_subsample_min_tiles, device)
            if glyph_valid_mask is not None and use_char:
                glyph_valid_mask = self._subsample_mask(
                    glyph_valid_mask, self.token_subsample_min_glyphs, device)

        # Process each modality through its branch
        tile_query_capacity = getattr(
            getattr(getattr(self, "tile_branch", None), "set_summarizer", None),
            "num_queries",
            None,
        )
        glyph_query_capacity = getattr(
            getattr(getattr(self, "glyph_branch", None), "set_summarizer", None),
            "num_queries",
            None,
        )
        tile_evidence_fraction = self._valid_fraction(
            tile_valid_mask, tile_query_capacity
        )
        glyph_evidence_fraction = self._valid_fraction(
            glyph_valid_mask, glyph_query_capacity
        )
        word_evidence_fraction = None
        tile_tokens = None
        tile_valid = None
        if use_visual:
            if tiles is not None:
                tile_branch_output = self.tile_branch(
                    tiles=tiles,
                    tile_coords=tile_coords,
                    valid_mask=tile_valid_mask,
                    page_segments=tile_page_segments,
                )
                # Branches return encoder tokens plus the corresponding valid
                # mask; tensor-only test/custom branches remain supported.
                if isinstance(tile_branch_output, tuple):
                    if len(tile_branch_output) != 2:
                        raise ValueError(
                            "TileBranch must return (tokens, valid_mask) during forward"
                        )
                    tile_tokens, tile_valid = tile_branch_output
                else:
                    tile_tokens = tile_branch_output
                    tile_valid = tile_valid_mask
                if tile_valid is not None:
                    tile_valid = tile_valid.to(device=tile_tokens.device, dtype=torch.bool)
        
        glyph_tokens = None
        glyph_valid = None
        if use_char:
            if glyph_patches is not None and glyph_coords is not None:
                glyph_tokens, glyph_valid = self.glyph_branch(
                    glyph_patches=glyph_patches,
                    glyph_coords=glyph_coords,
                    valid_mask=glyph_valid_mask,
                    page_segments=glyph_page_segments,
                    char_class_ids=char_class_ids,
                    debug_paths=paths,
                )  # [B, Mg, d_model], [B, Mg]
                glyph_valid = glyph_valid.to(device=glyph_tokens.device)
        
        word_tokens = None
        word_valid = None
        if use_word:
            if words is not None and word_metadata is not None:
                if word_page_segments is None:
                    word_page_segments = _word_page_segments_from_metadata(word_metadata)
                if self.training and self.token_subsample_enabled:
                    words, word_metadata, word_page_segments = self._subsample_words(
                        words, word_metadata, self.token_subsample_min_words, word_page_segments, device
                    )
                word_kwargs = dict(
                    words=words,
                    word_metadata=word_metadata,
                    device=device,
                    page_segments=word_page_segments,
                )
                supports_source_evidence = (
                    "return_source_evidence"
                    in inspect.signature(self.word_branch.forward).parameters
                )
                if supports_source_evidence:
                    word_kwargs["return_source_evidence"] = True
                word_branch_output = self.word_branch(**word_kwargs)
                if supports_source_evidence:
                    word_tokens, word_valid, word_evidence_fraction = word_branch_output
                else:
                    word_tokens, word_valid = word_branch_output
                    word_evidence_fraction = self._valid_fraction(word_valid)
        
        # Fuse modalities
        fusion_kwargs = dict(
            tile_tokens=tile_tokens,
            tile_valid_mask=tile_valid,
            glyph_tokens=glyph_tokens,
            glyph_valid_mask=glyph_valid,
            word_tokens=word_tokens,
            word_valid_mask=word_valid,
        )
        if isinstance(self.fusion, SymmetricRetrievalFusion):
            fusion_kwargs.update(
                tile_evidence_fraction=tile_evidence_fraction,
                glyph_evidence_fraction=glyph_evidence_fraction,
                word_evidence_fraction=word_evidence_fraction,
            )
        latent_repr, fusion_details = self.fusion(**fusion_kwargs)

        # Classify
        logits, latent = self.head(latent_repr)  # symmetric retrieval bypasses the legacy MLP/BN head
        
        # Handle auxiliary latent extraction (if requested)
        aux_latents = {}
        if return_aux_latents:
            # Batch size for zero-initialization (robust for all GPUs in DataParallel)
            B = latent.shape[0]
            
            # Initialize with zeros for all branches enabled in the model
            # This is CRITICAL for DataParallel.gather() which requires identical keys across GPUs.
            # We use latent.dtype (typically BFloat16) to ensure matching types during concatenation.
            fusion_aux_dim = latent.shape[1] if FUSION_METHOD == "symmetric" else self.d_model
            branch_aux_dim = (
                SYMMETRIC_BRANCH_DIM
                if FUSION_METHOD == "symmetric"
                and self.use_branch_adapters_and_summarizers
                else self.d_model
            )
            aux_latents['fusion'] = torch.zeros(B, fusion_aux_dim, device=device, dtype=latent.dtype)
            if self.use_visual_mod:
                aux_latents['tile'] = torch.zeros(B, branch_aux_dim, device=device, dtype=latent.dtype)
            if self.use_char_mod:
                aux_latents['glyph'] = torch.zeros(B, branch_aux_dim, device=device, dtype=latent.dtype)
            if self.use_word_mod:
                aux_latents['word'] = torch.zeros(B, branch_aux_dim, device=device, dtype=latent.dtype)

            # 1. Tile Latent (overwrite if data was processed)
            if FUSION_METHOD == "symmetric" and isinstance(fusion_details, dict):
                for key in ("tile", "glyph", "word", "fusion",
                            "reliability_weights", "modality_available",
                            "evidence_fractions"):
                    if key in fusion_details:
                        aux_latents[key] = fusion_details[key].to(dtype=latent.dtype)
            elif tile_tokens is not None and tile_valid is not None:
                mask_float = tile_valid.unsqueeze(-1).to(tile_tokens.dtype)
                sum_tokens = (tile_tokens * mask_float).sum(dim=1)
                count = mask_float.sum(dim=1).clamp(min=1e-6)
                aux_latents['tile'] = sum_tokens / count
            
            # 2. Glyph Latent (overwrite if data was processed)
            if FUSION_METHOD != "symmetric" and glyph_tokens is not None and glyph_valid is not None:
                mask_float = glyph_valid.unsqueeze(-1).to(glyph_tokens.dtype)
                sum_tokens = (glyph_tokens * mask_float).sum(dim=1)
                count = mask_float.sum(dim=1).clamp(min=1e-6)
                aux_latents['glyph'] = sum_tokens / count
                
            # 3. Word Latent (overwrite if data was processed)
            if FUSION_METHOD != "symmetric" and word_tokens is not None and word_valid is not None:
                mask_float = word_valid.unsqueeze(-1).to(word_tokens.dtype)
                sum_tokens = (word_tokens * mask_float).sum(dim=1)
                count = mask_float.sum(dim=1).clamp(min=1e-6)
                aux_latents['word'] = sum_tokens / count

            if FUSION_METHOD != "symmetric" and latent_repr is not None:
                aux_latents['fusion'] = latent_repr.mean(dim=1).to(dtype=latent.dtype)

        return logits, latent, aux_latents
    
    def forward_features(
        self,
        tiles: Optional[torch.Tensor] = None,
        tile_coords: Optional[torch.Tensor] = None,
        tile_valid_mask: Optional[torch.Tensor] = None,
        tile_page_segments: Optional[torch.Tensor] = None,
        glyph_patches: Optional[torch.Tensor] = None,
        glyph_coords: Optional[torch.Tensor] = None,
        glyph_valid_mask: Optional[torch.Tensor] = None,
        glyph_page_segments: Optional[torch.Tensor] = None,
        char_class_ids: Optional[torch.Tensor] = None,
        words: Optional[List[List[str]]] = None,
        word_metadata: Optional[List[List[Dict]]] = None,
        word_page_segments: Optional[List[List[int]]] = None,
        paths: Optional[List[str]] = None,
        batch_element_indices: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        return_aux_latents: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass that returns only the latent representation (for inference/clustering).
        
        Returns:
            Tuple of (latent, aux_latents)
        """
        logits, latent, aux_latents = self.forward(
            tiles=tiles,
            tile_coords=tile_coords,
            tile_valid_mask=tile_valid_mask,
            tile_page_segments=tile_page_segments,
            glyph_patches=glyph_patches,
            glyph_coords=glyph_coords,
            glyph_valid_mask=glyph_valid_mask,
            glyph_page_segments=glyph_page_segments,
            char_class_ids=char_class_ids,
            words=words,
            word_metadata=word_metadata,
            word_page_segments=word_page_segments,
            paths=paths,
            batch_element_indices=batch_element_indices,
            device=device,
            return_aux_latents=return_aux_latents,
        )
        return latent, aux_latents
    
    def forward_with_attention(
        self,
        tiles: Optional[torch.Tensor] = None,  # [B, N, 3, H, W]
        tile_coords: Optional[torch.Tensor] = None,  # [B, N, 2]
        tile_valid_mask: Optional[torch.Tensor] = None,  # [B, N] bool
        tile_page_segments: Optional[torch.Tensor] = None,  # [B, N] long
        glyph_patches: Optional[torch.Tensor] = None,  # [B, M, 3, H, W]
        glyph_coords: Optional[torch.Tensor] = None,  # [B, M, 4]
        glyph_valid_mask: Optional[torch.Tensor] = None,  # [B, M] bool
        glyph_page_segments: Optional[torch.Tensor] = None,  # [B, M] long
        char_class_ids: Optional[torch.Tensor] = None,  # [B, M] long
        words: Optional[List[List[str]]] = None,  # [B] list of word lists
        word_metadata: Optional[List[List[Dict]]] = None,  # [B] list of metadata lists
        word_page_segments: Optional[List[List[int]]] = None,  # [B] list of page_segment lists
        paths: Optional[List[str]] = None,  # [B] list/tuple of image paths (for fail-fast debug)
        batch_element_indices: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with attention extraction for visualization.
        Argument order matches forward() to avoid positional-argument footguns.
        Intentionally does not apply modality dropout or token subsampling (for stable viz);
        if called during training, behavior will differ from forward().
        
        Args (same order as forward):
            tiles: [B, N, 3, H, W] - tile crops
            tile_coords: [B, N, 2] - normalized tile coordinates
            tile_valid_mask: [B, N] bool - valid mask for tiles
            tile_page_segments: [B, N] long (optional)
            glyph_patches: [B, M, 3, H, W] - glyph patches
            glyph_coords: [B, M, 4] (optional) - glyph coordinates
            glyph_valid_mask: [B, M] bool - valid mask for glyphs
            glyph_page_segments: [B, M] long (optional)
            char_class_ids: [B, M] long (optional)
            words: [B] list of word string lists
            word_metadata: [B] list of metadata dict lists
            word_page_segments: [B] list of page_segment lists (optional)
            paths: [B] list of image paths (optional)
            device: torch.device (optional)
            
        Returns:
            Tuple of:
            - logits: [B, num_classes]
            - latent: [B, LATENT_DIM]
            - attention_dict: Dict with keys 'visual', 'character', 'word', 'fusion',
              and branch-local composition maps. The modality matrices use only
              row 0 for global-to-input contribution scores; their remaining rows
              are zero and must not be interpreted as token-to-token attention.
        """
        # Determine device (robust for word-only or empty batches)
        if device is None:
            for tensor in [tiles, glyph_patches]:
                if tensor is not None:
                    device = tensor.device
                    break
            if device is None:
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = torch.device("cpu")
        
        words, word_metadata, word_page_segments, paths = _apply_batch_element_indices_to_lists(
            batch_element_indices, words, word_metadata, word_page_segments, paths
        )
        
        # Use all modalities; no dropout or token subsampling (by design for inference/viz).
        use_visual = self.use_visual_mod
        use_char = self.use_char_mod
        use_word = self.use_word_mod

        tensor_batch_size = None
        for tensor in (tiles, glyph_patches):
            if tensor is not None:
                tensor_batch_size = int(tensor.shape[0])
                break
        if tensor_batch_size is not None and use_word:
            _validate_list_batch_size(words, tensor_batch_size, "words")
            _validate_list_batch_size(word_metadata, tensor_batch_size, "word_metadata")
            _validate_list_batch_size(word_page_segments, tensor_batch_size, "word_page_segments")

        if use_char and glyph_patches is not None:
            if glyph_valid_mask is None:
                glyph_valid_mask = torch.ones(
                    glyph_patches.shape[:2], dtype=torch.bool, device=glyph_patches.device
                )
            else:
                if glyph_valid_mask.ndim != 2 or tuple(glyph_valid_mask.shape) != tuple(glyph_patches.shape[:2]):
                    raise ValueError(
                        f"MultiModal.forward_with_attention: glyph_valid_mask must have shape {tuple(glyph_patches.shape[:2])} "
                        f"to match glyph_patches; got {tuple(glyph_valid_mask.shape)}."
                    )
                glyph_valid_mask = glyph_valid_mask.to(device=glyph_patches.device, dtype=torch.bool)
        
        attention_dict = {}
        
        # Process each modality through its branch with attention extraction
        tile_tokens = None
        tile_valid = None
        tile_attn = None
        tile_evidence_fraction = None
        if use_visual:
            if tiles is not None:
                if tile_valid_mask is None:
                    tile_valid_for_branch = torch.ones(
                        tiles.shape[:2], dtype=torch.bool, device=tiles.device
                    )
                else:
                    if tile_valid_mask.ndim != 2 or tuple(tile_valid_mask.shape) != tuple(tiles.shape[:2]):
                        raise ValueError(
                            f"MultiModal.forward_with_attention: tile_valid_mask must have shape {tuple(tiles.shape[:2])} "
                            f"to match tiles; got {tuple(tile_valid_mask.shape)}."
                        )
                    tile_valid_for_branch = tile_valid_mask.to(device=tiles.device, dtype=torch.bool)
                tile_query_capacity = getattr(
                    getattr(self.tile_branch, "set_summarizer", None),
                    "num_queries",
                    None,
                )
                tile_evidence_fraction = self._valid_fraction(
                    tile_valid_for_branch, tile_query_capacity
                )

                tile_branch_output = self.tile_branch(
                    tiles=tiles,
                    tile_coords=tile_coords,
                    valid_mask=tile_valid_for_branch,
                    page_segments=tile_page_segments,
                    return_attention=True,
                )
                if not isinstance(tile_branch_output, tuple):
                    tile_tokens = tile_branch_output
                    tile_valid = tile_valid_for_branch
                elif len(tile_branch_output) == 3:
                    tile_tokens, tile_valid, tile_attn = tile_branch_output
                elif len(tile_branch_output) == 2:
                    # Legacy/custom attention branches returned (tokens, attn)
                    # and retained the raw input mask.
                    tile_tokens, legacy_attention = tile_branch_output
                    tile_valid = tile_valid_for_branch
                    tile_attn = legacy_attention
                else:
                    raise ValueError(
                        "TileBranch must return (tokens, mask, attention) when "
                        "return_attention=True"
                    )
                tile_valid = tile_valid.to(device=tile_tokens.device, dtype=torch.bool)

                if tile_attn is not None:
                    attention_dict['tile_query_to_tile'] = tile_attn
                    # Branch-local fallback for fusion methods without global
                    # token contributions. Active query tokens are mean-pooled
                    # by the downstream branch aggregation, exactly as glyphs.
                    query_weights = tile_valid.to(tile_attn.dtype)
                    query_weights = query_weights / query_weights.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1.0)
                    local_tile_scores = torch.bmm(
                        query_weights.unsqueeze(1), tile_attn
                    ).squeeze(1)
                    raw_tile_count = tile_attn.shape[2]
                    visual_attn_matrix = torch.zeros(
                        tile_attn.shape[0], raw_tile_count + 1, raw_tile_count + 1,
                        device=tile_attn.device, dtype=tile_attn.dtype,
                    )
                    visual_attn_matrix[:, 0, 1:] = local_tile_scores
                    attention_dict['visual'] = visual_attn_matrix
        
        glyph_tokens = None
        glyph_valid = None
        glyph_attn = None
        glyph_query_capacity = getattr(
            getattr(getattr(self, "glyph_branch", None), "set_summarizer", None),
            "num_queries",
            None,
        )
        glyph_evidence_fraction = self._valid_fraction(
            glyph_valid_mask, glyph_query_capacity
        )
        if use_char:
            if glyph_patches is not None and glyph_coords is not None:
                glyph_tokens, glyph_valid, glyph_attn = self.glyph_branch(
                    glyph_patches=glyph_patches,
                    glyph_coords=glyph_coords,
                    valid_mask=glyph_valid_mask,
                    page_segments=glyph_page_segments,
                    char_class_ids=char_class_ids,
                    return_attention=True,
                    debug_paths=paths,
                )  # [B, Q, d_model], [B, Q], [B, Q, M]
                glyph_valid = glyph_valid.to(device=glyph_tokens.device)
                # Convert glyph attention to visualization-friendly forms.
                # Note: GlyphSetSummarizer may append a 1-token "null" key for numerical safety,
                # so glyph_attn can have M+1 keys even when the dataset mask has length M.
                if glyph_attn is not None:
                    B, num_summary, M_attn = glyph_attn.shape
                    M_mask = int(glyph_valid_mask.shape[1]) if glyph_valid_mask is not None else M_attn
                    M = min(M_attn, M_mask)
                    if M_attn != M_mask:
                        # Trim attention to match the dataset glyph count/mask (drop the null token for visualization).
                        glyph_attn = glyph_attn[..., :M]

                    # Save the *real* summarizer attention for debugging/visualization:
                    # glyph_attn is query->glyph attention: [B, num_summary_tokens, M]
                    attention_dict['glyph_query_to_glyph'] = glyph_attn
                    # Branch-local fallback used only when the selected fusion
                    # module cannot expose global token contributions. Averaging
                    # the active summary queries is the faithful composition for
                    # the summarizer; max pooling would invent a sharper map.
                    summary_weights = glyph_valid.to(glyph_attn.dtype)
                    summary_weights = summary_weights / summary_weights.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1.0)
                    cls_attn = torch.bmm(
                        summary_weights.unsqueeze(1), glyph_attn
                    ).squeeze(1)
                    char_attn_matrix = torch.zeros(
                        B, M + 1, M + 1, device=glyph_attn.device, dtype=glyph_attn.dtype
                    )
                    char_attn_matrix[:, 0, 1:] = cls_attn  # CLS attends to glyphs
                    attention_dict['character'] = char_attn_matrix
        
        word_tokens = None
        word_valid = None
        word_attn_local = None
        line_to_word_attn = None
        summary_to_line_attn = None
        word_evidence_fraction = None
        if use_word:
            if words is not None and word_metadata is not None:
                if word_page_segments is None:
                    word_page_segments = _word_page_segments_from_metadata(word_metadata)
                word_kwargs = dict(
                    words=words,
                    word_metadata=word_metadata,
                    device=device,
                    page_segments=word_page_segments,
                    return_attention=True,
                    return_line_to_word_attn=True,
                    return_summary_to_line_attn=True,
                )
                supports_source_evidence = (
                    "return_source_evidence"
                    in inspect.signature(self.word_branch.forward).parameters
                )
                if supports_source_evidence:
                    word_kwargs["return_source_evidence"] = True
                word_branch_output = self.word_branch(**word_kwargs)
                if supports_source_evidence:
                    (
                        word_tokens,
                        word_valid,
                        word_attn_local,
                        line_to_word_attn,
                        summary_to_line_attn,
                        word_evidence_fraction,
                    ) = word_branch_output
                else:
                    (
                        word_tokens,
                        word_valid,
                        word_attn_local,
                        line_to_word_attn,
                        summary_to_line_attn,
                    ) = word_branch_output
                    word_evidence_fraction = self._valid_fraction(word_valid)
                if word_attn_local is not None:
                    attention_dict['word_local'] = word_attn_local
        
        # Fuse modalities — request attention weights for visualization
        fusion_kwargs = dict(
            tile_tokens=tile_tokens,
            tile_valid_mask=tile_valid,
            glyph_tokens=glyph_tokens,
            glyph_valid_mask=glyph_valid,
            word_tokens=word_tokens,
            word_valid_mask=word_valid,
        )
        if isinstance(self.fusion, SymmetricRetrievalFusion):
            fusion_kwargs.update(
                tile_evidence_fraction=tile_evidence_fraction,
                glyph_evidence_fraction=glyph_evidence_fraction,
                word_evidence_fraction=word_evidence_fraction,
            )
        # Fusion modules that expose return_attention opt in for visualization.
        if "return_attention" in inspect.signature(self.fusion.forward).parameters:
            fusion_kwargs["return_attention"] = True
        latent_repr, fusion_attn = self.fusion(**fusion_kwargs)
        # [B, num_latents/1, d_model], [B, num_latents/1, total_tokens] or [B, H, S, S]
        if FUSION_METHOD == "symmetric" and isinstance(fusion_attn, dict):
            symmetric_details = fusion_attn
            attention_dict['reliability_weights'] = symmetric_details['reliability_weights']
            attention_dict['fusion_modalities'] = symmetric_details['reliability_weights']
            attention_dict['evidence_fractions'] = symmetric_details['evidence_fractions']
            # For symmetric fusion this is the input contribution prior:
            # reliability per modality multiplied by each mask-mean weight.
            fusion_attn = symmetric_details.get("attention")
        
        # Standardize fusion_attn: if 4D (multi-head), average over heads
        # Note: TransformerFusion and PerceiverFusion already return 3D [B, 1/L, S]
        if fusion_attn is not None and fusion_attn.dim() == 4:
            fusion_attn = fusion_attn.mean(dim=1)  # [B, S, S]
            # Extract CLS row (0) and modality columns (1:) ONLY if it's the raw square matrix.
            # TransformerFusion returns a pre-sliced [B, 1, total_modality_len].
            if fusion_attn.shape[1] == fusion_attn.shape[2]:
                fusion_attn = fusion_attn[:, 0:1, 1:]  # [B, 1, total_modality_len]
        
        # Store standardized fusion attention
        if fusion_attn is not None:
            attention_dict['fusion'] = fusion_attn
        
        # Create visual and character attention from fusion attention.
        # This ensures heatmaps show what the GLOBAL model cared about.
        if fusion_attn is not None:
            B = fusion_attn.shape[0]
            total_tokens = fusion_attn.shape[2]
            tile_summary_len = tile_tokens.shape[1] if tile_tokens is not None else 0
            glyph_summary_len = glyph_tokens.shape[1] if glyph_tokens is not None else 0
            
            # 1. Visual Patches (Tiles)
            if tile_summary_len > 0 and use_visual:
                tile_start = 0
                tile_end = tile_start + tile_summary_len
                latent_to_tile_summaries = fusion_attn[:, :, tile_start:tile_end]
                tile_summary_importance = latent_to_tile_summaries.mean(dim=1)

                if tile_attn is not None:
                    # Compose global summary-token contribution through the
                    # real query→raw-tile attention, matching the glyph path.
                    global_tile_attn = torch.bmm(
                        tile_summary_importance.to(tile_attn.dtype).unsqueeze(1),
                        tile_attn,
                    ).squeeze(1)
                    if tile_valid_for_branch is not None:
                        raw_count = global_tile_attn.shape[1]
                        global_tile_attn = global_tile_attn * tile_valid_for_branch[
                            :, :raw_count
                        ].to(device=global_tile_attn.device, dtype=global_tile_attn.dtype)
                    raw_tile_count = global_tile_attn.shape[1]
                    visual_attn_matrix = torch.zeros(
                        B, raw_tile_count + 1, raw_tile_count + 1,
                        device=global_tile_attn.device,
                        dtype=global_tile_attn.dtype,
                    )
                    visual_attn_matrix[:, 0, 1:] = global_tile_attn
                    attention_dict['visual'] = visual_attn_matrix
                    attention_dict['tile_query_to_tile'] = tile_attn
                else:
                    # Compatibility path for checkpoints/configurations without
                    # a tile summarizer: fusion scores already correspond to
                    # raw tiles and symmetric fusion therefore remains uniform.
                    raw_tile_scores = tile_summary_importance
                    if tile_valid is not None:
                        raw_tile_scores = raw_tile_scores * tile_valid.to(
                            device=raw_tile_scores.device,
                            dtype=raw_tile_scores.dtype,
                        )
                    raw_tile_count = raw_tile_scores.shape[1]
                    visual_attn_matrix = torch.zeros(
                        B, raw_tile_count + 1, raw_tile_count + 1,
                        device=raw_tile_scores.device,
                        dtype=raw_tile_scores.dtype,
                    )
                    visual_attn_matrix[:, 0, 1:] = raw_tile_scores
                    attention_dict['visual'] = visual_attn_matrix

            # 2. Character Patches (Glyphs)
            if glyph_summary_len > 0 and use_char:
                glyph_start = tile_summary_len
                glyph_end = tile_summary_len + glyph_summary_len
                latent_to_glyph_tokens = fusion_attn[:, :, glyph_start:glyph_end]
                glyph_token_importance = latent_to_glyph_tokens.mean(dim=1)

                if glyph_attn is not None:
                    # Legacy summarized path: compose summary importance through
                    # the learned query-to-glyph attention map.
                    global_glyph_attn = torch.bmm(
                        glyph_token_importance.unsqueeze(1), glyph_attn
                    ).squeeze(1)
                else:
                    # Active direct-token path: fusion entries already align
                    # one-to-one with raw glyph tokens.
                    global_glyph_attn = glyph_token_importance
                
                # Mask padding
                if glyph_valid_mask is not None:
                    M_avail = global_glyph_attn.shape[1]
                    mask = glyph_valid_mask[:, :M_avail].float()
                    global_glyph_attn = global_glyph_attn * mask.to(global_glyph_attn.device)
                
                # Create synthetic proxy matrix for trainer visualization [B, M+1, M+1]
                M = global_glyph_attn.shape[1]
                char_attn_matrix = torch.zeros(
                    B, M + 1, M + 1,
                    device=global_glyph_attn.device,
                    dtype=global_glyph_attn.dtype,
                )
                char_attn_matrix[:, 0, 1:] = global_glyph_attn
                attention_dict['character'] = char_attn_matrix
                if glyph_attn is not None:
                    attention_dict['glyph_query_to_glyph'] = glyph_attn

            # 3. Words: map line-token contributions back to the source words.
            word_summary_len = word_tokens.shape[1] if word_tokens is not None else 0
            if (
                word_summary_len > 0
                and use_word
                and line_to_word_attn is not None
            ):
                word_start = tile_summary_len + glyph_summary_len
                word_end = word_start + word_summary_len

                latent_to_word_tokens = fusion_attn[:, :, word_start:word_end]
                word_token_importance = latent_to_word_tokens.mean(dim=1)

                W = line_to_word_attn.shape[2]

                if summary_to_line_attn is not None:
                    # Legacy summarized path: [B, Q, L] @ [B, L, W].
                    token_to_word = torch.bmm(
                        summary_to_line_attn.to(line_to_word_attn.dtype),
                        line_to_word_attn,
                    )
                else:
                    # Active path: each fusion token is already a line token.
                    token_to_word = line_to_word_attn

                # Per-word importance (raw — no sum-normalization)
                global_word_attn = torch.bmm(
                    word_token_importance.to(token_to_word.dtype).unsqueeze(1),
                    token_to_word,
                ).squeeze(1)  # [B, W]

                word_has_line = line_to_word_attn.sum(dim=1) > 1e-8  # [B, W]
                global_word_attn = global_word_attn * word_has_line.float()

                attention_dict['word_cls_to_word'] = global_word_attn
                attention_dict['word_query_to_word'] = token_to_word

                # Visualization carrier: only row 0 is meaningful.
                word_attn_matrix = torch.zeros(
                    B, W + 1, W + 1,
                    device=global_word_attn.device,
                    dtype=global_word_attn.dtype,
                )
                word_attn_matrix[:, 0, 1:] = global_word_attn
                attention_dict['word'] = word_attn_matrix
                if summary_to_line_attn is not None:
                    attention_dict['word_query_to_line'] = summary_to_line_attn
        
        # Classify
        logits, latent = self.head(latent_repr)  # [B, num_classes], [B, LATENT_DIM]
        
        return logits, latent, attention_dict
    
