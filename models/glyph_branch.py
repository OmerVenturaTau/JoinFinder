"""
Glyph Branch Architecture - New Architecture Implementation

This module implements the glyph branch according to the new architecture:
1. GLYPH HARD QUALITY FILTER - drop tiny/blank/extreme aspect glyphs
2. GLYPH VISUAL ENCODER - ConvNeXt-Tiny / Swin-T, per-glyph feature -> [B,M,Dg]
3. GLYPH TOKEN ENRICHMENT - proj(g_feat) + pos_glyph=f([x,y,w,h]) + emb(line_id/region_id)
4. Optionally summarize the glyph set; the active path returns glyph tokens directly
"""

import torch
import torch.nn as nn
import timm
import math
import gc
from typing import Optional, List, Dict, Tuple
from system import (
    CHAR_PATCH_SIZE,
    D_MODEL,
    CHAR_ENCODE_CHUNK_SIZE,
    XML_PATCH_READING_DIRECTION_RTL,
    GLYPH_ENCODER_TYPE,
    GLYPH_NUM_SUMMARY_TOKENS,
    GLYPH_MIN_SUMMARY_TOKENS,
    GLYPH_QUALITY_MIN_AREA,
    GLYPH_QUALITY_MIN_WIDTH,
    GLYPH_QUALITY_MIN_HEIGHT,
    GLYPH_QUALITY_MAX_ASPECT_RATIO,
    OCR_GLYPH_CONFIDENCE_THRESHOLD,
    GLYPH_BRANCH_ENABLE_POS_ENCODING,
    GLYPH_SUMMARIZER_CROSS_ATTN_LAYERS,
    GLYPH_SUMMARIZER_ATTENTION_HEADS,
    DROPOUT,
    GLYPH_BRANCH_USE_FOURIER_POS,
    GLYPH_BRANCH_NUM_FREQS,
    NUM_GLYPH_CLASSES,
    USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
)


def _validate_glyph_mask_shape(
    mask: torch.Tensor,
    expected_shape: torch.Size | tuple,
    context: str,
    name: str = "valid_mask",
) -> None:
    if mask.ndim != 2 or tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"{context}: {name} must have shape {tuple(expected_shape)} to match the provided glyphs/tokens; "
            f"got {tuple(mask.shape)}."
        )


def _validate_glyph_coords_shape(
    coords: torch.Tensor,
    expected_shape: torch.Size | tuple,
    context: str,
) -> None:
    expected = tuple(expected_shape) + (4,)
    if coords.ndim != 3 or tuple(coords.shape) != expected:
        raise ValueError(
            f"{context}: glyph_coords must have shape {expected} to match the provided glyphs/tokens; "
            f"got {tuple(coords.shape)}."
        )


def _validate_integer_tensor(tensor: torch.Tensor, context: str, name: str) -> None:
    if tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        return
    raise ValueError(f"{context}: {name} must contain integer IDs; got dtype {tensor.dtype}.")


def _validate_char_class_id_range(
    char_class_ids: torch.Tensor,
    num_char_classes: int,
    context: str,
) -> None:
    if char_class_ids.numel() == 0:
        return
    min_id = int(char_class_ids.min().item())
    max_id = int(char_class_ids.max().item())
    if min_id < 0 or max_id >= int(num_char_classes):
        raise ValueError(
            f"{context}: char_class_ids must be in [0, {int(num_char_classes) - 1}]; "
            f"got min={min_id}, max={max_id}."
        )


class GlyphHardQualityFilter:
    """
    Hard quality filter for glyphs.
    
    Filters out:
    - Tiny glyphs (area < min_area)
    - Blank/empty glyphs (ink_fraction from extraction, or low variance fallback)
    - Extreme aspect ratios (too wide or too tall)
    
    Also provides diverse sampling: randomly samples glyphs preferring different characters
    with high confidence, regardless of spatial location.
    """
    def __init__(
        self,
        min_area: int = GLYPH_QUALITY_MIN_AREA,
        min_width: int = GLYPH_QUALITY_MIN_WIDTH,
        min_height: int = GLYPH_QUALITY_MIN_HEIGHT,
        max_aspect_ratio: float = GLYPH_QUALITY_MAX_ASPECT_RATIO,
        min_confidence: float = OCR_GLYPH_CONFIDENCE_THRESHOLD,
    ):
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.max_aspect_ratio = max_aspect_ratio
        self.min_confidence = min_confidence
    
    def filter_glyphs(
        self,
        glyph_patches: List[torch.Tensor],  # List of [3, H, W] patches
        glyph_metadata: List[Dict],  # List of metadata dicts with 'width', 'height', 'gc', 'char', etc.
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        Filter glyphs based on quality criteria (size, aspect ratio, confidence, blank detection).
        
        Args:
            glyph_patches: List of glyph patch tensors [3, H, W]
            glyph_metadata: List of metadata dicts with keys: 'width', 'height', 'gc', 'char', etc.
            
        Returns:
            Tuple of (filtered_patches, filtered_metadata)
        """
        if not glyph_patches:
            return [], []
        
        filtered_patches = []
        filtered_metadata = []
        
        for patch, meta in zip(glyph_patches, glyph_metadata):
            # Extract dimensions
            width = meta.get('width', 0.0)
            height = meta.get('height', 0.0)
            area = width * height
            confidence = meta.get('gc', 0.0)
            
            # Filter by area
            if area < self.min_area:
                continue
            
            # Filter by minimum dimensions
            if width < self.min_width or height < self.min_height:
                continue
            
            # Filter by aspect ratio
            if width > 0 and height > 0:
                aspect = max(width / height, height / width)
                if aspect > self.max_aspect_ratio:
                    continue
            
            # Filter by confidence
            if confidence < self.min_confidence:
                continue
            
            # Blank detection: square-padded glyph crops have low global variance even when
            # ink is present (parchment fill dominates). Prefer ink_fraction from extraction.
            ink_frac = meta.get("ink_fraction")
            if ink_frac is not None:
                if float(ink_frac) <= 0.0:
                    continue
            else:
                if patch.shape[0] == 3:  # RGB
                    patch_gray = patch.mean(dim=0)  # [H, W]
                else:
                    patch_gray = patch.squeeze(0) if patch.dim() == 3 else patch
                if patch_gray.var().item() < 0.01:
                    continue
            
            filtered_patches.append(patch)
            filtered_metadata.append(meta)
        
        return filtered_patches, filtered_metadata
    
    @staticmethod
    def sample_diverse_glyphs(
        glyph_patches: List[torch.Tensor],
        glyph_metadata: List[Dict],
        max_glyphs: int,
        prefer_diverse_chars: bool = True,
        min_confidence: float = 0.0,
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        Sample diverse glyphs, preferring different characters with high confidence.
        
        Strategy:
        1. Group glyphs by character
        2. Sort groups by average confidence (highest first)
        3. Sample from each group, prioritizing high-confidence glyphs
        4. If prefer_diverse_chars=True, try to get at least one glyph per character
        5. Fill remaining slots with highest confidence glyphs regardless of character
        
        Args:
            glyph_patches: List of glyph patch tensors [3, H, W]
            glyph_metadata: List of metadata dicts with 'char', 'gc', etc.
            max_glyphs: Maximum number of glyphs to sample
            prefer_diverse_chars: If True, prefer sampling different characters
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (sampled_patches, sampled_metadata)
        """
        if not glyph_patches or max_glyphs <= 0:
            return [], []
        
        # Filter by minimum confidence first
        valid_indices = [
            i for i, meta in enumerate(glyph_metadata)
            if meta.get('gc', 0.0) >= min_confidence
        ]
        
        if not valid_indices:
            return [], []
        
        if len(valid_indices) <= max_glyphs:
            # Return all valid glyphs
            return (
                [glyph_patches[i] for i in valid_indices],
                [glyph_metadata[i] for i in valid_indices]
            )
        
        if not prefer_diverse_chars:
            # Simple: sort by confidence and take top N
            sorted_indices = sorted(
                valid_indices,
                key=lambda i: glyph_metadata[i].get('gc', 0.0),
                reverse=True
            )
            selected_indices = sorted_indices[:max_glyphs]
        else:
            # Group by character
            char_groups: Dict[str, List[int]] = {}
            for i in valid_indices:
                char = glyph_metadata[i].get('char', '')
                if char not in char_groups:
                    char_groups[char] = []
                char_groups[char].append(i)
            
            # Sort glyphs within each group by confidence
            for char in char_groups:
                char_groups[char].sort(
                    key=lambda i: glyph_metadata[i].get('gc', 0.0),
                    reverse=True
                )
            
            # Sort groups by average confidence (highest first)
            group_avg_conf = {
                char: sum(glyph_metadata[i].get('gc', 0.0) for i in indices) / len(indices)
                for char, indices in char_groups.items()
            }
            sorted_chars = sorted(
                char_groups.keys(),
                key=lambda c: group_avg_conf[c],
                reverse=True
            )
            
            # Sample strategy: Round-Robin across character groups to ensure balance
            selected_indices = []
            
            # Keep track of which glyph index from each group we'll take next
            group_pointers = {char: 0 for char in sorted_chars}
            
            # Loop until we have enough glyphs or we've exhausted all available ones
            while len(selected_indices) < max_glyphs:
                glyphs_added_this_round = 0
                for char in sorted_chars:
                    if len(selected_indices) >= max_glyphs:
                        break
                    
                    ptr = group_pointers[char]
                    if ptr < len(char_groups[char]):
                        selected_indices.append(char_groups[char][ptr])
                        group_pointers[char] += 1
                        glyphs_added_this_round += 1
                
                # If we went through all characters and didn't add any new glyphs, we're done
                if glyphs_added_this_round == 0:
                    break
        
        # Return in order of selection (which prioritizes diversity and confidence)
        return (
            [glyph_patches[i] for i in selected_indices],
            [glyph_metadata[i] for i in selected_indices]
        )


class GlyphVisualEncoder(nn.Module):
    """
    Glyph Visual Encoder using ConvNeXt-Tiny or Swin-T.
    
    Processes glyph patches: [B, M, 3, H, W] -> [B, M, Dg]
    """
    def __init__(
        self,
        char_patch_size: int = CHAR_PATCH_SIZE,
        encoder_type: str = GLYPH_ENCODER_TYPE,
        d_model: int = D_MODEL,
        pretrained: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.encoder_type = encoder_type
        
        # Create encoder
        if encoder_type == "convnext_tiny":
            # ConvNeXt doesn't use img_size parameter, it adapts to input size
            self.encoder = timm.create_model(
                "convnext_tiny",
                pretrained=pretrained,
                num_classes=0,
            )
        elif encoder_type == "swin_tiny":
            # Swin-T accepts img_size
            self.encoder = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=pretrained,
                num_classes=0,
                img_size=char_patch_size,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # Get encoder feature dimension
        self.glyph_feat_dim = getattr(self.encoder, 'num_features', None)
        if self.glyph_feat_dim is None:
            self.glyph_feat_dim = getattr(self.encoder, 'embed_dim', 768)
        
        # Projection to d_model
        if self.glyph_feat_dim != d_model:
            self.proj = nn.Linear(self.glyph_feat_dim, d_model)
        else:
            self.proj = nn.Identity()
    
    def forward(
        self,
        glyph_patches: torch.Tensor,
        valid_mask: torch.Tensor,
        debug_paths: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Encode glyph patches.
        
        Args:
            glyph_patches: [B, M, 3, H, W] - glyph patches
            valid_mask: [B, M] (bool) - True for valid glyphs, False for padding
            
        Returns:
            glyph_features: [B, M, d_model] - encoded glyph features
        """
        B, M, C, H, W = glyph_patches.shape
        device = glyph_patches.device

        def _debug_header() -> str:
            # Keep this lightweight; it only runs on failure paths.
            try:
                vp = valid_mask
                if vp is None:
                    vp_shape = None
                    vp_dtype = None
                    vp_device = None
                    vp_contig = None
                    vp_strides = None
                    vp_sum = None
                    vp_sums = None
                else:
                    vp_shape = tuple(vp.shape)
                    vp_dtype = str(vp.dtype)
                    vp_device = str(vp.device)
                    vp_contig = bool(vp.is_contiguous())
                    vp_strides = tuple(vp.stride())
                    # Avoid huge prints; show aggregates and first few per-sample counts when possible.
                    vp_sum = int(vp.sum().item()) if vp.dtype != torch.bool or vp.ndim >= 1 else int(vp.sum().item())
                    if vp.ndim == 2:
                        per_sample = vp.sum(dim=1).detach().cpu().tolist()
                        vp_sums = per_sample[: min(8, len(per_sample))]
                    else:
                        vp_sums = None
            except Exception as e:
                vp_shape = vp_dtype = vp_device = vp_contig = vp_strides = vp_sum = vp_sums = f"<error: {e}>"

            gp_shape = tuple(glyph_patches.shape)
            gp_dtype = str(glyph_patches.dtype)
            gp_device = str(glyph_patches.device)
            gp_contig = bool(glyph_patches.is_contiguous())
            gp_strides = tuple(glyph_patches.stride())

            # Paths are not tensors; in DataParallel they get scattered per-replica as lists/tuples.
            # Only show a short preview to avoid huge logs.
            try:
                if debug_paths is None:
                    paths_info = "None"
                else:
                    dp = list(debug_paths)
                    preview = dp[: min(8, len(dp))]
                    paths_info = f"len={len(dp)}, first8={preview}"
            except Exception as e:
                paths_info = f"<error reading debug_paths: {e}>"

            return (
                "[GlyphVisualEncoder DEBUG]\n"
                f"- glyph_patches: shape={gp_shape}, dtype={gp_dtype}, device={gp_device}, contiguous={gp_contig}, strides={gp_strides}\n"
                f"- valid_mask: shape={vp_shape}, dtype={vp_dtype}, device={vp_device}, contiguous={vp_contig}, strides={vp_strides}, "
                f"sum={vp_sum}, per_sample_sum(first8)={vp_sums}\n"
                f"- paths: {paths_info}\n"
                f"- expected valid_mask shape: ({B}, {M})\n"
            )

        # If no mask is provided, treat all glyph slots as valid.
        if valid_mask is None:
            valid_mask = torch.ones(B, M, dtype=torch.bool, device=device)
        else:
            # FAIL-FAST: upstream code must provide a correctly shaped boolean mask on the same device.
            # Silent reshapes/truncation can hide the root cause and lead to CUDA illegal memory access.
            if valid_mask.device != device:
                raise RuntimeError(
                    "GlyphVisualEncoder: valid_mask is on a different device than glyph_patches.\n"
                    + _debug_header()
                )
            if valid_mask.dtype != torch.bool:
                raise RuntimeError(
                    "GlyphVisualEncoder: valid_mask must be dtype=bool.\n"
                    + _debug_header()
                )
            if valid_mask.ndim != 2:
                raise RuntimeError(
                    "GlyphVisualEncoder: valid_mask must have shape [B, M].\n"
                    + _debug_header()
                )
            if valid_mask.shape[0] != B or valid_mask.shape[1] != M:
                raise RuntimeError(
                    "GlyphVisualEncoder: valid_mask shape mismatch (this can cause out-of-bounds GPU indexing).\n"
                    + _debug_header()
                )

        glyph_patches_flat = glyph_patches.reshape(B * M, C, H, W)
        valid_mask_flat = valid_mask.reshape(B * M)
        
        # Process in chunks to control memory
        # Only allocate for valid glyphs to save memory
        if valid_mask_flat.any():
            valid_idx = torch.nonzero(valid_mask_flat, as_tuple=False).squeeze(1)
            num_valid = valid_idx.numel()

            # Fail-fast: if this ever happens, something upstream corrupted mask alignment.
            if num_valid > 0:
                max_idx = int(valid_idx.max().item())
                if max_idx >= (B * M):
                    offender_sample = max_idx // M if M > 0 else None
                    offender_glyph = max_idx % M if M > 0 else None
                    offender_path = None
                    try:
                        if debug_paths is not None and offender_sample is not None:
                            dp = list(debug_paths)
                            if 0 <= offender_sample < len(dp):
                                offender_path = dp[offender_sample]
                    except Exception:
                        offender_path = None
                    raise RuntimeError(
                        "GlyphVisualEncoder: valid_idx contains out-of-bounds indices for flattened glyph_patches.\n"
                        + _debug_header()
                        + f"- flattened_length(B*M)={B*M}, max_valid_idx={max_idx}\n"
                        + f"- inferred offender: sample_idx={offender_sample}, glyph_idx={offender_glyph}, path={offender_path}\n"
                    )
            
            # Pre-allocate output tensor for all positions (B*M) but we'll only fill valid ones
            # Note: We still need B*M size for reshaping, but we can optimize by processing
            # only valid glyphs and zeroing padding later
            glyph_feats = torch.zeros(B * M, self.glyph_feat_dim, device=glyph_patches.device, dtype=glyph_patches.dtype)
            
            # Process in chunks to control memory usage
            # With DataParallel, each GPU processes a split batch (not full batch),
            # but the model is replicated on each GPU, increasing memory usage.
            # Use smaller chunks to avoid OOM, especially with ConvNeXt-Tiny.
            chunk_size = min(CHAR_ENCODE_CHUNK_SIZE, num_valid)
            for start in range(0, num_valid, chunk_size):
                end = min(start + chunk_size, num_valid)
                idx_chunk = valid_idx[start:end]
                # Keep 1D so encoder gets [chunk_len, C, H, W]; single-element squeeze can become 0-dim.
                if idx_chunk.dim() == 0:
                    idx_chunk = idx_chunk.unsqueeze(0)
                
                # Process chunk through encoder
                encoded = self.encoder(glyph_patches_flat[idx_chunk].contiguous())  # [chunk_len, glyph_feat_dim]
                glyph_feats[idx_chunk] = encoded
                del encoded
        else:
            # No valid glyphs, return zeros
            glyph_feats = torch.zeros(B * M, self.glyph_feat_dim, device=glyph_patches.device, dtype=glyph_patches.dtype)
        
        # Project to d_model
        glyph_feats = self.proj(glyph_feats)  # [B*M, d_model]
        
        # Reshape back to [B, M, d_model]
        glyph_feats = glyph_feats.reshape(B, M, self.d_model)
        
        # Zero out padded positions
        glyph_feats = glyph_feats * valid_mask.unsqueeze(-1).to(glyph_feats.dtype)
        
        return glyph_feats  # [B, M, d_model]


class PositionalEncoding4D(nn.Module):
    """
    4D Positional encoding for glyph bounding boxes.
    
    Maps (x, y, w, h) coordinates to d_model-dimensional embeddings.
    
    For RTL reading order: transforms x-coordinate for positional encoding purposes
    so that large x (right side) is treated as "earlier" in reading order.
    Actual coordinates remain unchanged - only the positional encoding interpretation changes.
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        use_fourier: bool = GLYPH_BRANCH_USE_FOURIER_POS,
        num_freqs: int = GLYPH_BRANCH_NUM_FREQS,
        rtl: bool = XML_PATCH_READING_DIRECTION_RTL,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_fourier = use_fourier
        self.rtl = rtl
        
        if use_fourier:
            # Fourier features for 4D coordinates
            self.num_freqs = num_freqs
            freqs = torch.linspace(0.0, 1.0, num_freqs)
            self.register_buffer('freqs', freqs)
            # Project Fourier features to d_model
            fourier_dim = 4 * 2 * num_freqs  # (x,y,w,h) * (sin,cos) * num_freqs
            self.fourier_proj = nn.Linear(fourier_dim, d_model)
        else:
            # Simple MLP: (x, y, w, h) -> d_model
            self.mlp = nn.Sequential(
                nn.Linear(4, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode 4D coordinates to d_model-dimensional embeddings.
        
        Args:
            coords: [B, M, 4] - normalized coordinates (x, y, w, h)
              - x: [0, 1] (left=0, right=1) - actual spatial position
              - y: [0, 1] for single-page, [0, 2] for two-page (second page: y += 1.0)
              - w, h: [0, 1] (normalized width/height)
            
        Returns:
            pos_emb: [B, M, d_model] - positional embeddings
            
        Note: For RTL, x-coordinate is transformed for positional encoding so that
        large x (right side) is treated as "earlier" in reading order. This only
        affects the positional encoding, not the actual coordinates.
        """
        # For RTL reading order, transform x-coordinate for positional encoding
        # Large x (right side) should be treated as "earlier" in reading order
        coords_encoded = coords.clone()
        if self.rtl:
            coords_encoded[..., 0] = 1.0 - coords_encoded[..., 0]
        
        # For two-page manuscripts, y can be in [0, 2] (second page: y in [1, 2])
        # The encoding handles this naturally - y values > 1.0 represent the second page
        
        if self.use_fourier:
            # Fourier features for each coordinate (using transformed coordinates for RTL)
            x, y, w, h = coords_encoded[..., 0:1], coords_encoded[..., 1:2], coords_encoded[..., 2:3], coords_encoded[..., 3:4]
            
            # Generate sin/cos features
            x_sin = torch.sin(2 * math.pi * self.freqs * x)  # [B, M, num_freqs]
            x_cos = torch.cos(2 * math.pi * self.freqs * x)
            y_sin = torch.sin(2 * math.pi * self.freqs * y)
            y_cos = torch.cos(2 * math.pi * self.freqs * y)
            w_sin = torch.sin(2 * math.pi * self.freqs * w)
            w_cos = torch.cos(2 * math.pi * self.freqs * w)
            h_sin = torch.sin(2 * math.pi * self.freqs * h)
            h_cos = torch.cos(2 * math.pi * self.freqs * h)
            
            # Concatenate: [B, M, 8*num_freqs]
            fourier_feat = torch.cat([x_sin, x_cos, y_sin, y_cos, w_sin, w_cos, h_sin, h_cos], dim=-1)
            
            # Project to d_model
            pos_emb = self.fourier_proj(fourier_feat)  # [B, M, d_model]
        else:
            # MLP-based encoding (using transformed coordinates for RTL)
            pos_emb = self.mlp(coords_encoded)  # [B, M, d_model]
        
        return pos_emb


class GlyphTokenEnrichment(nn.Module):
    """
    Glyph Token Enrichment Module.
    
    Enriches glyph visual features with:
    1. Projection: g_tok = proj(g_feat)
    2. Positional embedding: + pos_glyph = f([x, y, w, h]) with RTL support
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        enable_pos_encoding: bool = GLYPH_BRANCH_ENABLE_POS_ENCODING,
        use_fourier_pos: bool = GLYPH_BRANCH_USE_FOURIER_POS,
        num_freqs: int = GLYPH_BRANCH_NUM_FREQS,
        rtl: bool = XML_PATCH_READING_DIRECTION_RTL,
        num_char_classes: int = NUM_GLYPH_CLASSES,
    ):
        super().__init__()
        self.d_model = d_model
        self.enable_pos_encoding = enable_pos_encoding
        
        # Projection layer
        self.proj = nn.Linear(d_model, d_model)

        # Kept for checkpoint compatibility, but modality-type embeddings
        # are applied in the fusion module.
        self.type_embed = nn.Embedding(1, d_model)
        
        # Learned character-class embedding (which Hebrew letter this glyph is)
        self.num_char_classes = int(num_char_classes)
        self.char_class_embed = nn.Embedding(num_char_classes, d_model)
        nn.init.trunc_normal_(self.char_class_embed.weight, std=0.02)
        
        # Positional encoding for glyph coordinates (x, y, w, h)
        # RTL-aware: transforms x for encoding so large x (right) is treated as "earlier"
        self.pos_encoder = PositionalEncoding4D(
            d_model=d_model,
            use_fourier=use_fourier_pos,
            num_freqs=num_freqs,
            rtl=rtl,
        )
    
    def forward(
        self,
        glyph_features: torch.Tensor,
        glyph_coords: Optional[torch.Tensor],  # [B, M, 4] - (x, y, w, h) normalized
        valid_mask: torch.Tensor = None,
        char_class_ids: Optional[torch.Tensor] = None,  # [B, M] long
    ) -> torch.Tensor:
        """
        Enrich glyph tokens with type, character-class, and (optional) positional embeddings.
        
        Args:
            glyph_features: [B, M, d_model] - visual features from GlyphVisualEncoder
            glyph_coords: [B, M, 4] - normalized glyph coordinates (x, y, w, h) in [0, 1]
            valid_mask: [B, M] (bool) - True for valid glyphs, False for padding
            char_class_ids: [B, M] long (optional) - Hebrew letter class index per glyph
            
        Returns:
            enriched_tokens: [B, M, d_model] - enriched glyph tokens
        """
        if glyph_coords is None and self.enable_pos_encoding:
            raise ValueError("GlyphTokenEnrichment requires glyph_coords when glyph positional encoding is enabled.")

        if valid_mask is not None:
            _validate_glyph_mask_shape(valid_mask, glyph_features.shape[:2], "GlyphTokenEnrichment")
            valid_mask = valid_mask.to(device=glyph_features.device, dtype=torch.bool)

        if glyph_coords is not None:
            _validate_glyph_coords_shape(glyph_coords, glyph_features.shape[:2], "GlyphTokenEnrichment")
            glyph_coords = glyph_coords.to(device=glyph_features.device, dtype=glyph_features.dtype)

        if char_class_ids is not None:
            _validate_glyph_mask_shape(char_class_ids, glyph_features.shape[:2], "GlyphTokenEnrichment", name="char_class_ids")
            _validate_integer_tensor(char_class_ids, "GlyphTokenEnrichment", "char_class_ids")
            char_class_ids = char_class_ids.to(device=glyph_features.device, dtype=torch.long)
            _validate_char_class_id_range(
                char_class_ids,
                self.num_char_classes,
                "GlyphTokenEnrichment",
            )

        # 1. Project visual features
        g_tok = self.proj(glyph_features)  # [B, M, d_model]

        # 2. Add character-class embedding (which Hebrew letter this glyph is)
        if char_class_ids is not None:
            char_emb = self.char_class_embed(char_class_ids)  # [B, M, d_model]
            g_tok = g_tok + char_emb

        # 3. Add positional embedding (optional — disabled when position is irrelevant)
        if self.enable_pos_encoding:
            pos_glyph = self.pos_encoder(glyph_coords)  # [B, M, d_model]
            g_tok = g_tok + pos_glyph  # [B, M, d_model]
        
        # Zero out padded positions
        if valid_mask is not None:
            g_tok = g_tok * valid_mask.unsqueeze(-1).to(g_tok.dtype)
        
        return g_tok  # [B, M, d_model]


class GlyphSetSummarizer(nn.Module):
    """
    Glyph Set Summarizer using Cross-Attention.
    
    Uses learnable queries to summarize variable-length glyph token sequences
    into a fixed-size representation.

    Every learned query attends to the complete valid glyph set. All query
    outputs remain active for a nonempty modality; samples with no valid glyphs
    receive zero outputs and an all-False summary mask.
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        num_queries: int = GLYPH_NUM_SUMMARY_TOKENS,
        num_heads: int = GLYPH_SUMMARIZER_ATTENTION_HEADS,
        num_cross_attn_layers: int = GLYPH_SUMMARIZER_CROSS_ATTN_LAYERS,
        dropout: float = DROPOUT,
        min_active_queries: int = GLYPH_MIN_SUMMARY_TOKENS,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_cross_attn_layers = int(num_cross_attn_layers)
        # Retained for checkpoint/API compatibility. Query validity is no
        # longer tied to source-token count.
        self.min_active_queries = max(1, min(int(min_active_queries), num_queries))
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(self.num_cross_attn_layers)
        ])
        self.query_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(self.num_cross_attn_layers)
        ])
        self.glyph_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(self.num_cross_attn_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def _build_summary_valid_mask(
        self,
        valid_mask: torch.Tensor,  # [B, M] bool
    ) -> torch.Tensor:
        """Keep every learned query when the glyph modality has evidence."""
        has_evidence = valid_mask.any(dim=1, keepdim=True)
        return has_evidence.expand(-1, self.num_queries).clone()

    def forward(
        self,
        glyph_tokens: torch.Tensor,  # [B, M, d_model]
        valid_mask: torch.Tensor,  # [B, M] (bool) - True for valid, False for padding
        return_attention: bool = False,
    ):
        """
        Summarize glyph tokens into fixed-size representation.
        
        Args:
            glyph_tokens: [B, M, d_model] - enriched glyph tokens
            valid_mask: [B, M] (bool) - True for valid glyphs, False for padding
            return_attention: If True, also return raw query->glyph attention.
            
        Returns:
            glyph_sum_tokens: [B, num_queries, d_model] - summary tokens
            summary_valid_mask: [B, num_queries] - all True for a nonempty
                glyph modality, all False when the modality has no valid glyphs
            attention_weights (only if return_attention=True): [B, num_queries, M]
        """
        B, M = glyph_tokens.shape[0], glyph_tokens.shape[1]
        device = glyph_tokens.device
        _validate_glyph_mask_shape(valid_mask, (B, M), "GlyphSetSummarizer")
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

        # Build per-sample summary valid mask BEFORE any early returns.
        if M == 0:
            summary_valid_mask = torch.zeros(B, self.num_queries, dtype=torch.bool, device=device)
        else:
            summary_valid_mask = self._build_summary_valid_mask(valid_mask)

        # If there are zero glyph tokens (e.g., missing XML everywhere for this sample),
        # return zeros rather than calling attention with M=0.
        if M == 0:
            out = torch.zeros(B, self.num_queries, self.d_model, device=device, dtype=glyph_tokens.dtype)
            if return_attention:
                attn = torch.zeros(B, self.num_queries, 0, device=device, dtype=glyph_tokens.dtype)
                return out, summary_valid_mask, attn
            return out, summary_valid_mask
        
        # Expand query tokens to batch size
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_queries, d_model]
        
        # If some samples have zero valid glyphs (common when XML is missing/empty),
        # do NOT run attention with all keys masked (can yield NaNs). Instead:
        # - return zero summary tokens for those samples
        # - return zero attention weights for those samples
        any_valid = valid_mask.any(dim=1)  # [B]
        
        # Cross-attention: queries attend to glyph_tokens
        # queries are the "target", glyph_tokens are the "source"
        # Pre-allocate outputs (fixed shapes)
        glyph_sum = torch.zeros(B, self.num_queries, self.d_model, device=device, dtype=glyph_tokens.dtype)
        attn_weights_out = None
        if return_attention:
            attn_weights_out = torch.zeros(B, self.num_queries, M, device=device, dtype=glyph_tokens.dtype)

        if any_valid.any():
            idx = torch.nonzero(any_valid, as_tuple=False).squeeze(1)
            glyph_tokens_v = glyph_tokens.index_select(0, idx)  # [Bv, M, d_model]
            valid_mask_v = valid_mask.index_select(0, idx)  # [Bv, M]
            queries_v = queries.index_select(0, idx)  # [Bv, Q, d_model]

            # Create attention mask (True = ignore, False = attend)
            attn_mask_v = ~valid_mask_v  # [Bv, M]

            attn_weights = None
            for layer_idx, (cross_attn, query_norm, glyph_norm) in enumerate(
                zip(self.cross_attn_layers, self.query_norms, self.glyph_norms)
            ):
                is_last = layer_idx == len(self.cross_attn_layers) - 1
                queries_norm = query_norm(queries_v)
                glyphs_norm = glyph_norm(glyph_tokens_v)
                queries_attn, layer_attn = cross_attn(
                    query=queries_norm,
                    key=glyphs_norm,
                    value=glyphs_norm,
                    key_padding_mask=attn_mask_v,
                    average_attn_weights=not is_last,
                )
                queries_v = queries_v + queries_attn
                if is_last:
                    attn_weights = layer_attn

            # Average attention over heads if multi-head.
            #
            # PyTorch returns different layouts depending on version/config:
            # - Common (batch_first=True): [B, num_heads, Q, M]
            # - Legacy:                 [num_heads, B, Q, M]
            if attn_weights is not None and attn_weights.dim() == 4:
                if attn_weights.shape[0] == queries_v.shape[0]:
                    attn_weights = attn_weights.mean(dim=1)  # [Bv, Q, M]
                else:
                    attn_weights = attn_weights.mean(dim=0)  # [Bv, Q, M]

            glyph_sum_v = self.out_norm(queries_v)
            glyph_sum.index_copy_(0, idx, glyph_sum_v)
            if return_attention and attn_weights_out is not None and attn_weights is not None:
                attn_weights_out.index_copy_(0, idx, attn_weights)
        
        if return_attention:
            return glyph_sum, summary_valid_mask, attn_weights_out  # [B, Q, D], [B, Q], [B, Q, M]
        return glyph_sum, summary_valid_mask  # [B, Q, D], [B, Q]


def apply_page_segment_to_glyph_coords(
    coords: torch.Tensor,
    page_segments: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply page_segment shift to y-coordinates for two-page manuscripts.
    
    For two-page manuscripts:
    - page_segment=0: first page, y stays in [0, 1]
    - page_segment=1: second page, y is shifted by +1.0 (y in [1, 2])
    
    Args:
        coords: [B, M, 4] - glyph coordinates (x, y, w, h)
        page_segments: [B, M] (long) - page segment IDs (0 or 1), None if single-page
        
    Returns:
        coords: [B, M, 4] - coordinates with y shifted for second page
    """
    if page_segments is None:
        return coords

    _validate_glyph_coords_shape(coords, coords.shape[:2], "apply_page_segment_to_glyph_coords")
    _validate_glyph_mask_shape(page_segments, coords.shape[:2], "apply_page_segment_to_glyph_coords", name="page_segments")
    page_segments = page_segments.to(device=coords.device, dtype=torch.long)
    
    coords = coords.clone()
    # Shift y-coordinate by +1.0 for second page (page_segment=1)
    second_page_mask = (page_segments == 1)  # [B, M]
    coords[..., 1] = coords[..., 1] + second_page_mask.to(coords.dtype)  # [B, M]
    
    return coords


class GlyphBranch(nn.Module):
    """
    Complete Glyph Branch: Quality Filter + Visual Encoder + Token Enrichment.
    
    This is the full glyph processing pipeline according to the new architecture:
    1. ALTO XML → glyph boxes + conf
    2. GLYPH HARD QUALITY FILTER - drop tiny/blank/extreme aspect
    3. GLYPH VISUAL ENCODER - ConvNeXt-Tiny / Swin-T → [B, M, d_model]
    4. GLYPH TOKEN ENRICHMENT - proj + pos_emb (with RTL support) → [B, M, d_model]
    5. Return enriched glyph tokens directly when summarization is disabled
    
    Supports two-page manuscripts: y-coordinates can be in [0, 2] (second page: y += 1.0)
    """
    def __init__(
        self,
        char_patch_size: int = CHAR_PATCH_SIZE,
        encoder_type: str = GLYPH_ENCODER_TYPE,
        d_model: int = D_MODEL,
        pretrained: bool = True,
        num_char_classes: int = NUM_GLYPH_CLASSES,
        num_summary_tokens: int = GLYPH_NUM_SUMMARY_TOKENS,
        min_summary_tokens: int = GLYPH_MIN_SUMMARY_TOKENS,
        enable_summarizer: bool = USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
        use_fourier_pos: bool = GLYPH_BRANCH_USE_FOURIER_POS,
        num_freqs: int = GLYPH_BRANCH_NUM_FREQS,
        rtl: bool = XML_PATCH_READING_DIRECTION_RTL,
        # Quality filter parameters
        min_area: int = GLYPH_QUALITY_MIN_AREA,
        min_width: int = GLYPH_QUALITY_MIN_WIDTH,
        min_height: int = GLYPH_QUALITY_MIN_HEIGHT,
        max_aspect_ratio: float = GLYPH_QUALITY_MAX_ASPECT_RATIO,
        min_confidence: float = OCR_GLYPH_CONFIDENCE_THRESHOLD,
    ):
        super().__init__()
        
        # Quality filter (non-trainable)
        self.quality_filter = GlyphHardQualityFilter(
            min_area=min_area,
            min_width=min_width,
            min_height=min_height,
            max_aspect_ratio=max_aspect_ratio,
            min_confidence=min_confidence,
        )
        
        # Visual encoder
        self.visual_encoder = GlyphVisualEncoder(
            char_patch_size=char_patch_size,
            encoder_type=encoder_type,
            d_model=d_model,
            pretrained=pretrained,
        )
        
        # Token enrichment
        self.token_enrichment = GlyphTokenEnrichment(
            d_model=d_model,
            use_fourier_pos=use_fourier_pos,
            num_freqs=num_freqs,
            rtl=rtl,
            num_char_classes=num_char_classes,
        )
        
        self.set_summarizer = (
            GlyphSetSummarizer(
                d_model=d_model,
                num_queries=num_summary_tokens,
                min_active_queries=min_summary_tokens,
            )
            if bool(enable_summarizer) and int(num_summary_tokens) > 0
            else None
        )
    
    def forward(
        self,
        glyph_patches: torch.Tensor,  # [B, M, 3, H, W]
        glyph_coords: torch.Tensor,  # [B, M, 4] - (x, y, w, h) normalized
        glyph_metadata: Optional[List[List[Dict]]] = None,  # Per-batch, per-glyph metadata
        valid_mask: torch.Tensor = None,  # [B, M] (bool)
        page_segments: Optional[torch.Tensor] = None,  # [B, M] (long) - page segment IDs
        image_sizes: Optional[List[Tuple[int, int]]] = None,  # Per-image (width, height)
        return_attention: bool = False,
        debug_paths: Optional[List[str]] = None,  # Per-image paths (for fail-fast debug)
        char_class_ids: Optional[torch.Tensor] = None,  # [B, M] long - Hebrew letter class IDs
    ):
        """
        Forward pass through the glyph branch.
        
        Args:
            glyph_patches: [B, M, 3, H, W] - glyph patches
            glyph_coords: [B, M, 4] - normalized glyph coordinates (x, y, w, h)
              - x: [0, 1]
              - y: [0, 1] for single-page, or can be in [0, 2] if page_segments provided
              - w, h: [0, 1]
            glyph_metadata: Optional list of per-image metadata lists (for quality filtering)
            valid_mask: [B, M] (bool) - True for valid glyphs, False for padding
            page_segments: [B, M] (long, optional) - page segment IDs (0 or 1) for two-page manuscripts
              If provided, y-coordinates will be shifted: page_segment=1 → y += 1.0
            image_sizes: Optional list of (width, height) tuples per image
            return_attention: If True, return optional summarizer attention
            
        Returns token embeddings and their valid mask. With summarization
        disabled, these are the enriched glyph tokens `[B, M, d_model]` and
        original mask `[B, M]`. If `return_attention=True`, the third return
        value is summarizer attention when available, otherwise `None`.
        """
        B, M = glyph_patches.shape[:2]
        device = glyph_patches.device
        
        if valid_mask is None:
            valid_mask = torch.ones(B, M, dtype=torch.bool, device=device)
        else:
            _validate_glyph_mask_shape(valid_mask, (B, M), "GlyphBranch")
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)

        if glyph_coords is None and self.token_enrichment.enable_pos_encoding:
            raise ValueError("GlyphBranch requires glyph_coords when glyph positional encoding is enabled.")
        if glyph_coords is not None:
            _validate_glyph_coords_shape(glyph_coords, (B, M), "GlyphBranch")
            glyph_coords = glyph_coords.to(device=device, dtype=glyph_patches.dtype)

        if char_class_ids is not None:
            _validate_glyph_mask_shape(char_class_ids, (B, M), "GlyphBranch", name="char_class_ids")
            _validate_integer_tensor(char_class_ids, "GlyphBranch", "char_class_ids")
            char_class_ids = char_class_ids.to(device=device, dtype=torch.long)
            _validate_char_class_id_range(
                char_class_ids,
                self.token_enrichment.num_char_classes,
                "GlyphBranch",
            )
        
        # Apply page_segment shift if provided
        if page_segments is not None:
            if glyph_coords is None:
                raise ValueError("GlyphBranch requires glyph_coords when page_segments are provided.")
            _validate_glyph_mask_shape(page_segments, (B, M), "GlyphBranch", name="page_segments")
            page_segments = page_segments.to(device=device, dtype=torch.long)
            glyph_coords = apply_page_segment_to_glyph_coords(glyph_coords, page_segments)
        
        # Note: Quality filtering is typically done at data loading time,
        # but we keep the filter here for optional runtime filtering.
        # For now, we assume glyphs are already filtered.
        
        # 1. Visual encoding
        glyph_features = self.visual_encoder(glyph_patches, valid_mask, debug_paths=debug_paths)  # [B, M, d_model]
        
        # 2. Token enrichment
        enriched_tokens = self.token_enrichment(
            glyph_features,
            glyph_coords,
            valid_mask=valid_mask,
            char_class_ids=char_class_ids,
        )  # [B, M, d_model]
        
        # 3. Optional set summarization. The active adapter-free symmetric path
        # returns the enriched per-glyph tokens for direct masked mean pooling.
        if self.set_summarizer is None:
            if return_attention:
                return enriched_tokens, valid_mask, None
            return enriched_tokens, valid_mask

        if return_attention:
            glyph_sum_tokens, summary_valid_mask, attn_weights = self.set_summarizer(
                enriched_tokens,
                valid_mask,
                return_attention=True,
            )
            return glyph_sum_tokens, summary_valid_mask, attn_weights
        glyph_sum_tokens, summary_valid_mask = self.set_summarizer(
            enriched_tokens,
            valid_mask,
        )
        return glyph_sum_tokens, summary_valid_mask
