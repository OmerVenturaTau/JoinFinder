from typing import Tuple, Optional, List
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import warnings
import logging
from utilities.augmentations.background_overlays import RandomLibraryBackgroundOverlay
from utilities.augmentations.manuscript_augmentations import (
    RandomZoomJitter,
    RandomTiltJitter,
    RandomLocalTexturePerturbation,
    RandomToneAndContrastJitter,
    RandomBorderMaskAndEdgeCrop,
    RandomWhiteBackground,
    WeightedOneOf,
)

# Set up logger for patch extraction
logger = logging.getLogger(__name__)

# Module-level collection for XML warnings (to avoid spam)
# Collected paths will be saved to a file after the first epoch
_xml_warning_paths = set()
_xml_warning_paths_lock = None  # Will be initialized if threading is needed

# Module-level tracking for batch label diversity
_batch_label_diversity_stats = {
    'total_batches': 0,
    'homogeneous_batches': 0,  # Batches where all labels are the same
    'low_diversity_batches': 0,  # Batches where unique labels < 30% of batch size
}
from system import (
    TILE_SIZE,
    TILE_STRIDE,
    CENTER_BIAS,
    DISABLE_PIL_LIMIT,
    DB_CONFIG_PATH,
    CHAR_PATCH_SIZE,
    MAX_CHARS_PER_IMAGE,
    MAX_WORDS_PER_IMAGE,
    USE_HEBREW_DICT_CHECK,
    USE_VISUAL_MOD,
    TILES_FALLBACK_GRID,
    USE_CHAR_MOD,
    USE_WORD_MOD,
    PATCH_LOADING_METHOD,
    CHAR_PATCH_APPLY_RANDAUGMENT,
    CHAR_PATCH_APPLY_IMAGENET_NORM,
    GLYPH_ONLY_MIDDLE_LETTERS,
    GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_GLYPH_CLASSES,
    AUGMENT_COLOR_JITTER,
    AUGMENT_APPLY_PROB,
    AUGMENT_COLOR_JITTER_PROB,
    AUGMENT_COLOR_JITTER_BRIGHTNESS,
    AUGMENT_COLOR_JITTER_CONTRAST,
    AUGMENT_COLOR_JITTER_SATURATION,
    AUGMENT_COLOR_JITTER_HUE,
    AUGMENT_RANDOM_GRAYSCALE_PROB,
    AUGMENT_GAUSSIAN_BLUR_PROB,
    AUGMENT_GAUSSIAN_BLUR_KERNEL,
    AUGMENT_BACKGROUND_PATTERN_PROB,
    AUGMENT_BACKGROUND_SURFACE_PATTERN_PROB,
    AUGMENT_BACKGROUND_LIBRARY_PATTERN_PROB,
    AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB,
    AUGMENT_BACKGROUND_SAMPLING_ALPHA,
    AUGMENT_BACKGROUND_SAMPLING_FLOOR,
    AUGMENT_BACKGROUND_SURFACE_PATTERN_TYPES,
    AUGMENT_BACKGROUND_LIBRARY_PATTERN_TYPES,
    AUGMENT_BACKGROUND_PATTERN_TYPES,
    AUGMENT_BACKGROUND_PATTERN_ALPHA_RANGE,
    AUGMENT_BACKGROUND_PATTERN_GRID_SPACING_RANGE,
    AUGMENT_BACKGROUND_PATTERN_GRID_LINE_WIDTH,
    AUGMENT_RANDOM_ERASING_PROB,
    AUGMENT_RANDOM_ERASING_SCALE,
    AUGMENT_RANDOM_ERASING_RATIO,
    GLYPH_GAUSSIAN_BLUR_KERNEL,
    GLYPH_RANDOM_ERASING_SCALE,
    GLYPH_AUG_COLOR_JITTER_PROB,
    GLYPH_AUG_APPLY_PROB,
    GLYPH_AUG_RANDOM_GRAYSCALE_PROB,
    GLYPH_AUG_GAUSSIAN_BLUR_PROB,
    GLYPH_AUG_BACKGROUND_RANDOM_LIBRARY_PROB,
    GLYPH_AUG_PARCHMENT_STAINS_PROB,
    GLYPH_AUG_WHITE_BACKGROUND_PROB,
    GLYPH_AUG_LOCAL_TEXTURE_PROB,
    GLYPH_AUG_TONE_CONTRAST_PROB,
    GLYPH_AUG_TILT_PROB,
    GLYPH_AUG_BORDER_CROP_PROB,
    GLYPH_AUG_ZOOM_PROB,
    GLYPH_AUG_RANDAUGMENT_PROB,
    AUGMENT_LOCAL_TEXTURE_PROB,
    AUGMENT_LOCAL_TEXTURE_ALPHA_RANGE,
    AUGMENT_TONE_CONTRAST_PROB,
    AUGMENT_BORDER_CROP_PROB,
    AUGMENT_BORDER_MAX_FRAC,
    AUGMENT_EDGE_CROP_MAX_FRAC,
    AUGMENT_ZOOM_PROB,
    AUGMENT_ZOOM_SCALE_RANGE,
    AUGMENT_TILT_PROB,
    AUGMENT_TILT_DEGREES,
    AUGMENT_RANDAUGMENT_PROB,
    GLYPH_INPUT_ALPHABET,
    GLYPHS_PER_CLASS,
    OCR_STRING_CONFIDENCE_THRESHOLD,
)
from utilities.VisionModule.xml_patch_extraction import extract_patches_with_xml
from utilities.VisionModule.xml_character_extraction import extract_character_patches
from utilities.ContextModule.xml_word_extraction import extract_words_from_alto
from utilities.xml_loader import find_xml_path_pretrain
from models.glyph_branch import GlyphHardQualityFilter
from train.db_loader import (
    get_db_connection, 
    load_patch_coordinates_from_db,
    PATCH_COORDINATES_TABLE_NAME
)

# Increase PIL image size limit to handle large manuscript images
if DISABLE_PIL_LIMIT:
    Image.MAX_IMAGE_PIXELS = None  # Disable the limit

def preprocess_before_patch(image):
    # Placeholder for future manipulations (e.g., denoising, cropping, etc.)
    return image


class ManuscriptDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        transform,
        label2idx,
        xml_paths=None,
        patch_size=TILE_SIZE,
        stride=TILE_STRIDE,
        to_rgb=True,
        max_tiles_per_image=None,
        center_bias=CENTER_BIAS,
        use_xml_extraction=None,
        use_db_coordinates=None,
        db_config_path=DB_CONFIG_PATH,
        split=None,
        patch_loading_method=None,
        use_visual_mod=None,
        use_char_mod=None,
        use_word_mod=None,
        glyph_input_alphabet=None,
        legacy_word_extraction_20260824=False,
    ):
        self.image_paths = image_paths
        self.labels = labels  # Store original labels
        self.xml_paths = xml_paths  # Store XML paths if provided
        self.transform = transform
        self.split = split
        self.use_visual_mod = USE_VISUAL_MOD if use_visual_mod is None else bool(use_visual_mod)
        self.use_char_mod = USE_CHAR_MOD if use_char_mod is None else bool(use_char_mod)
        self.use_word_mod = USE_WORD_MOD if use_word_mod is None else bool(use_word_mod)
        self.legacy_word_extraction_20260824 = bool(legacy_word_extraction_20260824)
        self.glyph_input_alphabet = (
            GLYPH_INPUT_ALPHABET
            if glyph_input_alphabet is None
            else glyph_input_alphabet
        )
        # Build character patch transform: color augment (train only) → ToTensor → Normalize.
        # Glyph-specific: smaller blur kernel and erasing scale than tiles, since
        # 128x128 crops are much smaller and character strokes are fragile.
        char_ops = []
        if split == "train":
            char_oneof_candidates = [
                (
                    "jitter",
                    transforms.ColorJitter(
                        brightness=AUGMENT_COLOR_JITTER_BRIGHTNESS,
                        contrast=AUGMENT_COLOR_JITTER_CONTRAST,
                        saturation=AUGMENT_COLOR_JITTER_SATURATION,
                        hue=AUGMENT_COLOR_JITTER_HUE,
                    ) if AUGMENT_COLOR_JITTER else None,
                    GLYPH_AUG_COLOR_JITTER_PROB,
                ),
                ("gray", transforms.RandomGrayscale(p=1.0), GLYPH_AUG_RANDOM_GRAYSCALE_PROB),
                ("blur", transforms.GaussianBlur(kernel_size=GLYPH_GAUSSIAN_BLUR_KERNEL), GLYPH_AUG_GAUSSIAN_BLUR_PROB),
                (
                    "random_library_background",
                    RandomLibraryBackgroundOverlay(
                        p=1.0,
                        pattern_types=("random_library_background",),
                        alpha_range=AUGMENT_BACKGROUND_PATTERN_ALPHA_RANGE,
                        grid_spacing_range=AUGMENT_BACKGROUND_PATTERN_GRID_SPACING_RANGE,
                        grid_line_width=AUGMENT_BACKGROUND_PATTERN_GRID_LINE_WIDTH,
                        allow_support_backing=True,
                        background_sampling_alpha=AUGMENT_BACKGROUND_SAMPLING_ALPHA,
                        background_sampling_floor=AUGMENT_BACKGROUND_SAMPLING_FLOOR,
                    ),
                    GLYPH_AUG_BACKGROUND_RANDOM_LIBRARY_PROB,
                ),
                (
                    "parchment_stains",
                    RandomLibraryBackgroundOverlay(
                        p=1.0,
                        pattern_types=("parchment_stains",),
                        alpha_range=AUGMENT_BACKGROUND_PATTERN_ALPHA_RANGE,
                        grid_spacing_range=AUGMENT_BACKGROUND_PATTERN_GRID_SPACING_RANGE,
                        grid_line_width=AUGMENT_BACKGROUND_PATTERN_GRID_LINE_WIDTH,
                        allow_support_backing=True,
                        background_sampling_alpha=AUGMENT_BACKGROUND_SAMPLING_ALPHA,
                        background_sampling_floor=AUGMENT_BACKGROUND_SAMPLING_FLOOR,
                    ),
                    GLYPH_AUG_PARCHMENT_STAINS_PROB,
                ),
                ("white_background", RandomWhiteBackground(p=1.0), GLYPH_AUG_WHITE_BACKGROUND_PROB),
                ("local_texture", RandomLocalTexturePerturbation(p=1.0, alpha_range=AUGMENT_LOCAL_TEXTURE_ALPHA_RANGE), GLYPH_AUG_LOCAL_TEXTURE_PROB),
                ("tone_contrast", RandomToneAndContrastJitter(p=1.0), GLYPH_AUG_TONE_CONTRAST_PROB),
                ("tilt", RandomTiltJitter(degrees=AUGMENT_TILT_DEGREES, p=1.0), GLYPH_AUG_TILT_PROB),
                ("border_crop", RandomBorderMaskAndEdgeCrop(p=1.0, max_border_frac=min(0.08, float(AUGMENT_BORDER_MAX_FRAC)), max_crop_frac=min(0.1, float(AUGMENT_EDGE_CROP_MAX_FRAC))), GLYPH_AUG_BORDER_CROP_PROB),
                ("zoom", RandomZoomJitter(scale_range=AUGMENT_ZOOM_SCALE_RANGE, p=1.0), GLYPH_AUG_ZOOM_PROB),
                ("randaugment", transforms.RandAugment(num_ops=2, magnitude=6) if bool(CHAR_PATCH_APPLY_RANDAUGMENT) else None, GLYPH_AUG_RANDAUGMENT_PROB),
            ]
            char_ops.append(WeightedOneOf(char_oneof_candidates, p=GLYPH_AUG_APPLY_PROB))
        char_ops.append(transforms.ToTensor())
        if split == "train" and AUGMENT_RANDOM_ERASING_PROB > 0:
            char_ops.append(transforms.RandomErasing(
                p=AUGMENT_RANDOM_ERASING_PROB,
                scale=GLYPH_RANDOM_ERASING_SCALE,
                ratio=AUGMENT_RANDOM_ERASING_RATIO,
                value="random",
            ))
        if bool(CHAR_PATCH_APPLY_IMAGENET_NORM):
            char_ops.append(transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
        self.char_transform = transforms.Compose(char_ops)
        self.label2idx = label2idx
        self.patch_size = patch_size
        self.stride = stride
        self.to_rgb = to_rgb
        self.max_tiles_per_image = max_tiles_per_image
        self.center_bias = center_bias
        self.db_config_path = db_config_path
        self.split = split  # 'train', 'val', or 'test' - for verification/debugging
        
        # Determine patch loading method from parameter or system config
        if patch_loading_method is None:
            patch_loading_method = PATCH_LOADING_METHOD
        
        # Set flags based on patch loading method
        if patch_loading_method == 'db':
            # Database only - fail if not available
            self.use_db_coordinates = True
            self.use_xml_extraction = False
        elif patch_loading_method == 'extract':
            # Extraction only - never use database
            self.use_db_coordinates = False
            self.use_xml_extraction = True
        elif patch_loading_method == 'auto':
            # Try DB first, fall back to extraction
            self.use_db_coordinates = True
            self.use_xml_extraction = True
        else:
            raise ValueError(f"Invalid PATCH_LOADING_METHOD: {patch_loading_method}. Must be 'db', 'extract', or 'auto'")
        
        # Override with explicit parameters if provided (for backward compatibility)
        if use_db_coordinates is not None:
            self.use_db_coordinates = use_db_coordinates
        if use_xml_extraction is not None:
            self.use_xml_extraction = use_xml_extraction
        
        # Create a single DB connection for the dataset (will be reused)
        self._db_conn = None
        
        # Note: Split verification is implicit - each dataset only receives paths from its split.
        # When querying DB by image_path, we can only get patches for images in this dataset's path list.
        # This ensures no data leakage between train/val/test splits.

    def _get_db_connection(self):
        """Get or create database connection (lazy initialization)."""
        if self._db_conn is None:
            try:
                self._db_conn = get_db_connection(self.db_config_path)
            except Exception as e:
                logger.exception(f"Could not connect to database (config: {self.db_config_path}). Will use extraction instead.")
                warnings.warn(f"Could not connect to database: {e}. Will use extraction instead.")
                self.use_db_coordinates = False
        return self._db_conn
    
    def _extract_patches_from_coords(self, image, coords_normalized: List[Tuple[float, float]]):
        """
        Extract patches from image using normalized coordinates from database.
        
        Args:
            image: PIL Image
            coords_normalized: List of (center_x, center_y) normalized coordinates [0, 1]
            
        Returns:
            Tuple of (patches tensor, coords tensor)
        """
        w, h = image.size
        patches = []
        coords = []
        
        for cx_norm, cy_norm in coords_normalized:
            # Convert normalized coordinates to pixel positions
            center_x = cx_norm * w
            center_y = cy_norm * h
            
            # Calculate patch bounding box
            left = max(0, int(center_x - self.patch_size / 2))
            top = max(0, int(center_y - self.patch_size / 2))
            right = min(w, left + self.patch_size)
            bottom = min(h, top + self.patch_size)
            
            # Adjust if patch would go out of bounds
            if right - left < self.patch_size:
                if left == 0:
                    right = min(w, self.patch_size)
                else:
                    left = max(0, right - self.patch_size)
            if bottom - top < self.patch_size:
                if top == 0:
                    bottom = min(h, self.patch_size)
                else:
                    top = max(0, bottom - self.patch_size)
            
            # Extract patch
            box = (left, top, right, bottom)
            patch = image.crop(box)
            
            # Resize if needed (shouldn't happen, but safety check)
            if patch.size[0] != self.patch_size or patch.size[1] != self.patch_size:
                patch = patch.resize((self.patch_size, self.patch_size))
            
            # Apply transform
            transformed_patch = self.transform(patch)
            patches.append(transformed_patch)
            
            # Store normalized coordinates
            coords.append((cx_norm, cy_norm))
        
        if patches:
            patches = torch.stack(patches)  # [num_patches, 3, patch_size, patch_size]
            coords = torch.tensor(coords, dtype=torch.float32)
            # Default to page 0 if not specified (will be improved when we add layout metadata to DB)
            page_segments = torch.zeros(len(patches), dtype=torch.long)
            return patches, coords, page_segments
        else:
            # Fallback: return single center patch
            patch = image.resize((self.patch_size, self.patch_size))
            transformed_patch = self.transform(patch)
            patches = torch.stack([transformed_patch])
            coords = torch.tensor([(0.5, 0.5)], dtype=torch.float32)
            page_segments = torch.zeros(1, dtype=torch.long)
            return patches, coords, page_segments
    
    def extract_patches(self, image, image_path: Optional[str] = None, xml_path: Optional[str] = None):
        """
        Extract patches from image.
        
        Args:
            image: PIL Image
            image_path: Path to image file (used for DB lookup and XML auto-detection)
            xml_path: Explicit path to XML file. If provided, uses this instead of auto-detecting.
                     This should be the xml_path from the database table.
        """
        image = preprocess_before_patch(image)
        
        # Try loading from database first if enabled
        if self.use_db_coordinates and image_path is not None:
            try:
                conn = self._get_db_connection()
                if conn is not None:
                    from train.db_loader import load_patch_metadata_from_db, PATCH_IMAGE_METADATA_TABLE_NAME
                    
                    # Load coordinates and metadata
                    coords_normalized = load_patch_coordinates_from_db(
                        image_path, 
                        conn, 
                        self.patch_size,
                        PATCH_COORDINATES_TABLE_NAME,
                        split=self.split  # Pass split for potential verification
                    )
                    metadata = load_patch_metadata_from_db(
                        image_path,
                        conn,
                        PATCH_IMAGE_METADATA_TABLE_NAME
                    )
                    
                    if coords_normalized is not None:
                        # Limit to max_tiles_per_image if specified
                        if self.max_tiles_per_image is not None and len(coords_normalized) > self.max_tiles_per_image:
                            coords_normalized = coords_normalized[:self.max_tiles_per_image]
                        
                        # Apply rotation to image if it was applied during extraction
                        # Coordinates in DB are normalized to rotated image, so we rotate the image
                        # and use coordinates directly (plug and play)
                        if metadata and metadata.get('rotation_corrected', False):
                            rotation_angle = metadata.get('rotation_angle', 0)
                            if rotation_angle != 0:
                                from utilities.page_rotation import rotate_image
                                image = rotate_image(image, rotation_angle)
                        
                        # Extract patches using coordinates from database
                        # Coordinates are already in rotated space, image is now rotated - plug and play!
                        patches, coords, page_segments = self._extract_patches_from_coords(image, coords_normalized)
                        return patches, coords, page_segments
            except Exception as e:
                # If DB loading fails and we're in 'db' mode, raise the error
                if not self.use_xml_extraction:
                    logger.exception(f"Failed to load patch coordinates from DB for {image_path}. DB-only mode enabled, cannot fall back.")
                    raise RuntimeError(f"Failed to load patch coordinates from DB for {image_path} and DB-only mode is enabled. Error: {e}") from e
                # Otherwise, fall through to XML extraction
                logger.exception(f"Failed to load patch coordinates from DB for {image_path}. Falling back to XML extraction.")
                warnings.warn(f"Failed to load patch coordinates from DB for {image_path}: {e}. Falling back to extraction.")
        
        # Use XML-based extraction if enabled and image_path is provided
        if self.use_xml_extraction and image_path is not None:
            try:
                patch_images, coords, page_segments, metadata = extract_patches_with_xml(
                    image=image,
                    image_path=image_path,
                    patch_size=self.patch_size,
                    stride=self.stride,
                    max_patches=self.max_tiles_per_image,
                    xml_path=xml_path,  # Use provided xml_path from DB table (if available)
                )
                
                # Check if XML extraction found text regions
                text_regions_count = metadata.get('text_regions_count', 0)
                if text_regions_count == 0:
                    xml_path = metadata.get('xml_path', None)
                    # Collect warning instead of logging immediately (to avoid spam)
                    # Will be saved to file after first epoch
                    _xml_warning_paths.add((image_path, xml_path))
                    if not TILES_FALLBACK_GRID:
                        empty_patches = torch.zeros(0, 3, self.patch_size, self.patch_size)
                        empty_coords = torch.zeros(0, 2)
                        empty_page_segments = torch.zeros(0, dtype=torch.long)
                        return empty_patches, empty_coords, empty_page_segments
                else:
                    # Apply transformations to each patch
                    patches = []
                    for patch_img in patch_images:
                        transformed_patch = self.transform(patch_img)
                        patches.append(transformed_patch)
                    
                    if patches:
                        patches = torch.stack(patches)  # [num_patches, 3, patch_size, patch_size]
                        return patches, coords, page_segments
                    # XML found text regions but returned no tile candidates.
                    # With fallback disabled, preserve the empty visual stream.
                    # With fallback enabled, fall through to the center/grid fallback below.
                    if not TILES_FALLBACK_GRID:
                        empty_patches = torch.zeros(0, 3, self.patch_size, self.patch_size)
                        empty_coords = torch.zeros(0, 2)
                        empty_page_segments = torch.zeros(0, dtype=torch.long)
                        return empty_patches, empty_coords, empty_page_segments
            except Exception as e:
                if not TILES_FALLBACK_GRID:
                    logger.exception(f"XML-based patch extraction failed for {image_path}. Returning empty visual stream because TILES_FALLBACK_GRID=False.")
                    warnings.warn(f"XML-based patch extraction failed for {image_path}: {e}. Returning empty visual stream because TILES_FALLBACK_GRID=False.")
                    empty_patches = torch.zeros(0, 3, self.patch_size, self.patch_size)
                    empty_coords = torch.zeros(0, 2)
                    empty_page_segments = torch.zeros(0, dtype=torch.long)
                    return empty_patches, empty_coords, empty_page_segments

                # If XML extraction fails, fall back to original method
                logger.exception(f"XML-based patch extraction failed for {image_path}. Falling back to center-based extraction.")
                warnings.warn(f"XML-based patch extraction failed for {image_path}: {e}. Falling back to center-based extraction.")
        
        # Fallback to original center-based extraction
        w, h = image.size
        patches = []
        coords = []
        # grid of candidate positions
        tops = list(range(0, max(1, h - self.patch_size + 1), self.stride))
        lefts = list(range(0, max(1, w - self.patch_size + 1), self.stride))
        if len(tops) == 0 or tops[-1] != max(0, h - self.patch_size):
            tops.append(max(0, h - self.patch_size))
        if len(lefts) == 0 or lefts[-1] != max(0, w - self.patch_size):
            lefts.append(max(0, w - self.patch_size))
        positions = [(t, l) for t in tops for l in lefts]

        # Optional center-biased subsampling BEFORE cropping to save work
        if self.max_tiles_per_image is not None and len(positions) > self.max_tiles_per_image:
            if self.center_bias:
                import random
                cx_img = w / 2.0
                cy_img = h / 2.0
                def score(pos):
                    top, left = pos
                    right = min(left + self.patch_size, w)
                    bottom = min(top + self.patch_size, h)
                    cx = (left + right) / 2.0
                    cy = (top + bottom) / 2.0
                    dx = (cx / w) - 0.5
                    dy = (cy / h) - 0.5
                    jitter = random.uniform(0.0, 1e-6)
                    return dx * dx + dy * dy + jitter
                positions = sorted(positions, key=score)[: self.max_tiles_per_image]
            else:
                import random
                positions = random.sample(positions, self.max_tiles_per_image)

        for top, left in positions:
            box = (left, top, min(left + self.patch_size, w), min(top + self.patch_size, h))
            patch = image.crop(box)
            if patch.size[0] != self.patch_size or patch.size[1] != self.patch_size:
                patch = patch.resize((self.patch_size, self.patch_size))

            # Apply transformations to each patch before appending
            transformed_patch = self.transform(patch)
            patches.append(transformed_patch)
            
            # center coordinates normalized
            cx = (box[0] + box[2]) / 2.0 / w
            cy = (box[1] + box[3]) / 2.0 / h
            coords.append((cx, cy))
        if not patches:
            # If the image is smaller than patch_size, resize and take one patch
            patch = image.resize((self.patch_size, self.patch_size))
            transformed_patch = self.transform(patch)
            patches.append(transformed_patch)
            coords.append((0.5, 0.5))

        patches = torch.stack(patches)  # [num_patches, 3, patch_size, patch_size]
        coords = torch.tensor(coords, dtype=torch.float32)
        page_segments = torch.zeros(len(patches), dtype=torch.long)
        return patches, coords, page_segments
    
    def __del__(self):
        """Close database connection when dataset is destroyed."""
        if hasattr(self, '_db_conn') and self._db_conn is not None:
            try:
                self._db_conn.close()
            except:
                pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[Tuple]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # DEBUG: Log first few samples to verify label consistency
        if idx < 10 or idx % 1000 == 0:
             logger.debug(f"[DEBUG] Dataset Access: idx={idx}, path={os.path.basename(img_path)}, label='{label}'")
        
        # CRITICAL: If label not found, raise error instead of returning -1
        # Returning -1 causes ArcFace to fail with index errors
        if label not in self.label2idx:
            logger.error(f"[DEBUG] Label lookup failed:")
            logger.error(f"  Image: {img_path}")
            logger.error(f"  Label: '{label}' (type: {type(label)})")
            logger.error(f"  Label2idx keys (first 10): {list(self.label2idx.keys())[:10]}")
            logger.error(f"  Label2idx size: {len(self.label2idx)}")
            logger.error(f"  All labels in dataset (first 10): {self.labels[:10]}")
            logger.error(f"  Unique labels in dataset: {len(set(self.labels))}")
            raise ValueError(
                f"Label '{label}' not found in label2idx mapping. "
                f"Available labels: {list(self.label2idx.keys())[:10]}... "
                f"This indicates a data inconsistency bug."
            )
        label_idx = self.label2idx[label]
        
        # DEBUG: Verify label_idx is valid
        if label_idx < 0:
            logger.error(f"[DEBUG] Invalid label_idx: {label_idx} for label '{label}'")
            raise ValueError(f"Label index {label_idx} is negative for label '{label}'")
        
        # Resolve XML path only when a modality needs it:
        # - Words: XML only (no image)
        # - Characters: image + XML
        # - Visual: image only, unless patch extraction uses XML (fallback or extract mode)
        need_xml = (
            self.use_char_mod
            or self.use_word_mod
            or (self.use_visual_mod and self.use_xml_extraction)
        )
        xml_path = None
        if need_xml:
            xml_path = self.xml_paths[idx] if self.xml_paths is not None else find_xml_path_pretrain(img_path)
        
        # Load image only when visual or character modality is enabled (words need only XML)
        need_image = self.use_visual_mod or self.use_char_mod
        pil_img = None
        if need_image:
            try:
                pil_img = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.exception(f"Could not load image {img_path}. Skipping sample.")
                print(f"Warning: Could not load image {img_path}. Error: {e}. Skipping sample.")
                return None
        
        # Extract visual patches only if visual modality is enabled
        if self.use_visual_mod and pil_img is not None:
            patches, coords, page_segments = self.extract_patches(pil_img, image_path=img_path, xml_path=xml_path)
        else:
            # Return dummy patches when visual modality is disabled
            patches = torch.zeros(1, 3, self.patch_size, self.patch_size)
            coords = torch.zeros(1, 2)
            page_segments = torch.zeros(1, dtype=torch.long)
        
        # NOTE: glyph_page_segments and word_page_segments are currently default 0
        # although they could be extracted from XML in the future if needed.
        
        # Extract character patches and words from XML (if modalities are enabled)
        char_patches_tensor = torch.zeros(0, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)
        char_metadata_list = []
        words = []
        word_metadata = []
        if self.use_char_mod or self.use_word_mod:
            try:
                # xml_path already resolved above
                
                if xml_path and os.path.exists(xml_path):
                    # Extract character patches
                    if self.use_char_mod:
                        char_patches, char_metadata = extract_character_patches(
                            image=pil_img,
                            image_path=img_path,
                            char_patch_size=CHAR_PATCH_SIZE,
                            max_chars=None,  # Get all first, then filter and sample
                            xml_path=xml_path,
                            only_middle_glyphs=GLYPH_ONLY_MIDDLE_LETTERS,
                            min_word_length_for_middle=GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE,
                            input_alphabet=self.glyph_input_alphabet,
                        )
                        
                        if char_patches:
                            quality_filter = GlyphHardQualityFilter()
                            
                            # Convert to tensors for quality filtering (blank detection uses variance)
                            patch_tensors = [transforms.ToTensor()(p) for p in char_patches]
                            # Track PIL images alongside tensors so we can augment later
                            pil_by_id = {id(t): pil for t, pil in zip(patch_tensors, char_patches)}

                            filtered_patches, filtered_metadata = quality_filter.filter_glyphs(patch_tensors, char_metadata)
                            final_patches, final_metadata = quality_filter.sample_diverse_glyphs(
                                filtered_patches, filtered_metadata, MAX_CHARS_PER_IMAGE
                            )
                            
                            assert len(final_patches) == len(final_metadata), (
                                f"Glyph patches/metadata length mismatch: "
                                f"{len(final_patches)} patches vs {len(final_metadata)} metadata entries "
                                f"(image: {img_path})"
                            )
                            
                            # Apply char_transform (color augment + ToTensor + Normalize) to PIL images.
                            # self.char_transform includes augmentations only for the train split.
                            char_patches_transformed = []
                            for patch_tensor in final_patches:
                                pil_img_glyph = pil_by_id.get(id(patch_tensor))
                                if pil_img_glyph is not None:
                                    char_patches_transformed.append(self.char_transform(pil_img_glyph))
                                else:
                                    if bool(CHAR_PATCH_APPLY_IMAGENET_NORM):
                                        char_patches_transformed.append(
                                            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)(patch_tensor)
                                        )
                                    else:
                                        char_patches_transformed.append(patch_tensor)
                            if char_patches_transformed:
                                char_patches_tensor = torch.stack(char_patches_transformed)
                            else:
                                char_patches_tensor = torch.zeros(0, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)
                            char_metadata_list = final_metadata if final_metadata is not None else []
                        else:
                            char_metadata_list = []
                    
                    # Extract words
                    if self.use_word_mod:
                        words, word_metadata = extract_words_from_alto(
                            alto_xml_path=xml_path,
                            string_conf_threshold=OCR_STRING_CONFIDENCE_THRESHOLD,
                            use_hebrew_dict=USE_HEBREW_DICT_CHECK,
                            max_words=MAX_WORDS_PER_IMAGE,
                            legacy_20260824=self.legacy_word_extraction_20260824,
                        )
            except Exception as e:
                # If XML extraction fails, continue without character/word data
                logger.exception(f"Failed to extract character/word data from XML for {img_path}. Continuing without character/word data.")
                warnings.warn(f"Failed to extract character/word data from XML for {img_path}: {e}")

        # Normalize word metadata coordinates when we have image dimensions (skipped for words-only, since image is not loaded)
        if word_metadata and pil_img:
            img_width, img_height = pil_img.size
            for meta in word_metadata:
                if 'normalized_center_x' not in meta and meta.get('hpos') is not None and meta.get('width') is not None:
                    center_x = (meta['hpos'] + meta['width'] / 2) / img_width
                    meta['normalized_center_x'] = center_x
                if 'normalized_center_y' not in meta and meta.get('vpos') is not None and meta.get('height') is not None:
                    center_y = (meta['vpos'] + meta['height'] / 2) / img_height
                    meta['normalized_center_y'] = center_y
        
        # Glyph coordinates are now normalized in extract_character_patches (consistent with tile coordinates)
        # No need to normalize here - they're already normalized at extraction time
        
        return patches, coords, page_segments, char_patches_tensor, char_metadata_list, words, word_metadata, label_idx, img_path


def tile_collate_with_padding(
    batch,
    glyph_input_alphabet=None,
    glyphs_per_class=GLYPHS_PER_CLASS,
):
    """
    Collate function for DataLoader that handles variable-length sequences for all modalities.
    
    Args:
        batch: list of (patches [Ni,3,H,W], coords [Ni,2], char_patches [Mi,3,H,W], char_metadata [Mi],
               words [Li], word_metadata [Li], label, path) tuples
        
    Returns:
        tiles: [B, Nmax, 3, H, W] - padded patch tensors
        valid_mask: [B, Nmax] bool - mask indicating which patches are valid
        coords: [B, Nmax, 2] - padded coordinate tensors
        char_patches: [B, Mmax, 3, H, W] - padded character patch tensors
        char_valid_mask: [B, Mmax] bool - mask for character patches
        glyph_page_segments: [B, Mmax] long - page segment IDs for glyphs
        char_metadata: List[List[Dict]] - per-image character metadata aligned to unpadded char_patches
        words: List[List[str]] - list of word lists per image
        word_metadata: List[List[Dict]] - list of word metadata lists per image
        labels: [B] - label tensor
        paths: tuple of image paths
    """
    # Drop samples that __getitem__ intentionally skipped, e.g. unreadable images.
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None

    # Unpack with page_segments
    patch_tensors, coord_tensors, tile_page_segment_tensors, char_patch_tensors, char_metadata_lists, word_lists, word_metadata_lists, labels, paths = zip(*batch)
    
    # DEBUG: Validate labels before creating tensor
    labels_list = list(labels)
    if len(labels_list) > 0:
        # Check for None or invalid labels
        invalid_labels = [i for i, lbl in enumerate(labels_list) if lbl is None or (isinstance(lbl, (int, float)) and (lbl < 0 or not isinstance(lbl, int)))]
        if invalid_labels:
            logger.error(f"[DEBUG] Collate: Found invalid labels at indices {invalid_labels}")
            for idx in invalid_labels[:5]:  # Show first 5
                logger.error(f"  Index {idx}: label={labels_list[idx]}, type={type(labels_list[idx])}, path={paths[idx]}")
        
        # Track batch label diversity
        label_values = [lbl for lbl in labels_list if isinstance(lbl, (int, float))]
        if label_values:
            unique_labels = len(set(label_values))
            batch_size = len(label_values)
            diversity_ratio = unique_labels / batch_size if batch_size > 0 else 0
            
            _batch_label_diversity_stats['total_batches'] += 1
            
            # Check if batch is homogeneous (all same label)
            if unique_labels == 1:
                _batch_label_diversity_stats['homogeneous_batches'] += 1
            # Check if batch has low diversity (< 30% unique labels)
            elif diversity_ratio < 0.3:
                _batch_label_diversity_stats['low_diversity_batches'] += 1
            
            # Log first batch details
            if not hasattr(tile_collate_with_padding, '_first_batch_logged'):
                logger.info(f"[DEBUG] Collate: First batch labels - min={min(label_values)}, max={max(label_values)}, count={len(label_values)}, unique={unique_labels}, diversity={diversity_ratio:.2%}")
                logger.info(f"[DEBUG] Collate: Sample labels: {labels_list[:min(5, len(labels_list))]}")
                tile_collate_with_padding._first_batch_logged = True
            
            # Note: per-batch homogeneous warnings removed — with small batch sizes
            # (e.g. 2) and many classes, homogeneous batches are expected by chance.
            # Use log_batch_label_diversity_stats() at epoch end for aggregate stats.
    
    # Handle visual patches
    # NOTE: If an entire batch has 0 extracted patches (can happen with strict XML filtering
    # or missing/empty XML), creating [B, 0, ...] tensors can crash some CUDA/PyTorch builds
    # inside the DataLoader pin-memory thread ("CUDA error: invalid argument").
    # As with glyphs, we return one fully-masked dummy tile to keep semantics identical.
    max_patches = max(p.shape[0] for p in patch_tensors) if patch_tensors else 0
    B = len(patch_tensors)
    C, H, W = patch_tensors[0].shape[1:]
    if max_patches > 0:
        tiles = torch.zeros(B, max_patches, C, H, W)
        coords = torch.zeros(B, max_patches, 2)
        tile_page_segments = torch.zeros(B, max_patches, dtype=torch.long)
        valid_mask = torch.zeros(B, max_patches, dtype=torch.bool)
        for i, (p, c, s) in enumerate(zip(patch_tensors, coord_tensors, tile_page_segment_tensors)):
            n = p.shape[0]
            if n > 0:
                tiles[i, :n] = p
                coords[i, :n] = c
                tile_page_segments[i, :n] = s
                valid_mask[i, :n] = True
    else:
        tiles = torch.zeros(B, 1, C, H, W)
        coords = torch.zeros(B, 1, 2)
        tile_page_segments = torch.zeros(B, 1, dtype=torch.long)
        valid_mask = torch.zeros(B, 1, dtype=torch.bool)
    
    # Handle character patches. Keep a fixed per-letter slot layout:
    # [letter_0 x GLYPHS_PER_CLASS, letter_1 x GLYPHS_PER_CLASS, ...].
    # Missing letters/glyphs stay masked out instead of shortening the sequence.
    effective_alphabet = (
        GLYPH_INPUT_ALPHABET
        if glyph_input_alphabet is None
        else glyph_input_alphabet
    )
    input_class_count = len(effective_alphabet)
    max_chars = input_class_count * int(glyphs_per_class)
    # Default char_class_id for padding: last class (unknown/other)
    unknown_class_id = input_class_count
    char_patches = torch.zeros(B, max_chars, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)
    char_valid_mask = torch.zeros(B, max_chars, dtype=torch.bool)
    glyph_coords = torch.zeros(B, max_chars, 4)
    glyph_page_segments = torch.zeros(B, max_chars, dtype=torch.long)
    char_class_ids = torch.full((B, max_chars), unknown_class_id, dtype=torch.long)
    aligned_char_metadata = [[{} for _ in range(max_chars)] for _ in range(B)]
    for i, (cp, metadata_list) in enumerate(zip(char_patch_tensors, char_metadata_lists)):
        class_counts = [0 for _ in range(input_class_count)]
        m = cp.shape[0]
        for src_idx in range(m):
            meta = metadata_list[src_idx] if src_idx < len(metadata_list) else {}
            if not meta or 'char_class_id' not in meta:
                continue
            class_id = int(meta['char_class_id'])
            if class_id < 0 or class_id >= input_class_count:
                continue
            class_offset = class_counts[class_id]
            if class_offset >= int(glyphs_per_class):
                continue
            dst_idx = class_id * int(glyphs_per_class) + class_offset
            if dst_idx >= max_chars:
                continue
            class_counts[class_id] += 1

            char_patches[i, dst_idx] = cp[src_idx]
            char_valid_mask[i, dst_idx] = True
            char_class_ids[i, dst_idx] = class_id
            aligned_char_metadata[i][dst_idx] = meta
            if 'normalized_x' in meta:
                glyph_coords[i, dst_idx, 0] = meta['normalized_x']
                glyph_coords[i, dst_idx, 1] = meta['normalized_y']
                glyph_coords[i, dst_idx, 2] = meta['normalized_w']
                glyph_coords[i, dst_idx, 3] = meta['normalized_h']
            if 'page_segment' in meta:
                glyph_page_segments[i, dst_idx] = int(meta.get('page_segment') or 0)
    
    # Words and metadata are already lists, just convert to list of lists
    char_metadata = aligned_char_metadata
    words = list(word_lists)
    word_metadata = list(word_metadata_lists)
    
    # Convert labels to tensor with validation
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # DEBUG: Validate labels tensor
    if len(labels_tensor) > 0:
        if torch.isnan(labels_tensor.float()).any():
            logger.error(f"[DEBUG] Collate: NaN detected in labels tensor!")
            logger.error(f"  Labels: {labels}")
            logger.error(f"  Labels tensor: {labels_tensor}")
        if (labels_tensor < 0).any():
            invalid_indices = (labels_tensor < 0).nonzero(as_tuple=True)[0]
            logger.error(f"[DEBUG] Collate: Negative labels detected at indices {invalid_indices.tolist()}")
            logger.error(f"  Invalid labels: {labels_tensor[invalid_indices].tolist()}")
            logger.error(f"  Corresponding paths: {[paths[i] for i in invalid_indices.tolist()]}")
    
    # Fail-fast: DataLoader output must be CPU tensors with non-zero numel.
    # (Pinning happens in the main process; never touch CUDA in worker processes.)
    tensors_to_check = {
        "tiles": tiles,
        "valid_mask": valid_mask,
        "coords": coords,
        "tile_page_segments": tile_page_segments,
        "char_patches": char_patches,
        "char_valid_mask": char_valid_mask,
        "glyph_coords": glyph_coords,
        "glyph_page_segments": glyph_page_segments,
        "char_class_ids": char_class_ids,
        "labels_tensor": labels_tensor,
    }
    for name, t in tensors_to_check.items():
        if not torch.is_tensor(t):
            raise RuntimeError(f"[COLLATE] Expected tensor for {name}, got {type(t)} (paths(first8)={list(paths)[:8]})")
        if t.is_cuda:
            raise RuntimeError(
                f"[COLLATE] Tensor '{name}' is CUDA (unexpected from DataLoader/collate).\n"
                f"  device={t.device}, dtype={t.dtype}, shape={tuple(t.shape)}, numel={t.numel()}\n"
                f"  paths(first8)={list(paths)[:8]}"
            )
        if t.numel() == 0:
            raise RuntimeError(
                f"[COLLATE] Tensor '{name}' has numel==0 (can crash pin_memory thread on some setups).\n"
                f"  device={t.device}, dtype={t.dtype}, shape={tuple(t.shape)}, numel={t.numel()}\n"
                f"  paths(first8)={list(paths)[:8]}"
            )

    return tiles, valid_mask, coords, tile_page_segments, char_patches, char_valid_mask, glyph_coords, glyph_page_segments, char_class_ids, char_metadata, words, word_metadata, labels_tensor, paths


def save_xml_warnings_to_file(output_dir: str = "logs", filename: str = None):
    """
    Save collected XML warning paths to a file and clear the collection.
    
    This should be called once after the first epoch (or periodically) to avoid
    logging spam during training.
    
    Args:
        output_dir: Directory to save the warning file (default: "logs")
        filename: Name of the output file (if None, will use timestamp-based name)
    """
    global _xml_warning_paths
    
    if not _xml_warning_paths:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp if not provided
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"xml_warnings_{timestamp}.txt"
    
    output_path = os.path.join(output_dir, filename)
    
    # Sort for consistent output
    sorted_warnings = sorted(_xml_warning_paths, key=lambda x: x[0])
    
    with open(output_path, 'w') as f:
        f.write(f"XML Extraction Warnings\n")
        f.write(f"Total images with missing or empty XML: {len(sorted_warnings)}\n")
        f.write(f"{'='*80}\n\n")
        
        missing_xml_count = 0
        empty_xml_count = 0
        
        for image_path, xml_path in sorted_warnings:
            if xml_path is None:
                missing_xml_count += 1
                f.write(f"MISSING XML: {image_path}\n")
            else:
                empty_xml_count += 1
                f.write(f"EMPTY XML (no text regions): {image_path}\n")
                f.write(f"  XML path: {xml_path}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"Summary:\n")
        f.write(f"  Missing XML files: {missing_xml_count}\n")
        f.write(f"  Empty XML files (no text regions): {empty_xml_count}\n")
        f.write(f"  Total: {len(sorted_warnings)}\n")
    
    logger.info(f"Saved {len(sorted_warnings)} XML warnings to {output_path}")
    print(f"Saved {len(sorted_warnings)} XML warnings to {output_path}")
    
    # Clear the collection after saving
    _xml_warning_paths.clear()


def log_batch_label_diversity_stats():
    """
    Log statistics about batch label diversity.
    Useful for detecting dataset ordering or shuffling issues.
    """
    global _batch_label_diversity_stats
    
    total = _batch_label_diversity_stats['total_batches']
    if total == 0:
        return
    
    homogeneous = _batch_label_diversity_stats['homogeneous_batches']
    low_diversity = _batch_label_diversity_stats['low_diversity_batches']
    
    homogeneous_pct = (homogeneous / total * 100) if total > 0 else 0
    low_diversity_pct = (low_diversity / total * 100) if total > 0 else 0
    
    logger.info(f"[Batch Diversity Stats] Total batches: {total}")
    logger.info(f"[Batch Diversity Stats] Homogeneous batches (all same label): {homogeneous} ({homogeneous_pct:.1f}%)")
    logger.info(f"[Batch Diversity Stats] Low diversity batches (<30% unique): {low_diversity} ({low_diversity_pct:.1f}%)")
    
    if homogeneous_pct > 20:
        logger.warning(f"[Batch Diversity Stats] High percentage ({homogeneous_pct:.1f}%) of homogeneous batches detected. "
                      f"This may indicate dataset ordering issues or insufficient shuffling.")
    
    # Reset stats for next epoch
    _batch_label_diversity_stats = {
        'total_batches': 0,
        'homogeneous_batches': 0,
        'low_diversity_batches': 0,
    }
