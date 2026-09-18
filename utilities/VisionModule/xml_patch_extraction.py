"""
XML-based patch extraction for manuscript images.

This module extracts patches from manuscript images based on OCR text regions
defined in ALTO XML files, rather than using center-biased extraction.
It handles both single-page and two-page images by using OCR regions.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict
import torch
from PIL import Image
import random
import numpy as np
import logging

# Set up logger for XML patch extraction
logger = logging.getLogger(__name__)
from system import (
    TILES_FALLBACK_GRID,
    XML_PATCH_STRIDE_MULTIPLIER,
    XML_PATCH_OVERLAP_THRESHOLD,
    XML_PATCH_RELAXED_THRESHOLDS,
    XML_PATCH_EXPAND_RATIO,
    XML_PATCH_CONSTRAIN_TO_BOUNDS,
    XML_PATCH_FILTER_BY_TEXT_COVERAGE,
    XML_PATCH_MIN_TEXT_COVERAGE,
    XML_PATCH_RANK_BY_INK_DENSITY,
    XML_PATCH_INK_THRESHOLD,
    XML_PATCH_MIN_INK_RATIO,
    XML_PATCH_APPLY_READING_ORDER,
    XML_PATCH_READING_DIRECTION_RTL,
    XML_PATCH_DETECT_ROTATION,
)
from utilities.page_rotation import (
    detect_page_rotation,
    apply_rotation_correction,
)


def _get_local_tag(tag: str) -> str:
    """Extract local tag name from namespaced XML tag."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def parse_alto_text_regions(alto_xml_path: str) -> Tuple[List[Dict], Tuple[int, int]]:
    """
    Parse ALTO XML file and extract text regions (TextLine elements).
    
    Args:
        alto_xml_path: Path to ALTO XML file
        
    Returns:
        Tuple of (text_regions, image_size) where:
        - text_regions: List of dicts with keys: 'hpos', 'vpos', 'width', 'height', 'center_x', 'center_y'
        - image_size: Tuple of (width, height) from Page element
    """
    if not os.path.exists(alto_xml_path):
        return [], (0, 0)
    
    try:
        tree = ET.parse(alto_xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.exception(f"Could not parse XML file {alto_xml_path}")
        print(f"Warning: Could not parse XML file {alto_xml_path}: {e}")
        return [], (0, 0)
    
    # Extract page dimensions
    page_width = 0
    page_height = 0
    for page_elem in root.iter():
        if _get_local_tag(page_elem.tag) == 'Page':
            page_width = int(float(page_elem.attrib.get('WIDTH', 0)))
            page_height = int(float(page_elem.attrib.get('HEIGHT', 0)))
            break
    
    if page_width == 0 or page_height == 0:
        print(f"Warning: Could not find valid Page dimensions in {alto_xml_path}")
        return [], (0, 0)
    
    # Extract TextLine elements
    text_regions = []
    for textline_elem in root.iter():
        if _get_local_tag(textline_elem.tag) != 'TextLine':
            continue
        
        hpos = textline_elem.attrib.get('HPOS')
        vpos = textline_elem.attrib.get('VPOS')
        width = textline_elem.attrib.get('WIDTH')
        height = textline_elem.attrib.get('HEIGHT')
        
        if hpos is None or vpos is None or width is None or height is None:
            continue
        
        try:
            hpos = float(hpos)
            vpos = float(vpos)
            width = float(width)
            height = float(height)
        except (ValueError, TypeError):
            continue
        
        # Skip invalid regions
        if width <= 0 or height <= 0:
            continue
        
        # Calculate center
        center_x = hpos + width / 2.0
        center_y = vpos + height / 2.0
        
        # Extract baseline if available (for Hebrew text orientation detection)
        baseline = textline_elem.attrib.get('BASELINE')
        baseline_points = None
        if baseline:
            try:
                # BASELINE format: "x1 y1 [x2 y2 ...] xN yN" (polyline points)
                # Store first and last points for direction detection
                parts = baseline.strip().split()
                if len(parts) >= 4:
                    x1, y1 = float(parts[0]), float(parts[1])
                    x2, y2 = float(parts[-2]), float(parts[-1])
                    baseline_points = (x1, y1, x2, y2)
            except (ValueError, IndexError):
                pass
        
        text_regions.append({
            'hpos': hpos,
            'vpos': vpos,
            'width': width,
            'height': height,
            'center_x': center_x,
            'center_y': center_y,
            'baseline': baseline_points,  # (x1, y1, x2, y2) or None
        })
    
    return text_regions, (page_width, page_height)


def find_xml_path(image_path: str) -> Optional[str]:
    """
    Find the corresponding ALTO XML file for an image path.
    
    Looks for XML files with patterns like:
    - {image_name}_improved_polys.xml
    - {image_name}.xml
    - {image_name}-alto.xml
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Path to XML file if found, None otherwise
    """
    if not os.path.exists(image_path):
        return None
    
    image_dir = os.path.dirname(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Try different XML filename patterns
    xml_patterns = [
        f"{image_name}_improved_polys.xml",
        f"{image_name}.xml",
        f"{image_name}-alto.xml",
    ]
    
    for pattern in xml_patterns:
        xml_path = os.path.join(image_dir, pattern)
        if os.path.exists(xml_path):
            return xml_path
    
    return None


def calculate_patch_text_coverage(
    patch_top: int,
    patch_left: int,
    patch_size: int,
    text_regions: List[Dict],
) -> float:
    """
    Calculate the fraction of a patch area that overlaps with text regions.
    
    Args:
        patch_top: Top coordinate of the patch
        patch_left: Left coordinate of the patch
        patch_size: Size of the patch (width and height)
        text_regions: List of text region dicts with 'hpos', 'vpos', 'width', 'height'
        
    Returns:
        Coverage ratio [0.0, 1.0] representing the fraction of patch area covered by text regions
    """
    patch_right = patch_left + patch_size
    patch_bottom = patch_top + patch_size
    patch_area = patch_size * patch_size
    
    if patch_area == 0:
        return 0.0
    
    # Calculate total overlapping area
    # We'll use a simple approach: sum intersection areas (may slightly overcount overlaps)
    total_overlap = 0.0
    
    for region in text_regions:
        r_left = region['hpos']
        r_top = region['vpos']
        r_right = r_left + region['width']
        r_bottom = r_top + region['height']
        
        # Calculate intersection
        intersect_left = max(patch_left, r_left)
        intersect_top = max(patch_top, r_top)
        intersect_right = min(patch_right, r_right)
        intersect_bottom = min(patch_bottom, r_bottom)
        
        if intersect_right > intersect_left and intersect_bottom > intersect_top:
            overlap_area = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
            total_overlap += overlap_area
    
    # Cap at patch_area to handle any overcounting from overlapping text regions
    coverage = min(total_overlap / patch_area, 1.0)
    return coverage


def calculate_patch_ink_ratio(
    image_gray_np: np.ndarray,
    patch_top: int,
    patch_left: int,
    patch_size: int,
    threshold: int,
) -> float:
    """
    Fraction of pixels darker than ``threshold`` inside the requested patch
    of a precomputed grayscale numpy array.

    A high ink ratio indicates the patch is dense with letters; a low ratio
    means it is mostly blank parchment. Slicing a precomputed array keeps
    this fast even when called for many candidate positions.
    """
    h, w = image_gray_np.shape[:2]
    top = max(0, int(patch_top))
    left = max(0, int(patch_left))
    bottom = min(h, top + int(patch_size))
    right = min(w, left + int(patch_size))
    if bottom <= top or right <= left:
        return 0.0
    region = image_gray_np[top:bottom, left:right]
    if region.size == 0:
        return 0.0
    return float((region < int(threshold)).mean())


def extract_patches_from_text_regions(
    image: Image.Image,
    text_regions: List[Dict],
    patch_size: int,
    stride: int,
    max_patches: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
    constrain_to_bounds: bool = True,
) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]], Optional[Tuple[float, float, float, float]]]:
    """
    Extract patch positions from text regions.
    
    Args:
        image: PIL Image
        text_regions: List of text region dicts with 'hpos', 'vpos', 'width', 'height', 'center_x', 'center_y'
        patch_size: Size of patches to extract
        stride: Stride for patch extraction
        max_patches: Maximum number of patches to return (None for all)
        image_size: Optional (width, height) tuple. If None, uses image.size
        constrain_to_bounds: If True, ensure patches are within original text bounds (not expanded)
        
    Returns:
        Tuple of (patch_positions, normalized_coords, effective_bounds) where:
        - patch_positions: List of (top, left) tuples
        - normalized_coords: List of (cx, cy) normalized coordinates [0, 1]
        - effective_bounds: Tuple describing the final constrained bounds (or None)
    """
    w, h = image.size if image_size is None else image_size
    
    # Get original text bounds for constraint checking
    original_bounds = None
    effective_bounds = None
    if constrain_to_bounds and text_regions:
        original_bounds = get_text_region_bounds(text_regions)
        if original_bounds:
            min_x, min_y, max_x, max_y = original_bounds
            # Clamp to image dimensions
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(w, max_x)
            max_y = min(h, max_y)

            original_bounds = (min_x, min_y, max_x, max_y)
            effective_bounds = original_bounds
    
    if not text_regions:
        if not TILES_FALLBACK_GRID:
            return [], [], effective_bounds

        # Fallback: generate grid positions if no text regions found
        tops = list(range(0, max(1, h - patch_size + 1), stride))
        lefts = list(range(0, max(1, w - patch_size + 1), stride))
        if len(tops) == 0 or tops[-1] != max(0, h - patch_size):
            tops.append(max(0, h - patch_size))
        if len(lefts) == 0 or lefts[-1] != max(0, w - patch_size):
            lefts.append(max(0, w - patch_size))
        positions = [(t, l) for t in tops for l in lefts]
        
        if max_patches is not None and len(positions) > max_patches:
            positions = random.sample(positions, max_patches)
        
        coords = []
        for top, left in positions:
            cx = (left + min(left + patch_size, w)) / 2.0 / w
            cy = (top + min(top + patch_size, h)) / 2.0 / h
            coords.append((cx, cy))
        
        return positions, coords, effective_bounds
    
    # Generate candidate patch positions around text regions
    # Use the provided stride, but ensure it's at least the minimum from multiplier
    effective_stride = max(stride, int(patch_size * XML_PATCH_STRIDE_MULTIPLIER))
    
    candidate_positions = set()
    
    if constrain_to_bounds and original_bounds:
        # If constraining to bounds, generate patches only within original text bounds
        min_x, min_y, max_x, max_y = original_bounds
        
        # Generate patch positions within original bounds
        # Patches must be fully contained, so adjust bounds
        region_tops = list(range(int(min_y), min(int(max_y), h - patch_size + 1), effective_stride))
        region_lefts = list(range(int(min_x), min(int(max_x), w - patch_size + 1), effective_stride))
        
        # Ensure we cover the region (add one at the end if needed)
        if region_tops and region_tops[-1] < int(max_y) - patch_size:
            region_tops.append(max(int(min_y), int(max_y) - patch_size))
        if region_lefts and region_lefts[-1] < int(max_x) - patch_size:
            region_lefts.append(max(int(min_x), int(max_x) - patch_size))
        
        for t in region_tops:
            for l in region_lefts:
                # Double-check patch is fully within bounds
                if (int(min_x) <= l and l + patch_size <= int(max_x) and
                    int(min_y) <= t and t + patch_size <= int(max_y) and
                    0 <= t < h and 0 <= l < w):
                    candidate_positions.add((t, l))
    else:
        # Original behavior: expand regions to capture context
        for region in text_regions:
            # Get bounding box of text region
            r_left = max(0, int(region['hpos']))
            r_top = max(0, int(region['vpos']))
            r_right = min(w, int(region['hpos'] + region['width']))
            r_bottom = min(h, int(region['vpos'] + region['height']))
            
            # Expand region to capture context (configurable ratio)
            expand = int(patch_size * XML_PATCH_EXPAND_RATIO)
            r_left = max(0, r_left - expand)
            r_top = max(0, r_top - expand)
            r_right = min(w, r_right + expand)
            r_bottom = min(h, r_bottom + expand)
            
            # Generate patch positions within this expanded region with larger stride
            region_tops = list(range(r_top, min(r_bottom, h - patch_size + 1), effective_stride))
            region_lefts = list(range(r_left, min(r_right, w - patch_size + 1), effective_stride))
            
            # Ensure we cover the region (add one at the end if needed)
            if region_tops and region_tops[-1] < r_bottom - patch_size:
                region_tops.append(max(r_top, r_bottom - patch_size))
            if region_lefts and region_lefts[-1] < r_right - patch_size:
                region_lefts.append(max(r_left, r_right - patch_size))
            
            for t in region_tops:
                for l in region_lefts:
                    if 0 <= t < h and 0 <= l < w:
                        candidate_positions.add((t, l))
    
    # If we have no candidates, fall back to grid only when configured.
    if not candidate_positions:
        if not TILES_FALLBACK_GRID:
            return [], [], effective_bounds

        tops = list(range(0, max(1, h - patch_size + 1), stride))
        lefts = list(range(0, max(1, w - patch_size + 1), stride))
        if len(tops) == 0 or tops[-1] != max(0, h - patch_size):
            tops.append(max(0, h - patch_size))
        if len(lefts) == 0 or lefts[-1] != max(0, w - patch_size):
            lefts.append(max(0, w - patch_size))
        candidate_positions = {(t, l) for t in tops for l in lefts}
    
    # Filter candidate positions by text coverage BEFORE scoring/selection
    # This ensures we only consider patches that meet the coverage threshold.
    # Falls back to the unfiltered set if the threshold is too aggressive
    # for this page so we never return zero patches just from filtering.
    if XML_PATCH_FILTER_BY_TEXT_COVERAGE and text_regions:
        filtered_candidates = set()
        for top, left in candidate_positions:
            coverage = calculate_patch_text_coverage(top, left, patch_size, text_regions)
            if coverage >= XML_PATCH_MIN_TEXT_COVERAGE:
                filtered_candidates.add((top, left))
        if filtered_candidates:
            candidate_positions = filtered_candidates
        elif not TILES_FALLBACK_GRID:
            return [], [], effective_bounds

    # Optional ink-density filter + ranking. TextLine bboxes include ascender/
    # descender slack, so a high bbox-coverage tile can still be mostly blank
    # parchment. Computing real ink ratios from a grayscale page lets us drop
    # those blank tiles and prefer dense text rows.
    image_gray_np: Optional[np.ndarray] = None
    ink_ratio_by_pos: Dict[Tuple[int, int], float] = {}
    if XML_PATCH_RANK_BY_INK_DENSITY and candidate_positions:
        image_gray_np = np.asarray(image.convert("L"))
        kept: set = set()
        for top, left in candidate_positions:
            ratio = calculate_patch_ink_ratio(
                image_gray_np, top, left, patch_size, XML_PATCH_INK_THRESHOLD
            )
            if ratio < XML_PATCH_MIN_INK_RATIO:
                continue
            ink_ratio_by_pos[(top, left)] = ratio
            kept.add((top, left))
        # Only apply the ink filter if at least one candidate survives;
        # otherwise fall back to the bbox-filtered set so we never return
        # zero patches just because the threshold was too aggressive.
        if kept:
            candidate_positions = kept
        elif not TILES_FALLBACK_GRID:
            return [], [], effective_bounds

    positions = list(candidate_positions)
    
    # Score positions by proximity to text regions and select with non-overlapping constraint
    if max_patches is not None and len(positions) > max_patches:
        def score_position(pos):
            top, left = pos
            patch_cx = (left + min(left + patch_size, w)) / 2.0
            patch_cy = (top + min(top + patch_size, h)) / 2.0
            
            # Find minimum distance to any text region center
            min_dist_sq = float('inf')
            for region in text_regions:
                dx = (patch_cx - region['center_x']) / w
                dy = (patch_cy - region['center_y']) / h
                dist_sq = dx * dx + dy * dy
                min_dist_sq = min(min_dist_sq, dist_sq)

            jitter = random.uniform(0.0, 1e-6)
            if XML_PATCH_RANK_BY_INK_DENSITY:
                # Primary: highest ink density. Tie-break by proximity to a
                # text-region center. Negate ink ratio so that ascending sort
                # surfaces the densest tiles first.
                ink_ratio = ink_ratio_by_pos.get(pos)
                if ink_ratio is None and image_gray_np is not None:
                    ink_ratio = calculate_patch_ink_ratio(
                        image_gray_np, top, left, patch_size, XML_PATCH_INK_THRESHOLD
                    )
                ink_ratio = float(ink_ratio or 0.0)
                return (-ink_ratio, min_dist_sq + jitter)

            return min_dist_sq + jitter
        
        # Sort by score (closest to text regions first)
        sorted_positions = sorted(positions, key=score_position)
        
        # Greedy non-overlapping selection: select patches that don't overlap with already selected ones
        selected = []
        # Minimum center distance based on stride to ensure we respect the stride parameter
        min_center_distance = max(effective_stride * 0.8, patch_size * 0.5)  # At least 80% of stride or 50% of patch size
        
        def patches_overlap(pos1, pos2):
            """Check if two patches overlap significantly based on actual patch overlap area and center distance."""
            top1, left1 = pos1
            top2, left2 = pos2
            
            # First check: center distance should respect stride
            cx1 = left1 + patch_size / 2
            cy1 = top1 + patch_size / 2
            cx2 = left2 + patch_size / 2
            cy2 = top2 + patch_size / 2
            center_dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            
            if center_dist < min_center_distance:
                return True  # Too close based on stride
            
            # Second check: actual patch overlap area
            right1 = left1 + patch_size
            bottom1 = top1 + patch_size
            right2 = left2 + patch_size
            bottom2 = top2 + patch_size
            
            # Calculate intersection
            intersect_left = max(left1, left2)
            intersect_top = max(top1, top2)
            intersect_right = min(right1, right2)
            intersect_bottom = min(bottom1, bottom2)
            
            if intersect_right <= intersect_left or intersect_bottom <= intersect_top:
                return False  # No overlap
            
            # Calculate overlap area
            overlap_area = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
            patch_area = patch_size * patch_size
            
            # Patches overlap if overlap area exceeds threshold fraction
            overlap_ratio = overlap_area / patch_area
            return overlap_ratio > XML_PATCH_OVERLAP_THRESHOLD
        
        # First pass: try to get as many non-overlapping patches as possible
        for pos in sorted_positions:
            # Check if this patch overlaps significantly with any already selected patch
            overlaps = False
            for selected_pos in selected:
                if patches_overlap(pos, selected_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                selected.append(pos)
                if len(selected) >= max_patches:
                    break
        
        # Second pass: if we don't have enough non-overlapping patches, 
        # relax the overlap constraint gradually
        if len(selected) < max_patches:
            remaining = [p for p in sorted_positions if p not in selected]
            
            # Try with progressively more lenient overlap thresholds (from system config)
            for relaxed_overlap_threshold in XML_PATCH_RELAXED_THRESHOLDS:
                # Relaxed minimum distance (allow closer patches)
                relaxed_min_distance = max(effective_stride * 0.5, patch_size * 0.3)
                
                def relaxed_overlap(pos1, pos2):
                    """Check overlap with relaxed threshold based on actual patch overlap."""
                    top1, left1 = pos1
                    top2, left2 = pos2
                    
                    # Check center distance with relaxed threshold
                    cx1 = left1 + patch_size / 2
                    cy1 = top1 + patch_size / 2
                    cx2 = left2 + patch_size / 2
                    cy2 = top2 + patch_size / 2
                    center_dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                    
                    if center_dist < relaxed_min_distance:
                        return True  # Too close
                    
                    # Calculate actual patch overlap
                    right1 = left1 + patch_size
                    bottom1 = top1 + patch_size
                    right2 = left2 + patch_size
                    bottom2 = top2 + patch_size
                    
                    intersect_left = max(left1, left2)
                    intersect_top = max(top1, top2)
                    intersect_right = min(right1, right2)
                    intersect_bottom = min(bottom1, bottom2)
                    
                    if intersect_right <= intersect_left or intersect_bottom <= intersect_top:
                        return False  # No overlap
                    
                    overlap_area = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
                    patch_area = patch_size * patch_size
                    overlap_ratio = overlap_area / patch_area
                    
                    # Patches overlap if overlap area exceeds relaxed threshold
                    return overlap_ratio > relaxed_overlap_threshold
                
                # Create a copy of remaining to iterate over, and track what to remove
                to_add = []
                for pos in remaining:
                    if len(selected) >= max_patches:
                        break
                    
                    overlaps = False
                    for selected_pos in selected:
                        if relaxed_overlap(pos, selected_pos):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        to_add.append(pos)
                
                # Add the non-overlapping patches
                for pos in to_add:
                    if len(selected) >= max_patches:
                        break
                    selected.append(pos)
                
                # Remove added positions from remaining list
                remaining = [p for p in remaining if p not in to_add]
                
                if len(selected) >= max_patches:
                    break
            
            # Final fallback: if still not enough, just take the best remaining patches
            # even if they overlap (this handles cases with very dense text regions)
            if len(selected) < max_patches:
                remaining = [p for p in sorted_positions if p not in selected]
                selected.extend(remaining[:max_patches - len(selected)])
        
        positions = selected
    
    # Calculate normalized coordinates
    coords = []
    for top, left in positions:
        cx = (left + min(left + patch_size, w)) / 2.0 / w
        cy = (top + min(top + patch_size, h)) / 2.0 / h
        coords.append((cx, cy))
    
    return positions, coords, effective_bounds


def detect_page_layout(text_regions: List[Dict], image_width: float, image_height: float) -> Tuple[bool, Optional[float]]:
    """
    Detect if the image contains one or two pages by analyzing text region distribution.
    Uses row-based analysis: groups text lines by vertical position and checks for 
    consistent two-column patterns across multiple rows.
    
    Args:
        text_regions: List of text region dicts
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        Tuple of (is_two_page, split_x) where:
        - is_two_page: True if two pages detected, False otherwise
        - split_x: X coordinate of the page split (None if single page)
    """
    if not text_regions or image_width <= 0 or image_height <= 0:
        return False, None
    
    if len(text_regions) < 4:  # Need at least a few text lines to detect pattern
        return False, None
    
    # Group text lines into rows based on vertical position
    # Lines with similar y positions are in the same row
    row_tolerance = image_height * 0.02  # 2% of image height tolerance for grouping into rows
    
    # Sort by vertical position
    sorted_regions = sorted(text_regions, key=lambda r: r['center_y'])
    
    # Group into rows
    rows = []
    current_row = [sorted_regions[0]]
    current_row_y = sorted_regions[0]['center_y']
    
    for region in sorted_regions[1:]:
        if abs(region['center_y'] - current_row_y) <= row_tolerance:
            # Same row
            current_row.append(region)
            # Update row center (average)
            current_row_y = sum(r['center_y'] for r in current_row) / len(current_row)
        else:
            # New row
            if len(current_row) > 0:
                rows.append(current_row)
            current_row = [region]
            current_row_y = region['center_y']
    
    if len(current_row) > 0:
        rows.append(current_row)
    
    if len(rows) < 2:
        return False, None
    
    # For each row, check if it has two distinct horizontal clusters
    # (indicating two columns/pages)
    row_split_candidates = []
    
    for row in rows:
        if len(row) < 2:
            continue
        
        # Sort row by horizontal position
        row_sorted = sorted(row, key=lambda r: r['center_x'])
        row_x_positions = [r['center_x'] for r in row_sorted]
        
        # Find the best split point for this row
        # Look for the largest gap between consecutive text lines
        max_gap = 0
        best_split_idx = None
        
        for i in range(len(row_x_positions) - 1):
            gap = row_x_positions[i + 1] - row_x_positions[i]
            if gap > max_gap:
                max_gap = gap
                best_split_idx = i
        
        # Check if the gap is significant (at least 15% of image width)
        min_gap_threshold = image_width * 0.15
        if max_gap > min_gap_threshold and best_split_idx is not None:
            # Calculate split point
            split_x = (row_x_positions[best_split_idx] + row_x_positions[best_split_idx + 1]) / 2.0
            
            # Verify both sides have text
            left_count = best_split_idx + 1
            right_count = len(row) - left_count
            
            if left_count >= 1 and right_count >= 1:
                row_split_candidates.append(split_x)
    
    # If multiple rows show the same split pattern, it's likely two pages
    if len(row_split_candidates) < 2:
        return False, None
    
    # Find the most common split position (within tolerance)
    split_tolerance = image_width * 0.05  # 5% tolerance for matching splits
    
    # Cluster split candidates
    split_clusters = []
    for split_x in row_split_candidates:
        # Find existing cluster within tolerance
        matched = False
        for cluster in split_clusters:
            cluster_center = sum(cluster) / len(cluster)
            if abs(split_x - cluster_center) <= split_tolerance:
                cluster.append(split_x)
                matched = True
                break
        
        if not matched:
            split_clusters.append([split_x])
    
    # Find the largest cluster
    if split_clusters:
        largest_cluster = max(split_clusters, key=len)
        
        # Require at least 30% of rows to show this split pattern
        min_rows_required = max(2, int(len(rows) * 0.3))
        if len(largest_cluster) >= min_rows_required:
            # Calculate average split position
            avg_split_x = sum(largest_cluster) / len(largest_cluster)
            
            # Verify: check that text lines on each side are vertically distributed
            left_lines = [r for r in text_regions if r['center_x'] < avg_split_x]
            right_lines = [r for r in text_regions if r['center_x'] >= avg_split_x]
            
            if len(left_lines) >= 2 and len(right_lines) >= 2:
                # Check vertical distribution
                left_y_range = max(r['center_y'] for r in left_lines) - min(r['center_y'] for r in left_lines)
                right_y_range = max(r['center_y'] for r in right_lines) - min(r['center_y'] for r in right_lines)
                
                # Both sides should have reasonable vertical spread (at least 15% of image height)
                min_vertical_spread = image_height * 0.15
                if left_y_range > min_vertical_spread and right_y_range > min_vertical_spread:
                    # Ensure split is not too close to edges
                    min_margin = image_width * 0.1
                    if min_margin < avg_split_x < image_width - min_margin:
                        return True, avg_split_x
    
    return False, None


def determine_reading_order(
    patch_positions: List[Tuple[int, int]],
    patch_size: int,
    text_regions: List[Dict],
    image_width: int,
    image_height: int,
    rtl: bool = True,
) -> List[int]:
    """
    Determine reading order for patches (top-to-bottom, right-to-left for RTL).
    Handles both single-page and two-page layouts.
    
    Args:
        patch_positions: List of (top, left) tuples for each patch
        patch_size: Size of patches
        text_regions: List of text region dicts (for page detection)
        image_width: Width of the image
        image_height: Height of the image
        rtl: If True, reading order is right-to-left (default for Hebrew)
        
    Returns:
        List of indices representing the reading order (permutation of [0, 1, ..., len(patch_positions)-1])
    """
    if not patch_positions:
        return []
    
    # Detect page layout
    is_two_page, split_x = detect_page_layout(text_regions, image_width, image_height)
    
    # Calculate patch centers for sorting
    patch_data = []
    for idx, (top, left) in enumerate(patch_positions):
        center_x = left + patch_size / 2.0
        center_y = top + patch_size / 2.0
        patch_data.append({
            'idx': idx,
            'top': top,
            'left': left,
            'center_x': center_x,
            'center_y': center_y,
        })
    
    if is_two_page and split_x is not None:
        # Two-page layout: sort each page separately
        # For RTL: right page first, then left page
        right_page = [p for p in patch_data if p['center_x'] >= split_x]
        left_page = [p for p in patch_data if p['center_x'] < split_x]
        
        # Sort each page: top-to-bottom, then right-to-left (for RTL)
        def sort_key_rtl(p):
            # Primary: vertical position (top to bottom)
            # Secondary: horizontal position (right to left, so higher x first)
            return (p['center_y'], -p['center_x'])
        
        def sort_key_ltr(p):
            # Primary: vertical position (top to bottom)
            # Secondary: horizontal position (left to right)
            return (p['center_y'], p['center_x'])
        
        sort_key = sort_key_rtl if rtl else sort_key_ltr
        
        right_page_sorted = sorted(right_page, key=sort_key)
        left_page_sorted = sorted(left_page, key=sort_key)
        
        # Combine: right page first for RTL, left page first for LTR
        if rtl:
            ordered_patches = right_page_sorted + left_page_sorted
        else:
            ordered_patches = left_page_sorted + right_page_sorted
    else:
        # Single-page layout: sort all patches
        # Top-to-bottom, then right-to-left (for RTL) or left-to-right (for LTR)
        if rtl:
            # Primary: vertical position, Secondary: horizontal position (right to left)
            ordered_patches = sorted(patch_data, key=lambda p: (p['center_y'], -p['center_x']))
        else:
            # Primary: vertical position, Secondary: horizontal position (left to right)
            ordered_patches = sorted(patch_data, key=lambda p: (p['center_y'], p['center_x']))
    
    # Return indices in reading order
    return [p['idx'] for p in ordered_patches]


def page_segment_for_center_x(center_x: float, split_x: float, rtl: bool = True) -> int:
    """
    Return reading-order page segment for a two-page spread.

    Segment 0 is the first page in reading order. For Hebrew/RTL spreads that is
    the right page; for LTR spreads it is the left page.
    """
    is_right = center_x >= split_x
    if rtl:
        return 0 if is_right else 1
    return 0 if not is_right else 1


def get_text_region_bounds(text_regions: List[Dict], expanded: bool = False, patch_size: int = None) -> Optional[Tuple[float, float, float, float]]:
    """
    Get bounding box that encompasses all text regions.
    Useful for determining if image has two pages.
    
    Args:
        text_regions: List of text region dicts
        expanded: If True, return bounds expanded by XML_PATCH_EXPAND_RATIO * patch_size
        patch_size: Patch size for expansion calculation (required if expanded=True)
        
    Returns:
        Tuple of (min_x, min_y, max_x, max_y) or None if no regions
    """
    if not text_regions:
        return None
    
    min_x = min(r['hpos'] for r in text_regions)
    min_y = min(r['vpos'] for r in text_regions)
    max_x = max(r['hpos'] + r['width'] for r in text_regions)
    max_y = max(r['vpos'] + r['height'] for r in text_regions)
    
    if expanded and patch_size is not None:
        expand = int(patch_size * XML_PATCH_EXPAND_RATIO)
        min_x = max(0, min_x - expand)
        min_y = max(0, min_y - expand)
        max_x = max_x + expand
        max_y = max_y + expand
    
    return (min_x, min_y, max_x, max_y)


def extract_patches_with_xml(
    image: Image.Image,
    image_path: str,
    patch_size: int,
    stride: int,
    max_patches: Optional[int] = None,
    xml_path: Optional[str] = None,
) -> Tuple[List[Image.Image], torch.Tensor, torch.Tensor, Dict]:
    """
    Extract patches from image using XML-based text region information.
    
    Args:
        image: PIL Image
        image_path: Path to image file (used to find XML)
        patch_size: Size of patches to extract
        stride: Stride for patch extraction
        max_patches: Maximum number of patches to return
        xml_path: Optional explicit path to XML file. If None, will try to find it.
        
    Returns:
        Tuple of (patches, coords, page_segments, metadata) where:
        - patches: List of PIL Images (will be transformed by dataset)
        - coords: Tensor of shape [num_patches, 2] with normalized coordinates
        - page_segments: Tensor of shape [num_patches] (0 or 1 for two-page layout)
        - metadata: Dict with 'text_regions_count', 'image_size', 'text_bounds', etc.
    """
    w, h = image.size
    original_size = (w, h)
    rotation_angle = 0
    
    # Find XML file - try pretrain path first, then fall back to local
    if xml_path is None:
        from utilities.xml_loader import find_xml_path_pretrain
        # Try pretrain path first (for pretrain dataset)
        xml_path = find_xml_path_pretrain(image_path)
        # Fall back to local search if pretrain path doesn't find it
        if xml_path is None:
            xml_path = find_xml_path(image_path)
    
    # Parse text regions from XML
    text_regions, xml_image_size = parse_alto_text_regions(xml_path) if xml_path else ([], (w, h))

    # IMPORTANT: If there are no text regions, do NOT do any grid-based extraction here.
    # The dataset will detect text_regions_count==0 and fall back to its (faster) center-biased
    # grid extraction. Doing grid extraction twice wastes a lot of time and can look like a "hang"
    # at the start of an epoch when many samples have missing/empty XML.
    if not text_regions:
        empty_coords = torch.empty((0, 2), dtype=torch.float32)
        empty_page_segments = torch.empty(0, dtype=torch.long)
        metadata = {
            'text_regions_count': 0,
            'image_size': (w, h),
            'text_bounds': None,
            'text_bounds_expanded': None,
            'effective_bounds': None,
            'xml_path': xml_path,
            'rotation_angle': 0,
            'rotation_corrected': False,
        }
        return [], empty_coords, empty_page_segments, metadata
    
    # Use XML image size if available and different from actual image
    if xml_image_size[0] > 0 and xml_image_size[1] > 0:
        # Check if image needs scaling
        if xml_image_size != (w, h):
            # Scale text regions to match actual image size
            scale_x = w / xml_image_size[0]
            scale_y = h / xml_image_size[1]
            for region in text_regions:
                region['hpos'] *= scale_x
                region['vpos'] *= scale_y
                region['width'] *= scale_x
                region['height'] *= scale_y
                region['center_x'] *= scale_x
                region['center_y'] *= scale_y
                # Scale baseline coordinates if present
                if region.get('baseline') is not None:
                    x1, y1, x2, y2 = region['baseline']
                    region['baseline'] = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
    
    # Detect and correct rotation if enabled
    rotation_angle = 0
    if XML_PATCH_DETECT_ROTATION and text_regions:
        rotation_angle = detect_page_rotation(text_regions, w, h)
        
        if rotation_angle != 0:
            # Apply rotation correction using the dedicated rotation module
            # This transforms:
            # 1. The image (rotates it)
            # 2. The text_regions coordinates (transforms them to match rotated image)
            # After this, text_regions are in rotated image space, not original space
            old_w, old_h = w, h
            image, text_regions = apply_rotation_correction(image, text_regions, rotation_angle)
            # Update image dimensions after rotation
            w, h = image.size  # w, h are now rotated dimensions
    
    # Get text bounds for metadata (both original and expanded)
    text_bounds = get_text_region_bounds(text_regions)
    text_bounds_expanded = get_text_region_bounds(text_regions, expanded=True, patch_size=patch_size)
    
    # Extract patch positions
    # Use system config to determine if patches should be constrained to original OCR text boundaries
    positions, coords, effective_bounds = extract_patches_from_text_regions(
        image, text_regions, patch_size, stride, max_patches, (w, h),
        constrain_to_bounds=XML_PATCH_CONSTRAIN_TO_BOUNDS
    )
    
    if not positions:
        empty_coords = torch.empty((0, 2), dtype=torch.float32)
        empty_page_segments = torch.empty(0, dtype=torch.long)
        metadata = {
            'text_regions_count': len(text_regions),
            'image_size': (w, h),
            'original_image_size': original_size,
            'rotation_angle': rotation_angle,
            'rotation_corrected': rotation_angle != 0,
            'text_bounds': text_bounds,
            'text_bounds_expanded': text_bounds_expanded,
            'text_bounds_effective': effective_bounds,
            'xml_image_size': xml_image_size,
            'xml_path': xml_path,
            'num_patches': 0,
            'reading_order_applied': False,
            'reading_direction_rtl': XML_PATCH_READING_DIRECTION_RTL,
            'is_two_page': False,
            'page_split_x': None,
            'patch_positions': [],
            'no_patch_reason': 'no_candidates_after_text_coverage_filter',
        }
        return [], empty_coords, empty_page_segments, metadata

    # Extract actual patches
    patches = []
    for top, left in positions:
        box = (left, top, min(left + patch_size, w), min(top + patch_size, h))
        patch = image.crop(box)
        if patch.size[0] != patch_size or patch.size[1] != patch_size:
            patch = patch.resize((patch_size, patch_size))
        patches.append(patch)
    
    # Apply reading order if enabled
    if XML_PATCH_APPLY_READING_ORDER and len(patches) > 1 and text_regions:
        reading_order_indices = determine_reading_order(
            positions, patch_size, text_regions, w, h, rtl=XML_PATCH_READING_DIRECTION_RTL
        )
        
        # Reorder patches, coords, and positions according to reading order
        patches = [patches[i] for i in reading_order_indices]
        coords = [coords[i] for i in reading_order_indices]
        positions = [positions[i] for i in reading_order_indices]
    
    # Detect page layout for metadata
    is_two_page, split_x = detect_page_layout(text_regions, w, h) if text_regions else (False, None)
    
    # Assign page segments (0 for first page, 1 for second)
    page_segments = []
    if is_two_page and split_x is not None:
        for top, left in positions:
            cx = left + patch_size / 2.0
            page_segments.append(
                page_segment_for_center_x(
                    cx,
                    split_x,
                    rtl=XML_PATCH_READING_DIRECTION_RTL,
                )
            )
    else:
        page_segments = [0] * len(positions)
    
    # Convert to tensors (patches will be converted to tensors by transform later)
    # For now, return as list of PIL Images
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    page_segments_tensor = torch.tensor(page_segments, dtype=torch.long)
    
    metadata = {
        'text_regions_count': len(text_regions),
        'image_size': (w, h),
        'original_image_size': original_size,
        'rotation_angle': rotation_angle,
        'rotation_corrected': rotation_angle != 0,
        'text_bounds': text_bounds,
        'text_bounds_expanded': text_bounds_expanded,
        'text_bounds_effective': effective_bounds,
        'xml_image_size': xml_image_size,
        'xml_path': xml_path,
        'num_patches': len(patches),
        'reading_order_applied': XML_PATCH_APPLY_READING_ORDER and len(patches) > 1 and text_regions,
        'reading_direction_rtl': XML_PATCH_READING_DIRECTION_RTL,
        'is_two_page': is_two_page,
        'page_split_x': split_x,
        'patch_positions': positions,  # List of (top, left) tuples for visualization/debugging
    }
    
    return patches, coords_tensor, page_segments_tensor, metadata
