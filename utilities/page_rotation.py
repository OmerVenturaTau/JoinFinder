"""
Page rotation detection and correction for manuscript images.

This module provides functions to detect page rotation using XML baseline information
and to apply rotation corrections with proper coordinate transformations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def detect_page_rotation(text_regions: List[Dict], image_width: float, image_height: float) -> int:
    """
    Detect if the page is rotated by analyzing text line orientations and baselines.
    Uses baseline information for Hebrew text (RTL) to determine correct orientation.
    
    For Hebrew text (ALTO baseline caveat):
    Many ALTO producers encode baselines left-to-right regardless of script direction.
    So baseline *direction* (0° vs 180°) is often NOT reliable for deciding 180° flips.
    We primarily use baselines to detect vertical rotations (90/270) vs horizontal (0).
    
    - Correctly oriented: baseline is horizontal (either ~0° or ~180°)
    - Rotated 90°: baseline is vertical, going from top to bottom (~90°)
    - Rotated 270°: baseline is vertical, going from bottom to top (~270°)
    - Rotated 180°: NOT inferred from baseline direction alone (too error-prone)
    
    Args:
        text_regions: List of text region dicts with 'width', 'height', and optionally 'baseline'
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        Rotation angle in degrees (0, 90, 180, or 270) needed to correct the orientation.
        Returns 0 if no rotation is detected or if detection is uncertain.
    """
    if not text_regions or len(text_regions) < 3:
        return 0
    
    if image_width <= 0 or image_height <= 0:
        return 0
    
    # First, try to use baseline information if available (most reliable for Hebrew)
    baselines_with_angles = []
    for region in text_regions:
        baseline = region.get('baseline')
        if baseline is not None:
            x1, y1, x2, y2 = baseline
            # Calculate baseline vector
            dx = x2 - x1
            dy = y2 - y1
            
            # Calculate angle in degrees (0° = horizontal right, 90° = vertical down)
            # Use atan2 to get angle in range [-180, 180]
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            # Normalize to [0, 360)
            if angle_deg < 0:
                angle_deg += 360
            
            baselines_with_angles.append(angle_deg)
    
    # If we have baseline information, use it for robust detection
    if len(baselines_with_angles) >= 3:
        # Analyze baseline angles
        # For correctly oriented Hebrew text, baselines should be ~180° (horizontal, RTL)
        # For rotated text:
        #   - 90° rotation: baselines ~90° or ~270° (vertical)
        #   - 180° rotation: baselines ~0° (horizontal, LTR)
        #   - 270° rotation: baselines ~90° or ~270° (vertical, opposite direction)
        
        # Count baselines in different angle ranges
        horizontal_rtl = 0  # ~180° (correct for Hebrew)
        horizontal_ltr = 0   # ~0° or ~360° (180° rotated)
        vertical_down = 0    # ~90° (90° or 270° rotated)
        vertical_up = 0      # ~270° (90° or 270° rotated, opposite)
        
        for angle in baselines_with_angles:
            # Normalize angle to [0, 360)
            angle_norm = angle % 360
            
            # Check if horizontal (within ±30° of 0° or 180°)
            if abs(angle_norm - 0) < 30 or abs(angle_norm - 360) < 30:
                horizontal_ltr += 1
            elif abs(angle_norm - 180) < 30:
                horizontal_rtl += 1
            # Check if vertical (within ±30° of 90° or 270°)
            elif abs(angle_norm - 90) < 30:
                vertical_down += 1
            elif abs(angle_norm - 270) < 30:
                vertical_up += 1
        
        total_baselines = len(baselines_with_angles)
        
        # Decision based on baseline angles
        if (horizontal_rtl + horizontal_ltr) / total_baselines > 0.6:
            # Baselines are horizontal. Check for 180° rotation by analyzing
            # baseline *direction* of major text lines (wide lines only).
            # ALTO typically encodes baselines left-to-right regardless of script.
            # If major text lines have RTL baselines (~180°), the page is likely
            # upside down — the LTR encoding got reversed by the 180° flip.
            major_rtl = 0
            major_ltr = 0
            min_major_width = image_width * 0.3
            
            for region in text_regions:
                if region.get('width', 0) < min_major_width:
                    continue
                baseline = region.get('baseline')
                if baseline is None:
                    continue
                bx1, _, bx2, _ = baseline
                dx = bx2 - bx1
                if dx < 0:
                    major_rtl += 1
                elif dx > 0:
                    major_ltr += 1
            
            total_major = major_ltr + major_rtl
            if total_major >= 3 and major_rtl / total_major > 0.7:
                logger.info(
                    f"Detected 180° rotation: {major_rtl}/{total_major} major text lines "
                    f"have RTL baselines (>70% threshold)"
                )
                return 180
            
            return 0
        elif (vertical_down + vertical_up) / total_baselines > 0.6:
            # Most baselines are vertical - page is rotated 90° or 270°
            # For Hebrew RTL text, correct baselines are at ~180° (horizontal, RTL)
            # 
            # If baselines are at ~90° (pointing down): text is vertical top-to-bottom
            #   This means the page is rotated 270° clockwise from correct
            #   To correct: rotate 90° clockwise → return 270 (which triggers rotate(90))
            #
            # If baselines are at ~270° (pointing up): text is vertical bottom-to-top  
            #   This means the page is rotated 90° counter-clockwise from correct
            #   To correct: rotate -90° (90° CCW) → return 90 (which triggers rotate(-90))
            
            # Use baseline direction to determine rotation
            if vertical_down > vertical_up:
                # Most baselines point down (~90°) - page rotated 270° CW from correct
                # Need to rotate 90° CW to correct → return 270 (which triggers rotate(90))
                return 270
            else:
                # Most baselines point up (~270°) - page rotated 90° CCW from correct  
                # Need to rotate -90° (90° CCW) to correct → return 90 (which triggers rotate(-90))
                return 90
    
    # Fallback to aspect ratio method if baseline info not available
    # Analyze text line aspect ratios
    # Normal text lines should be wider than tall (width > height)
    # Rotated text lines will be taller than wide (height > width)
    
    aspect_ratios = []
    for region in text_regions:
        width = region.get('width', 0)
        height = region.get('height', 0)
        
        if width > 0 and height > 0:
            aspect_ratio = width / height
            aspect_ratios.append(aspect_ratio)
    
    if len(aspect_ratios) < 3:
        return 0
    
    # Count how many text lines are horizontal vs vertical
    # Horizontal: aspect_ratio > 1 (wider than tall)
    # Vertical: aspect_ratio < 1 (taller than wide)
    horizontal_count = sum(1 for ar in aspect_ratios if ar > 1.2)  # Clearly horizontal
    vertical_count = sum(1 for ar in aspect_ratios if ar < 0.8)    # Clearly vertical
    
    total_count = len(aspect_ratios)
    horizontal_ratio = horizontal_count / total_count
    vertical_ratio = vertical_count / total_count
    
    # Check image aspect ratio to help determine rotation direction
    image_aspect = image_width / image_height if image_height > 0 else 1.0
    
    if vertical_ratio > 0.6:
        # Most text lines are vertical - page is rotated 90° or 270°
        if image_aspect > 1.2:
            # Image is wide, text is vertical -> rotated 270° clockwise -> need 90° to correct
            return 90
        elif image_aspect < 0.8:
            # Image is tall, text is vertical -> rotated 90° clockwise -> need 270° to correct
            return 270
        else:
            # Ambiguous - check text line distribution
            centers_x = [r['center_x'] for r in text_regions]
            avg_x = sum(centers_x) / len(centers_x)
            # If average x is in right half, likely 270° clockwise
            if avg_x > image_width * 0.6:
                return 270
            else:
                return 90
    
    elif horizontal_ratio > 0.6:
        # Most text lines are horizontal - page might be correct or rotated 180°
        # Get vertical distribution of text lines
        centers_y = [r['center_y'] for r in text_regions]
        avg_y = sum(centers_y) / len(centers_y)
        
        # If most text is in bottom half, might be rotated 180°
        if avg_y > image_height * 0.6:
            if image_aspect < 0.7:
                return 180
        # Otherwise assume correct orientation
        return 0
    
    # If unclear, return 0 (no rotation)
    return 0


def rotate_image(image: Image.Image, rotation_angle: int) -> Image.Image:
    """
    Rotate an image by the specified angle.
    
    Args:
        image: PIL Image to rotate
        rotation_angle: Rotation angle in degrees (90, 180, or 270)
        
    Returns:
        Rotated PIL Image
    """
    if rotation_angle == 0:
        return image
    elif rotation_angle == 90:
        # Rotate -90° (90° CCW) to correct 90° rotation
        return image.rotate(-90, expand=True)
    elif rotation_angle == 270:
        # Rotate 90° (90° CW) to correct 270° rotation
        return image.rotate(90, expand=True)
    elif rotation_angle == 180:
        return image.rotate(180, expand=True)
    else:
        logger.warning(f"Unsupported rotation angle: {rotation_angle}. Returning original image.")
        return image


def transform_coordinates(
    text_regions: List[Dict],
    rotation_angle: int,
    old_width: int,
    old_height: int
) -> None:
    """
    Transform text region coordinates to match rotated image.
    Modifies the text_regions list in place.
    
    Args:
        text_regions: List of text region dicts to transform
        rotation_angle: Rotation angle in degrees (90, 180, or 270)
        old_width: Original image width before rotation
        old_height: Original image height before rotation
    """
    if rotation_angle == 0:
        return
    
    for region in text_regions:
        old_hpos = region['hpos']
        old_vpos = region['vpos']
        old_width_box = region['width']
        old_height_box = region['height']
        
        if rotation_angle == 90:
            # Image rotated -90° (90° CCW) to correct 90° rotation
            # Coordinate transformation for 90° CCW rotation:
            # Point (x, y) in original -> (old_h - y, x) in rotated
            # For a box: top-left (hpos, vpos) with size (width, height)
            # New top-left: (old_h - (vpos + height), hpos)
            # New size: (height, width)
            region['hpos'] = old_height - (old_vpos + old_height_box)
            region['vpos'] = old_hpos
            region['width'] = old_height_box
            region['height'] = old_width_box
            region['center_x'] = region['hpos'] + region['width'] / 2.0
            region['center_y'] = region['vpos'] + region['height'] / 2.0
            
            # Transform baseline coordinates
            if region.get('baseline') is not None:
                x1, y1, x2, y2 = region['baseline']
                # Transform both points
                new_x1 = old_height - y1
                new_y1 = x1
                new_x2 = old_height - y2
                new_y2 = x2
                region['baseline'] = (new_x1, new_y1, new_x2, new_y2)
                
        elif rotation_angle == 270:
            # Image rotated 90° (270° CCW) to correct 270° rotation
            # Coordinate transformation for 90° CW rotation:
            # Point (x, y) in original -> (y, old_w - x) in rotated
            # For a box: top-left (hpos, vpos) with size (width, height)
            # New top-left: (vpos, old_w - (hpos + width))
            # New size: (height, width)
            region['hpos'] = old_vpos
            region['vpos'] = old_width - (old_hpos + old_width_box)
            region['width'] = old_height_box
            region['height'] = old_width_box
            region['center_x'] = region['hpos'] + region['width'] / 2.0
            region['center_y'] = region['vpos'] + region['height'] / 2.0
            
            # Transform baseline coordinates
            if region.get('baseline') is not None:
                x1, y1, x2, y2 = region['baseline']
                # Transform both points
                new_x1 = y1
                new_y1 = old_width - x1
                new_x2 = y2
                new_y2 = old_width - x2
                region['baseline'] = (new_x1, new_y1, new_x2, new_y2)
                
        elif rotation_angle == 180:
            # Image rotated 180° to correct 180° rotation
            # Coordinate transformation for 180° rotation:
            # Point (x, y) in original -> (old_w - x, old_h - y) in rotated
            # For a box: top-left (hpos, vpos) with size (width, height)
            # New top-left: (old_w - (hpos + width), old_h - (vpos + height))
            # New size: (width, height) - same
            region['hpos'] = old_width - (old_hpos + old_width_box)
            region['vpos'] = old_height - (old_vpos + old_height_box)
            # width and height stay the same
            region['center_x'] = region['hpos'] + region['width'] / 2.0
            region['center_y'] = region['vpos'] + region['height'] / 2.0
            
            # Transform baseline coordinates
            if region.get('baseline') is not None:
                x1, y1, x2, y2 = region['baseline']
                # Transform both points
                new_x1 = old_width - x1
                new_y1 = old_height - y1
                new_x2 = old_width - x2
                new_y2 = old_height - y2
                region['baseline'] = (new_x1, new_y1, new_x2, new_y2)


def transform_point(x: float, y: float, rotation_angle: int, old_width: int, old_height: int) -> Tuple[float, float]:
    """
    Transform a single point coordinate after image rotation.
    
    Args:
        x: Original x coordinate
        y: Original y coordinate
        rotation_angle: Rotation angle in degrees (90, 180, or 270)
        old_width: Original image width before rotation
        old_height: Original image height before rotation
        
    Returns:
        Tuple of (new_x, new_y) coordinates
    """
    if rotation_angle == 0:
        return (x, y)
    elif rotation_angle == 90:
        # 90° CCW rotation: (x, y) -> (old_h - y, x)
        return (old_height - y, x)
    elif rotation_angle == 270:
        # 90° CW rotation: (x, y) -> (y, old_w - x)
        return (y, old_width - x)
    elif rotation_angle == 180:
        # 180° rotation: (x, y) -> (old_w - x, old_h - y)
        return (old_width - x, old_height - y)
    else:
        return (x, y)


def apply_rotation_correction(
    image: Image.Image,
    text_regions: List[Dict],
    rotation_angle: int
) -> Tuple[Image.Image, List[Dict]]:
    """
    Apply rotation correction to image and transform coordinates.
    
    This is a convenience function that combines rotate_image and transform_coordinates.
    
    Args:
        image: PIL Image to rotate
        text_regions: List of text region dicts to transform
        rotation_angle: Rotation angle in degrees (90, 180, or 270)
        
    Returns:
        Tuple of (rotated_image, transformed_text_regions)
        Note: text_regions is modified in place, but returned for convenience
    """
    if rotation_angle == 0:
        return image, text_regions
    
    old_width, old_height = image.size
    
    # Rotate image
    rotated_image = rotate_image(image, rotation_angle)
    
    # Transform coordinates
    transform_coordinates(text_regions, rotation_angle, old_width, old_height)
    
    return rotated_image, text_regions

