"""
XML path resolution utility shared across modules.

Training uses xml_path from the DB. This module provides find_xml_path_pretrain() as a fallback
when a row has no xml_path (or for scripts that don't use the DB). With XML_BASE_PATH set it builds
paths under that base; otherwise it looks next to the image file (_find_xml_local).
"""

import os
from typing import Optional
import system  # Import module, not variables, so we get the latest values at runtime


def find_xml_path_pretrain(image_path: str) -> Optional[str]:
    """
    Find the corresponding ALTO XML file for an image path.

    When system.XML_BASE_PATH is set: builds path as
      XML_BASE_PATH/manuscript_id/parent_directory/picture_id_improved_polys.xml
    (with image_path under BASE_DIR). Otherwise falls back to _find_xml_local (same dir as image).

    Training normally uses xml_path from the DB; this is only used when a row has no xml_path.
    """
    xml_base_path = getattr(system, "XML_BASE_PATH", None)
    if not xml_base_path or not xml_base_path.strip():
        return _find_xml_local(image_path)

    base_dir = system.BASE_DIR.rstrip("/")
    if not image_path.startswith(base_dir):
        return _find_xml_local(image_path)

    rel_path = os.path.relpath(image_path, base_dir)
    parts = rel_path.split(os.sep)
    if len(parts) < 3:
        return _find_xml_local(image_path)

    manuscript_id = parts[0]
    parent_directory = parts[1]
    picture_id = os.path.splitext(parts[-1])[0]

    xml_path = os.path.join(
        xml_base_path,
        manuscript_id,
        parent_directory,
        f"{picture_id}_improved_polys.xml",
    )
    if os.path.exists(xml_path):
        return xml_path

    alt_paths = [
        os.path.join(xml_base_path, manuscript_id, parent_directory, f"{picture_id}—reco_improved_polys_improved_reading_order.xml"),
        os.path.join(xml_base_path, manuscript_id, parent_directory, f"{picture_id}.xml"),
    ]
    for p in alt_paths:
        if os.path.exists(p):
            return p

    return _find_xml_local(image_path)


def _find_xml_local(image_path: str) -> Optional[str]:
    """Fallback only when image is not under BASE_DIR (e.g. non-pretrain use)."""
    image_dir = os.path.dirname(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

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
