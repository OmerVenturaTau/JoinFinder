"""
ALTO XML parsing utilities for extracting strings, glyphs, and text lines.

This module provides the core parsing functionality extracted from alto_visualize.py,
with proper Hebrew dictionary integration and threshold filtering.
"""

import os
import sys
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add project root to path
# Use __file__ to get absolute path regardless of working directory
_this_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(_this_file)))
# Ensure project_root is absolute
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system import OCR_GLYPH_CONFIDENCE_THRESHOLD, OCR_STRING_CONFIDENCE_THRESHOLD


def _get_local(tag: str) -> str:
    """Extract local tag name from namespaced XML tag."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def _is_hebrew(s: str) -> bool:
    """Check if string contains Hebrew characters."""
    for ch in s:
        if '\u0590' <= ch <= '\u05FF':
            return True
    return False


def _safe_float(x: Optional[str]) -> Optional[float]:
    """Safely convert string to float."""
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _parse_points(points_str: str) -> List[Tuple[float, float]]:
    """
    Robust ALTO POINTS parser. Handles "x,y x,y ..." and "x y x y ..." forms.
    """
    parts = points_str.strip().replace(',', ' ').split()
    pts: List[Tuple[float, float]] = []
    for i in range(0, len(parts), 2):
        try:
            x = float(parts[i])
            y = float(parts[i + 1])
            pts.append((x, y))
        except (IndexError, ValueError):
            break
    return pts


def _polygon_bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Get bounding box from polygon points: (min_x, min_y, max_x, max_y)."""
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


class TextLineInfo:
    """Information about a text line."""
    def __init__(self, line_id: str, polygon: Optional[List[Tuple[float, float]]], baseline: Optional[str]):
        self.line_id = line_id
        self.polygon = polygon
        self.baseline = baseline


class GlyphInfo:
    """Information about a single glyph (character)."""
    def __init__(
        self,
        char: str,
        gc: Optional[float],
        hpos: Optional[float],
        vpos: Optional[float],
        width: Optional[float],
        height: Optional[float],
        polygon: Optional[List[Tuple[float, float]]] = None
    ):
        self.char = char
        self.gc = gc
        self.hpos = hpos
        self.vpos = vpos
        self.width = width
        self.height = height
        self.polygon = polygon


class StringInfo:
    """Information about a string (word)."""
    def __init__(
        self,
        content: str,
        hpos: Optional[float],
        vpos: Optional[float],
        width: Optional[float],
        height: Optional[float],
        wc: Optional[float],
        glyphs: List[GlyphInfo],
        polygon: Optional[List[Tuple[float, float]]] = None,
        textline: Optional[TextLineInfo] = None
    ):
        self.content = content
        self.hpos = hpos
        self.vpos = vpos
        self.width = width
        self.height = height
        self.wc = wc
        self.glyphs = glyphs
        self.polygon = polygon
        self.textline = textline


class HebrewDictChecker:
    """Checker for Hebrew words using unified_word_check.py logic."""
    
    def __init__(self):
        # Try to import unified_word_check from utilities/HebrewDict
        # Use absolute paths to avoid any path issues (important when running with nohup from different directory)
        _HEBREW_DICT_DIR = os.path.abspath(os.path.join(project_root, 'utilities', 'HebrewDict'))
        _HEBREW_DICT_SCRIPT = os.path.abspath(os.path.join(_HEBREW_DICT_DIR, 'unified_word_check.py'))
        _HAS_DICT_MODULE = False
        do_check = None
        has_hebrew = None
        import_error = None
        
        # Debug: Log path information (helps when running with nohup)
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"HebrewDictChecker initialization:")
        logger.debug(f"  Project root: {project_root}")
        logger.debug(f"  Dictionary dir: {_HEBREW_DICT_DIR}")
        logger.debug(f"  Script path: {_HEBREW_DICT_SCRIPT}")
        logger.debug(f"  Current working directory: {os.getcwd()}")
        logger.debug(f"  __file__ location: {os.path.abspath(__file__)}")
        
        # Check if the script file exists
        if not os.path.exists(_HEBREW_DICT_SCRIPT):
            # Try alternative path resolution
            alt_dict_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utilities', 'HebrewDict')
            alt_script = os.path.join(alt_dict_dir, 'unified_word_check.py')
            
            error_msg = (
                f"Hebrew dictionary script not found.\n"
                f"  Expected path: {_HEBREW_DICT_SCRIPT}\n"
                f"  Dictionary directory: {_HEBREW_DICT_DIR}\n"
                f"  Directory exists: {os.path.exists(_HEBREW_DICT_DIR)}\n"
                f"  Project root: {project_root}\n"
                f"  Current working directory: {os.getcwd()}\n"
                f"  __file__ location: {os.path.abspath(__file__)}\n"
            )
            if os.path.exists(alt_script):
                error_msg += f"  Alternative path found: {alt_script}\n"
            if os.path.exists(_HEBREW_DICT_DIR):
                error_msg += f"  Files in dict dir: {os.listdir(_HEBREW_DICT_DIR)}\n"
            else:
                error_msg += f"  Dictionary directory does not exist\n"
            raise RuntimeError(error_msg)
        
        # Add the directory to sys.path for importing (use absolute path)
        if _HEBREW_DICT_DIR not in sys.path:
            sys.path.insert(0, _HEBREW_DICT_DIR)
        
        # Try to import the module
        try:
            # Import with better error handling
            import unified_word_check  # type: ignore
            do_check = getattr(unified_word_check, 'do_check', None)
            has_hebrew = getattr(unified_word_check, 'has_hebrew', None)
            
            # Verify the required functions exist
            if do_check is None:
                raise RuntimeError("unified_word_check.do_check function not found in module")
            if has_hebrew is None:
                raise RuntimeError("unified_word_check.has_hebrew function not found in module")
            
            # Test that do_check is callable
            if not callable(do_check):
                raise RuntimeError(f"unified_word_check.do_check is not callable (type: {type(do_check)})")
            if not callable(has_hebrew):
                raise RuntimeError(f"unified_word_check.has_hebrew is not callable (type: {type(has_hebrew)})")
            
            _HAS_DICT_MODULE = True
        except ImportError as e:
            import_error = e
            _HAS_DICT_MODULE = False
            # Check if it's the tf.fabric import that's failing
            error_str = str(e)
            if 'tf' in error_str or 'fabric' in error_str.lower():
                raise RuntimeError(
                    f"Failed to import unified_word_check.py: {e}\n"
                    f"  The module requires 'tf.fabric' which may not be installed.\n"
                    f"  However, do_check() and has_hebrew() functions don't require tf.fabric.\n"
                    f"  Consider modifying unified_word_check.py to make tf.fabric import optional."
                ) from e
        except AttributeError as e:
            import_error = e
            _HAS_DICT_MODULE = False
        except Exception as e:
            import_error = e
            _HAS_DICT_MODULE = False
        
        # If module is not available, raise an error with details
        if not _HAS_DICT_MODULE or do_check is None:
            abs_script = os.path.abspath(_HEBREW_DICT_SCRIPT)
            abs_dir = os.path.abspath(_HEBREW_DICT_DIR)
            error_msg = (
                f"Failed to load Hebrew dictionary module.\n"
                f"  Script path: {abs_script}\n"
                f"  Script exists: {os.path.exists(abs_script)}\n"
                f"  Dictionary directory: {abs_dir}\n"
                f"  Directory in sys.path: {abs_dir in sys.path}\n"
            )
            if import_error:
                error_msg += f"  Import error: {import_error}\n"
            if do_check is None:
                error_msg += f"  do_check function: None\n"
            if has_hebrew is None:
                error_msg += f"  has_hebrew function: None\n"
            raise RuntimeError(error_msg)
        
        # Store references
        self._do_check = do_check
        self._has_hebrew = has_hebrew
        self._is_available = True
        
        # Load wordsets
        self.wordsets: List[Dict] = []
        script_dir = Path(_HEBREW_DICT_DIR)
        default_paths = [
            script_dir / "bhsa_wordset.json",
            script_dir / "mh_wordset.json",
        ]
        ws_list = [str(p) for p in default_paths if p.exists()]
        
        for ws in ws_list:
            p = Path(ws)
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text(encoding='utf-8'))
                d['name'] = p.stem
                # Pre-cache membership sets for fast repeated lookups
                # (do_check() will also cache on demand, but doing it once here avoids any first-call hiccup)
                try:
                    d["__set_hebrew"] = set(d.get("hebrew", []) or [])
                    d["__set_hebrew_consonants"] = set(d.get("hebrew_consonants", []) or [])
                    d["__set_trans"] = set(d.get("trans", []) or [])
                    d["__set_trans_cons"] = set(d.get("trans_cons", []) or [])
                except Exception as e:
                    logging.debug(f"Could not create cached sets for wordset {p}: {type(e).__name__}: {e}")
                self.wordsets.append(d)
            except Exception as e:
                logging.warning(f"Could not load wordset {p}: {type(e).__name__}: {e}", exc_info=True)
        
        # If no wordsets found, raise an error
        if not self.wordsets:
            abs_dict_dir = os.path.abspath(_HEBREW_DICT_DIR)
            raise RuntimeError(
                f"No Hebrew dictionary wordsets found.\n"
                f"  Dictionary directory: {abs_dict_dir}\n"
                f"  Expected files: bhsa_wordset.json, mh_wordset.json\n"
                f"  Files found: {[f.name for f in Path(abs_dict_dir).glob('*.json')]}"
            )
    
    def check_word(self, word: str) -> bool:
        """Check if word exists in dictionary using existing do_check function."""
        # If dictionary is not available, always return False (word not found)
        if not self._is_available or self._do_check is None:
            return False
        
        if not word:
            return False
        
        # Check if it's Hebrew using the imported function or fallback
        if self._has_hebrew:
            if not self._has_hebrew(word):
                return False
        elif not _is_hebrew(word):
            return False
        
        # Use the existing do_check function directly
        result = self._do_check(word, self.wordsets)
        return result.get("found", False)

    def lookup_word(self, word: str) -> Dict:
        """
        Return full lookup info (existence + optional frequency fields if present in the wordset JSON).
        This is still fast: it only does cached set membership + dict lookups.
        """
        if not self._is_available or self._do_check is None:
            return {"word": word, "found": False, "matches": []}
        return self._do_check(word, self.wordsets)
    
    def is_available(self) -> bool:
        """Check if dictionary checking is available."""
        return self._is_available


def parse_alto_strings(alto_xml_path: str) -> List[StringInfo]:
    """
    Parse ALTO XML file and extract strings (words) with their glyphs.
    
    This is the main parsing function that extracts all String elements
    from an ALTO XML file, along with their associated Glyph elements
    and TextLine information.
    
    Args:
        alto_xml_path: Path to ALTO XML file
        
    Returns:
        List of StringInfo objects containing word content, glyphs, and metadata
    """
    if not os.path.exists(alto_xml_path):
        return []
    
    try:
        tree = ET.parse(alto_xml_path)
        root = tree.getroot()
    except Exception as e:
        logging.warning(f"Could not parse XML file {alto_xml_path}: {type(e).__name__}: {e}", exc_info=True)
        return []
    
    strings: List[StringInfo] = []
    
    # Build parent mapping for ElementTree (since it doesn't have getparent())
    parent_map: Dict[ET.Element, Optional[ET.Element]] = {}
    for parent in root.iter():
        for child in list(parent):
            parent_map[child] = parent
    
    # First pass: collect TextLines with their polygons
    textlines: Dict[str, TextLineInfo] = {}
    for textline_elem in root.iter():
        if _get_local(textline_elem.tag) != 'TextLine':
            continue
        textline_id = textline_elem.attrib.get('ID', '')
        baseline = textline_elem.attrib.get('BASELINE')
        
        textline_poly: Optional[List[Tuple[float, float]]] = None
        for child in list(textline_elem):
            if _get_local(child.tag) == 'Shape':
                for shp_child in list(child):
                    if _get_local(shp_child.tag) == 'Polygon':
                        pts = shp_child.attrib.get('POINTS')
                        if pts:
                            textline_poly = _parse_points(pts)
                            break
        
        textlines[textline_id] = TextLineInfo(line_id=textline_id, polygon=textline_poly, baseline=baseline)
    
    # Second pass: collect Strings and associate with their parent TextLine
    for string_elem in root.iter():
        if _get_local(string_elem.tag) != 'String':
            continue
        content = string_elem.attrib.get('CONTENT', '')
        hpos = _safe_float(string_elem.attrib.get('HPOS'))
        vpos = _safe_float(string_elem.attrib.get('VPOS'))
        width = _safe_float(string_elem.attrib.get('WIDTH'))
        height = _safe_float(string_elem.attrib.get('HEIGHT'))
        wc = _safe_float(string_elem.attrib.get('WC'))
        
        glyphs: List[GlyphInfo] = []
        poly_points: Optional[List[Tuple[float, float]]] = None
        for child in list(string_elem):
            if _get_local(child.tag) != 'Glyph':
                # Capture String-level polygon if present
                if _get_local(child.tag) == 'Shape':
                    for shp_child in list(child):
                        if _get_local(shp_child.tag) == 'Polygon':
                            pts = shp_child.attrib.get('POINTS')
                            if pts:
                                poly_points = _parse_points(pts)
                continue
            char = child.attrib.get('CONTENT', '')
            gc = _safe_float(child.attrib.get('GC'))
            gh = _safe_float(child.attrib.get('HEIGHT'))
            gw = _safe_float(child.attrib.get('WIDTH'))
            gx = _safe_float(child.attrib.get('HPOS'))
            gy = _safe_float(child.attrib.get('VPOS'))
            # Capture Glyph-level polygon if present
            glyph_poly: Optional[List[Tuple[float, float]]] = None
            for gchild in list(child):
                if _get_local(gchild.tag) == 'Shape':
                    for shp_child in list(gchild):
                        if _get_local(shp_child.tag) == 'Polygon':
                            pts = shp_child.attrib.get('POINTS')
                            if pts:
                                glyph_poly = _parse_points(pts)
                                break
            glyphs.append(GlyphInfo(char=char, gc=gc, hpos=gx, vpos=gy, width=gw, height=gh, polygon=glyph_poly))
        
        # Find parent TextLine using parent_map
        textline_info: Optional[TextLineInfo] = None
        current = string_elem
        while current is not None:
            current = parent_map.get(current)
            if current is not None and _get_local(current.tag) == 'TextLine':
                textline_id = current.attrib.get('ID', '')
                if textline_id in textlines:
                    textline_info = textlines[textline_id]
                break
        
        if glyphs:
            strings.append(StringInfo(
                content=content,
                hpos=hpos,
                vpos=vpos,
                width=width,
                height=height,
                wc=wc,
                glyphs=glyphs,
                polygon=poly_points,
                textline=textline_info
            ))
    
    return strings

