import random
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities.augmentations.background_overlays import RandomLibraryBackgroundOverlay


def test_background_overlay_preserves_size_and_mode():
    random.seed(0)
    image = Image.new("RGB", (128, 128), (200, 180, 150))

    augmented = RandomLibraryBackgroundOverlay(p=1.0)(image)

    assert augmented.size == image.size
    assert augmented.mode == "RGB"


def test_background_overlay_adds_library_colored_backing_near_edges():
    random.seed(0)
    image = Image.new("RGB", (128, 128), (200, 180, 150))

    augmented = RandomLibraryBackgroundOverlay(p=1.0, pattern_types=["library_blue_grid"])(image)

    border_points = [(0, 0), (127, 127), (0, 64), (64, 0), (127, 64), (64, 127)]
    assert any(augmented.getpixel(pt) != image.getpixel(pt) for pt in border_points)
    assert augmented.getpixel((64, 64)) == image.getpixel((64, 64))

    changed = sum(1 for p_in, p_out in zip(image.getdata(), augmented.getdata()) if p_in != p_out)
    changed_ratio = changed / (image.size[0] * image.size[1])
    assert changed_ratio < 0.3


def test_surface_aging_changes_fragment_surface_without_support_backing():
    random.seed(0)
    image = Image.new("RGB", (128, 128), (200, 180, 150))

    augmented = RandomLibraryBackgroundOverlay(p=1.0, pattern_types=["parchment_stains"])(image)

    assert augmented.getpixel((64, 64)) != image.getpixel((64, 64))


def test_glyph_scale_paper_stains_keeps_visible_surface_change():
    random.seed(0)
    image = Image.new("RGB", (96, 96), (200, 180, 150))

    augmented = RandomLibraryBackgroundOverlay(p=1.0, pattern_types=["parchment_stains"])(image)

    assert augmented.getpixel((48, 48)) != image.getpixel((48, 48))


def test_padding_edge_artifacts_changes_border_more_than_center():
    random.seed(0)
    image = Image.new("RGB", (96, 96), (240, 240, 240))

    augmented = RandomLibraryBackgroundOverlay(p=1.0, pattern_types=["padding_edge_artifacts"])(image)

    assert augmented.getpixel((0, 0)) != image.getpixel((0, 0))
    assert augmented.getpixel((48, 48)) == image.getpixel((48, 48))


def test_disabling_support_backing_skips_blue_support_variants():
    random.seed(0)
    image = Image.new("RGB", (128, 128), (200, 180, 150))

    augmented = RandomLibraryBackgroundOverlay(
        p=1.0,
        pattern_types=["library_blue_grid", "library_blue_board"],
        allow_support_backing=False,
    )(image)

    assert list(augmented.getdata()) == list(image.getdata())
