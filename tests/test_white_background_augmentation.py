import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities.augmentations.manuscript_augmentations import RandomWhiteBackground, WeightedOneOf


def test_white_background_preserves_size_and_mode():
    random.seed(0)
    image = Image.new("RGB", (96, 96), (218, 190, 145))

    augmented = RandomWhiteBackground(p=1.0)(image)

    assert augmented.size == image.size
    assert augmented.mode == "RGB"


def test_white_background_whitens_background_and_preserves_ink():
    random.seed(0)
    image = Image.new("RGB", (96, 96), (218, 190, 145))
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 28, 68, 68), fill=(24, 22, 20))

    augmented = RandomWhiteBackground(p=1.0)(image)

    assert augmented.getpixel((4, 4)) == (255, 255, 255)
    assert max(augmented.getpixel((48, 48))) < 60


def test_white_background_probability_zero_returns_original_object():
    image = Image.new("RGB", (24, 24), (220, 200, 170))

    augmented = RandomWhiteBackground(p=0.0)(image)

    assert augmented is image


def test_weighted_oneof_probabilities_split_conditional_and_effective():
    transform = WeightedOneOf(
        [
            ("a", lambda image: image, 1.0),
            ("b", lambda image: image, 3.0),
        ],
        p=0.8,
    )

    probs = transform.normalized_probabilities()
    effective = transform.effective_probabilities()

    assert probs == {"a": 0.25, "b": 0.75}
    assert effective == {"a": 0.2, "b": 0.6000000000000001}
    assert abs(sum(probs.values()) - 1.0) < 1e-12
    assert abs(sum(effective.values()) - 0.8) < 1e-12


def test_weighted_oneof_apply_probability_zero_returns_original_object():
    image = Image.new("RGB", (24, 24), (220, 200, 170))
    transform = WeightedOneOf([("white", RandomWhiteBackground(p=1.0), 1.0)], p=0.0)

    augmented = transform(image)

    assert augmented is image
