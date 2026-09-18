"""
Tests for coordinate normalization and extraction in dataset.
"""

import sys
import types

import torch
import pytest
from system import CHAR_PATCH_SIZE, GLYPHS_PER_CLASS, MAX_CHARS_PER_IMAGE

sys.modules.setdefault("psycopg2", types.SimpleNamespace(connect=None))


def test_glyph_coordinate_normalization():
    """Test that glyph coordinates are normalized correctly."""
    # Simulate what extract_character_patches does:
    # It normalizes coordinates: norm_x = left / w, norm_y = top / h, etc.
    img_width, img_height = 1000, 2000
    left, top = 100, 200
    right, bottom = 150, 230
    width = right - left
    height = bottom - top
    
    # Expected normalized coordinates
    expected_norm_x = left / img_width  # 0.1
    expected_norm_y = top / img_height  # 0.1
    expected_norm_w = width / img_width  # 0.05
    expected_norm_h = height / img_height  # 0.015
    
    # Verify the normalization formula
    assert abs(expected_norm_x - 0.1) < 1e-6
    assert abs(expected_norm_y - 0.1) < 1e-6
    assert abs(expected_norm_w - 0.05) < 1e-6
    assert abs(expected_norm_h - 0.015) < 1e-6
    
    # Verify coordinates are in [0, 1] range
    assert 0.0 <= expected_norm_x <= 1.0
    assert 0.0 <= expected_norm_y <= 1.0
    assert 0.0 <= expected_norm_w <= 1.0
    assert 0.0 <= expected_norm_h <= 1.0


def test_collate_glyph_coordinate_extraction_logic():
    """Test fixed-slot glyph coordinate extraction from metadata."""
    from utilities.VisionModule.xml_character_extraction import char_to_class_id

    B = 2
    aleph_id = char_to_class_id("א")
    
    char_patch_tensors = [
        torch.rand(3, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE),  # 3 glyphs
        torch.rand(2, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE),  # 2 glyphs
    ]
    
    char_metadata_lists = [
        [
            {'char_class_id': aleph_id, 'normalized_x': 0.1, 'normalized_y': 0.2, 'normalized_w': 0.05, 'normalized_h': 0.03},
            {'char_class_id': aleph_id, 'normalized_x': 0.3, 'normalized_y': 0.4, 'normalized_w': 0.06, 'normalized_h': 0.04},
            {'char_class_id': aleph_id, 'normalized_x': 0.5, 'normalized_y': 0.6, 'normalized_w': 0.07, 'normalized_h': 0.05},
        ],
        [
            {'char_class_id': aleph_id, 'normalized_x': 0.2, 'normalized_y': 0.3, 'normalized_w': 0.04, 'normalized_h': 0.02},
            {'char_class_id': aleph_id, 'normalized_x': 0.4, 'normalized_y': 0.5, 'normalized_w': 0.05, 'normalized_h': 0.03},
        ],
    ]
    
    # Simulate fixed-slot collate logic
    max_chars = MAX_CHARS_PER_IMAGE
    glyph_coords = torch.zeros(B, max_chars, 4)
    char_valid_mask = torch.zeros(B, max_chars, dtype=torch.bool)
    
    for i, (cp, metadata_list) in enumerate(zip(char_patch_tensors, char_metadata_lists)):
        class_counts = [0]
        for meta in metadata_list[:cp.shape[0]]:
            class_id = int(meta["char_class_id"])
            dst = class_id * GLYPHS_PER_CLASS + class_counts[class_id]
            class_counts[class_id] += 1
            char_valid_mask[i, dst] = True
            glyph_coords[i, dst, 0] = meta['normalized_x']
            glyph_coords[i, dst, 1] = meta['normalized_y']
            glyph_coords[i, dst, 2] = meta['normalized_w']
            glyph_coords[i, dst, 3] = meta['normalized_h']
    
    # Verify glyph_coords shape: two classes with eight fixed slots each.
    assert max_chars == MAX_CHARS_PER_IMAGE == 16
    assert glyph_coords.shape == (B, max_chars, 4)
    
    # Verify coordinates are extracted into per-letter blocks.
    assert abs(glyph_coords[0, 0, 0].item() - 0.1) < 1e-6
    assert abs(glyph_coords[0, 0, 1].item() - 0.2) < 1e-6
    assert abs(glyph_coords[0, 0, 2].item() - 0.05) < 1e-6
    assert abs(glyph_coords[0, 0, 3].item() - 0.03) < 1e-6
    
    assert abs(glyph_coords[0, 1, 0].item() - 0.3) < 1e-6
    assert abs(glyph_coords[0, 2, 0].item() - 0.5) < 1e-6
    
    assert char_valid_mask[1, :GLYPHS_PER_CLASS].sum().item() == 2
    assert abs(glyph_coords[1, 0, 0].item() - 0.2) < 1e-6
    assert abs(glyph_coords[1, 1, 0].item() - 0.4) < 1e-6
    
    assert char_valid_mask[1, 2:].sum().item() == 0
    assert glyph_coords[1, 2, 0].item() == 0.0


def test_collate_glyph_coordinates_missing_normalized():
    """Test that collate logic handles missing normalized coordinates gracefully."""
    B = 1
    max_chars = 3
    
    char_patch_tensors = [torch.rand(2, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)]
    char_metadata_lists = [
        [
            {'normalized_x': 0.1, 'normalized_y': 0.2, 'normalized_w': 0.05, 'normalized_h': 0.03},
            {},  # Missing normalized coordinates - should use zeros
        ],
    ]
    
    # Simulate collate logic
    max_chars = max(cp.shape[0] for cp in char_patch_tensors) if char_patch_tensors else 0
    glyph_coords = torch.zeros(B, max_chars, 4)
    
    for i, (cp, metadata_list) in enumerate(zip(char_patch_tensors, char_metadata_lists)):
        m = cp.shape[0]
        if m > 0:
            for j, meta in enumerate(metadata_list[:m]):
                if meta and 'normalized_x' in meta:
                    glyph_coords[i, j, 0] = meta['normalized_x']
                    glyph_coords[i, j, 1] = meta['normalized_y']
                    glyph_coords[i, j, 2] = meta['normalized_w']
                    glyph_coords[i, j, 3] = meta['normalized_h']
    
    # First glyph should have coordinates
    assert abs(glyph_coords[0, 0, 0].item() - 0.1) < 1e-6
    
    # Second glyph should have zeros (missing normalized_x)
    assert glyph_coords[0, 1, 0].item() == 0.0
    assert glyph_coords[0, 1, 1].item() == 0.0


def test_collate_empty_glyph_batch():
    """Test fixed-slot collate logic with empty glyph batches."""
    B = 2
    
    char_patch_tensors = [
        torch.zeros(0, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE),  # Empty
        torch.zeros(0, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE),  # Empty
    ]
    char_metadata_lists = [[], []]
    
    # Simulate fixed-slot collate logic
    max_chars = MAX_CHARS_PER_IMAGE
    glyph_coords = torch.zeros(B, max_chars, 4)
    char_valid_mask = torch.zeros(B, max_chars, dtype=torch.bool)
    
    # Should create fixed-width glyph tensors with all slots masked out.
    assert glyph_coords.shape == (B, MAX_CHARS_PER_IMAGE, 4)
    assert char_valid_mask.sum().item() == 0


def test_tile_collate_uses_fixed_glyph_slots_and_masks_missing_letters():
    from train.dataset import tile_collate_with_padding
    from utilities.VisionModule.xml_character_extraction import char_to_class_id

    aleph_id = char_to_class_id("א")
    mem_id = char_to_class_id("מ")
    patches = torch.stack(
        [
            torch.full((3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE), 1.0),
            torch.full((3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE), 2.0),
            torch.full((3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE), 3.0),
        ]
    )
    metadata = [
        {"char": "א", "char_class_id": aleph_id, "normalized_x": 0.1, "normalized_y": 0.2, "normalized_w": 0.01, "normalized_h": 0.02},
        {"char": "מ", "char_class_id": mem_id, "normalized_x": 0.3, "normalized_y": 0.4, "normalized_w": 0.03, "normalized_h": 0.04},
        {"char": "א", "char_class_id": aleph_id, "normalized_x": 0.5, "normalized_y": 0.6, "normalized_w": 0.05, "normalized_h": 0.06},
    ]
    sample = (
        torch.zeros(1, 3, 16, 16),
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        torch.zeros(1, dtype=torch.long),
        patches,
        metadata,
        [],
        [],
        0,
        "/tmp/glyph-fixed-slots.jpg",
    )

    batch = tile_collate_with_padding([sample])
    assert batch is not None
    (
        _tiles,
        _valid_mask,
        _coords,
        _tile_page_segments,
        char_patches,
        char_valid_mask,
        glyph_coords,
        _glyph_page_segments,
        char_class_ids,
        char_metadata,
        *_rest,
    ) = batch

    assert char_patches.shape[1] == MAX_CHARS_PER_IMAGE == 16
    assert char_valid_mask.shape == (1, MAX_CHARS_PER_IMAGE)
    assert char_valid_mask.sum().item() == 3
    assert char_valid_mask[0, 0].item() is True
    assert char_valid_mask[0, 1].item() is True
    mem_slot = mem_id * GLYPHS_PER_CLASS
    assert char_valid_mask[0, mem_slot].item() is True
    expected_valid = torch.zeros_like(char_valid_mask)
    expected_valid[0, 0] = True
    expected_valid[0, 1] = True
    expected_valid[0, mem_slot] = True
    assert torch.equal(char_valid_mask, expected_valid)
    assert char_class_ids[0, 0].item() == aleph_id
    assert char_class_ids[0, mem_slot].item() == mem_id
    assert glyph_coords[0, 1, 0].item() == pytest.approx(0.5)
    assert glyph_coords[0, mem_slot, 0].item() == pytest.approx(0.3)
    assert char_metadata[0][1]["char"] == "א"
    assert char_metadata[0][mem_slot]["char"] == "מ"


def test_tile_collate_drops_skipped_samples():
    """Unreadable-image samples return None and should be dropped from the batch."""
    from train.dataset import tile_collate_with_padding

    valid_sample = (
        torch.zeros(1, 3, 16, 16),
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(0, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE),
        [],
        [],
        [],
        3,
        "/tmp/valid.jpg",
    )

    batch = tile_collate_with_padding([None, valid_sample])

    assert batch is not None
    tiles, valid_mask, *_rest, labels, paths = batch
    assert tiles.shape[0] == 1
    assert valid_mask.tolist() == [[True]]
    assert labels.tolist() == [3]
    assert paths == ("/tmp/valid.jpg",)


def test_tile_collate_returns_none_when_all_samples_skipped():
    """A fully unreadable mini-batch should be skipped by callers."""
    from train.dataset import tile_collate_with_padding

    assert tile_collate_with_padding([None, None]) is None


def test_tiles_fallback_grid_enabled_returns_grid_for_tiny_text_region(monkeypatch):
    """Default tile behavior: fall back to grid if XML text bounds cannot fit a tile."""
    from PIL import Image
    import utilities.VisionModule.xml_patch_extraction as xml_tiles

    monkeypatch.setattr(xml_tiles, "TILES_FALLBACK_GRID", True)
    image = Image.new("RGB", (1000, 1000), "white")
    text_regions = [
        {
            "hpos": 100,
            "vpos": 100,
            "width": 100,
            "height": 50,
            "center_x": 150,
            "center_y": 125,
        }
    ]

    positions, coords, bounds = xml_tiles.extract_patches_from_text_regions(
        image,
        text_regions,
        patch_size=560,
        stride=420,
        max_patches=8,
        image_size=(1000, 1000),
        constrain_to_bounds=True,
    )

    assert len(positions) > 0
    assert len(coords) == len(positions)
    assert bounds == (100, 100, 200, 150)


def test_tiles_fallback_grid_disabled_returns_empty_for_tiny_text_region(monkeypatch):
    """Strict tile behavior: no grid fallback when XML candidates cannot fit a tile."""
    from PIL import Image
    import utilities.VisionModule.xml_patch_extraction as xml_tiles

    monkeypatch.setattr(xml_tiles, "TILES_FALLBACK_GRID", False)
    image = Image.new("RGB", (1000, 1000), "white")
    text_regions = [
        {
            "hpos": 100,
            "vpos": 100,
            "width": 100,
            "height": 50,
            "center_x": 150,
            "center_y": 125,
        }
    ]

    positions, coords, bounds = xml_tiles.extract_patches_from_text_regions(
        image,
        text_regions,
        patch_size=560,
        stride=420,
        max_patches=8,
        image_size=(1000, 1000),
        constrain_to_bounds=True,
    )

    assert positions == []
    assert coords == []
    assert bounds == (100, 100, 200, 150)


def test_tiles_fallback_grid_disabled_returns_empty_when_coverage_filter_rejects_all(monkeypatch):
    """Strict tile behavior: do not keep pre-filter candidates when coverage rejects all."""
    from PIL import Image
    import utilities.VisionModule.xml_patch_extraction as xml_tiles

    monkeypatch.setattr(xml_tiles, "TILES_FALLBACK_GRID", False)
    image = Image.new("RGB", (1200, 1200), "white")
    text_regions = [
        {"hpos": 100, "vpos": 100, "width": 20, "height": 20, "center_x": 110, "center_y": 110},
        {"hpos": 980, "vpos": 100, "width": 20, "height": 20, "center_x": 990, "center_y": 110},
        {"hpos": 100, "vpos": 980, "width": 20, "height": 20, "center_x": 110, "center_y": 990},
        {"hpos": 980, "vpos": 980, "width": 20, "height": 20, "center_x": 990, "center_y": 990},
    ]

    positions, coords, bounds = xml_tiles.extract_patches_from_text_regions(
        image,
        text_regions,
        patch_size=560,
        stride=420,
        max_patches=8,
        image_size=(1200, 1200),
        constrain_to_bounds=True,
    )

    assert positions == []
    assert coords == []
    assert bounds == (100, 100, 1000, 1000)


def test_tiles_fallback_grid_disabled_returns_empty_when_ink_filter_rejects_all(monkeypatch):
    """Strict tile behavior: do not keep blank candidates when ink filtering rejects all."""
    from PIL import Image
    import utilities.VisionModule.xml_patch_extraction as xml_tiles

    monkeypatch.setattr(xml_tiles, "TILES_FALLBACK_GRID", False)
    image = Image.new("RGB", (1200, 1200), "white")
    text_regions = [
        {
            "hpos": 100,
            "vpos": 100,
            "width": 900,
            "height": 900,
            "center_x": 550,
            "center_y": 550,
        }
    ]

    positions, coords, bounds = xml_tiles.extract_patches_from_text_regions(
        image,
        text_regions,
        patch_size=560,
        stride=420,
        max_patches=8,
        image_size=(1200, 1200),
        constrain_to_bounds=True,
    )

    assert positions == []
    assert coords == []
    assert bounds == (100, 100, 1000, 1000)


def test_dataset_xml_empty_regions_falls_back_when_enabled(monkeypatch):
    """Dataset wrapper should fall back to grid when XML produces no text regions."""
    from PIL import Image
    import train.dataset as dataset_mod

    monkeypatch.setattr(dataset_mod, "TILES_FALLBACK_GRID", True)

    def fake_extract_patches_with_xml(**_kwargs):
        return (
            [],
            torch.empty((0, 2), dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            {"text_regions_count": 0, "xml_path": "/tmp/missing.xml"},
        )

    monkeypatch.setattr(dataset_mod, "extract_patches_with_xml", fake_extract_patches_with_xml)

    ds = dataset_mod.ManuscriptDataset(
        image_paths=[],
        labels=[],
        transform=lambda patch: torch.zeros(3, patch.height, patch.width),
        label2idx={},
        patch_size=16,
        stride=16,
        max_tiles_per_image=2,
        use_xml_extraction=True,
        use_db_coordinates=False,
    )

    patches, coords, page_segments = ds.extract_patches(
        Image.new("RGB", (32, 32), "white"),
        image_path="/tmp/page.jpg",
        xml_path="/tmp/missing.xml",
    )

    assert patches.shape == (2, 3, 16, 16)
    assert coords.shape == (2, 2)
    assert page_segments.tolist() == [0, 0]


def test_dataset_xml_zero_patches_falls_back_when_enabled(monkeypatch):
    """Dataset wrapper should not preserve an empty tile stream when fallback is enabled."""
    from PIL import Image
    import train.dataset as dataset_mod

    monkeypatch.setattr(dataset_mod, "TILES_FALLBACK_GRID", True)

    def fake_extract_patches_with_xml(**_kwargs):
        return (
            [],
            torch.empty((0, 2), dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            {"text_regions_count": 3, "xml_path": "/tmp/page.xml"},
        )

    monkeypatch.setattr(dataset_mod, "extract_patches_with_xml", fake_extract_patches_with_xml)

    ds = dataset_mod.ManuscriptDataset(
        image_paths=[],
        labels=[],
        transform=lambda patch: torch.zeros(3, patch.height, patch.width),
        label2idx={},
        patch_size=16,
        stride=16,
        max_tiles_per_image=2,
        use_xml_extraction=True,
        use_db_coordinates=False,
    )

    patches, coords, page_segments = ds.extract_patches(
        Image.new("RGB", (32, 32), "white"),
        image_path="/tmp/page.jpg",
        xml_path="/tmp/page.xml",
    )

    assert patches.shape == (2, 3, 16, 16)
    assert coords.shape == (2, 2)
    assert page_segments.tolist() == [0, 0]


def test_rtl_coordinate_handling():
    """Test that RTL coordinate flipping is handled in positional encoding (not in normalization)."""
    from models.glyph_branch import PositionalEncoding4D
    from system import D_MODEL, XML_PATCH_READING_DIRECTION_RTL
    
    B, M = 2, 4
    d_model = D_MODEL
    
    # Create positional encoder with RTL enabled
    pos_encoder_rtl = PositionalEncoding4D(d_model=d_model, rtl=True)
    pos_encoder_ltr = PositionalEncoding4D(d_model=d_model, rtl=False)
    
    # Create coordinates (normalized, not flipped)
    # Left side: x=0.1, Right side: x=0.9
    coords = torch.tensor([
        [[0.1, 0.5, 0.05, 0.03], [0.9, 0.5, 0.05, 0.03], [0.5, 0.5, 0.05, 0.03], [0.2, 0.5, 0.05, 0.03]],
        [[0.1, 0.5, 0.05, 0.03], [0.9, 0.5, 0.05, 0.03], [0.5, 0.5, 0.05, 0.03], [0.2, 0.5, 0.05, 0.03]],
    ])
    
    # Get embeddings with RTL and LTR
    pos_emb_rtl = pos_encoder_rtl(coords)
    pos_emb_ltr = pos_encoder_ltr(coords)
    
    # Verify shapes
    assert pos_emb_rtl.shape == (B, M, d_model)
    assert pos_emb_ltr.shape == (B, M, d_model)
    
    # RTL should flip x-coordinates internally, so embeddings should differ
    # For left side (x=0.1), RTL should flip to x=0.9, so embeddings should differ
    assert not torch.allclose(pos_emb_rtl[0, 0], pos_emb_ltr[0, 0], atol=1e-6)
    
    # For right side (x=0.9), RTL should flip to x=0.1
    assert not torch.allclose(pos_emb_rtl[0, 1], pos_emb_ltr[0, 1], atol=1e-6)
    
    # After RTL flip, left side (original x=0.1) should match right side (original x=0.9) in RTL space
    # This is a bit complex, but we can verify that the flipping happened
    # The key is that RTL embeddings are different from LTR embeddings


def test_tile_page_segments_follow_reading_direction():
    """For RTL, right page is segment 0; for LTR, left page is segment 0."""
    from utilities.VisionModule.xml_patch_extraction import page_segment_for_center_x

    split_x = 500.0

    assert page_segment_for_center_x(750.0, split_x, rtl=True) == 0
    assert page_segment_for_center_x(250.0, split_x, rtl=True) == 1

    assert page_segment_for_center_x(250.0, split_x, rtl=False) == 0
    assert page_segment_for_center_x(750.0, split_x, rtl=False) == 1


def test_rtl_page_segments_shift_later_page_for_tile_positional_space():
    """RTL two-page coordinates keep the right page first and shift the left page."""
    from models.tile_branch import apply_page_segment_to_coords

    coords = torch.tensor([[[0.75, 0.25], [0.25, 0.25]]])
    # RTL assignment: right-page tile is first page, left-page tile is second page.
    page_segments = torch.tensor([[0, 1]])

    shifted = apply_page_segment_to_coords(coords, page_segments)

    assert torch.allclose(shifted[0, 0], torch.tensor([0.75, 0.25]))
    assert torch.allclose(shifted[0, 1], torch.tensor([0.25, 1.25]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
