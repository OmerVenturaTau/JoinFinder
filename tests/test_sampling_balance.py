"""
Verification test for balanced glyph sampling.
"""

import torch
import pytest
from collections import Counter
from models.glyph_branch import GlyphHardQualityFilter

def test_sampling_balance():
    """
    Verify that sampling is balanced across available characters.
    Scenario:
    - 3 characters: A, B, C
    - A has 100 glyphs
    - B has 2 glyphs
    - C has 2 glyphs
    - max_glyphs = 6
    
    Old behavior (1 per char + top-K overall):
    - Takes A[0], B[0], C[0]
    - Then takes top 3 from remaining: likely A[1], A[2], A[3]
    - Result: A: 4, B: 1, C: 1
    
    New behavior (Round-Robin):
    - Round 1: A[0], B[0], C[0]
    - Round 2: A[1], B[1], C[1]
    - Result: A: 2, B: 2, C: 2 (if available)
    """
    max_glyphs = 6
    glyph_patches = [torch.rand(3, 128, 128) for _ in range(104)]
    
    # Character A: 100 glyphs (high confidence)
    # Character B: 2 glyphs
    # Character C: 2 glyphs
    glyph_metadata = []
    for i in range(100):
        glyph_metadata.append({'char': 'A', 'gc': 0.9, 'id': i})
    for i in range(2):
        glyph_metadata.append({'char': 'B', 'gc': 0.8, 'id': 100+i})
    for i in range(2):
        glyph_metadata.append({'char': 'C', 'gc': 0.8, 'id': 102+i})
        
    sampled_patches, sampled_meta = GlyphHardQualityFilter.sample_diverse_glyphs(
        glyph_patches, glyph_metadata, max_glyphs=max_glyphs
    )
    
    counts = Counter(m['char'] for m in sampled_meta)
    
    print(f"\nSampled counts: {dict(counts)}")
    
    # With 3 characters and max_glyphs=6, each should have exactly 2 samples
    assert counts['A'] == 2
    assert counts['B'] == 2
    assert counts['C'] == 2
    assert len(sampled_meta) == 6

def test_sampling_balance_uneven():
    """
    Scenario:
    - A: 10
    - B: 1
    - C: 10
    - max_glyphs = 5
    
    Expected:
    - Round 1: A, B, C (3)
    - Round 2: A, C (B exhausted) (2)
    - Total: A: 2, B: 1, C: 2
    """
    max_glyphs = 5
    glyph_patches = [torch.rand(3, 128, 128) for _ in range(21)]
    
    glyph_metadata = []
    for i in range(10): glyph_metadata.append({'char': 'A', 'gc': 0.9})
    for i in range(1):  glyph_metadata.append({'char': 'B', 'gc': 0.9})
    for i in range(10): glyph_metadata.append({'char': 'C', 'gc': 0.9})
    
    _, sampled_meta = GlyphHardQualityFilter.sample_diverse_glyphs(
        glyph_patches, glyph_metadata, max_glyphs=max_glyphs
    )
    
    counts = Counter(m['char'] for m in sampled_meta)
    print(f"\nSampled counts (uneven): {dict(counts)}")
    
    assert counts['A'] == 2
    assert counts['B'] == 1
    assert counts['C'] == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
