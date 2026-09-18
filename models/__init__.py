"""Models package for the current branch-based JoinsFinder architecture."""
from .multimodal_model import MultiModal
from .tile_branch import TileBranch, TileVisualEncoder, TileTokenEnrichment, TileSetSummarizer, PositionalEncoding2D, apply_page_segment_to_coords
from .glyph_branch import (
    GlyphBranch,
    GlyphHardQualityFilter,
    GlyphVisualEncoder,
    GlyphTokenEnrichment,
    GlyphSetSummarizer,
    PositionalEncoding4D,
    apply_page_segment_to_glyph_coords,
)
from .word_branch import (
    WordBranch,
    WordHardQualityFilter,
    AlephBERTLineEncoder,
    group_words_by_lines,
    determine_line_reading_order,
    BibleTfidfDictionary,
    load_bible_tfidf_dictionary,
    compute_tfidf_weights,
)
from .perceiver_fusion import PerceiverFusion, PerceiverHead
from .vlad_fusion import VLADFusion
from .symmetric_fusion import SymmetricRetrievalFusion, SymmetricRetrievalHead

__all__ = [
    'MultiModal',
    'TileBranch',
    'TileVisualEncoder',
    'TileTokenEnrichment',
    'TileSetSummarizer',
    'PositionalEncoding2D',
    'GlyphBranch',
    'GlyphHardQualityFilter',
    'GlyphVisualEncoder',
    'GlyphTokenEnrichment',
    'GlyphSetSummarizer',
    'PositionalEncoding4D',
    'apply_page_segment_to_coords',
    'apply_page_segment_to_glyph_coords',
    'WordBranch',
    'WordHardQualityFilter',
    'AlephBERTLineEncoder',
    'group_words_by_lines',
    'determine_line_reading_order',
    'BibleTfidfDictionary',
    'load_bible_tfidf_dictionary',
    'compute_tfidf_weights',
    'PerceiverFusion',
    'PerceiverHead',
    'VLADFusion',
    'SymmetricRetrievalFusion',
    'SymmetricRetrievalHead',
]
