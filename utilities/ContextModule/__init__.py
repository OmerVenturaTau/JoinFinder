"""
Context (text) module for JoinsFinder.

This package groups utilities for:
- extracting textual content from ALTO XML files
- encoding Hebrew text with transformer models (e.g., AlephBERT)
"""

from .alto_text_extractor import extract_text_lines, extract_text_blocks
from .text_encoder import HebrewTextEncoder, TextEncoderConfig

__all__ = [
    "extract_text_lines",
    "extract_text_blocks",
    "HebrewTextEncoder",
    "TextEncoderConfig",
]

