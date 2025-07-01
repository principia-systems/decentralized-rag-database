"""
Core functionality for document processing.

This module provides classes and functions for converting, chunking, embedding,
and processing documents.
"""

from src.core.chunker import (
    chunk,
    chunk_from_url,
    fixed_length,
    recursive_character,
    markdown_aware,
    semantic_split,
)
from src.core.converter import convert, convert_from_url
from src.core.embedder import embed, embed_from_url, openai
from src.core.processor import Processor
