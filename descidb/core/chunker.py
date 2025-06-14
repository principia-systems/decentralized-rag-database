"""
Text chunking module for DeSciDB.

This module provides functions for splitting text into smaller chunks
for processing and embedding.
"""

import re
from typing import List, Dict

from descidb.types.chunker import ChunkerType, ChunkerFunc
from descidb.utils.logging_utils import get_logger
from descidb.utils.utils import download_from_url

# Get module logger
logger = get_logger(__name__)


def chunk_from_url(
    chunker_type: ChunkerType, input_url: str, user_temp_dir: str = "./tmp"
) -> List[str]:
    """Chunk based on the specified chunking type."""
    download_path = download_from_url(url=input_url, output_folder=user_temp_dir)

    with open(download_path, "r") as file:
        input_text = file.read()

    return chunk(
        chunker_type=chunker_type, input_text=input_text
    )


def chunk(
    chunker_type: ChunkerType, input_text: str
) -> List[str]:
    """Chunk based on the specified chunking type."""

    # Mapping chunking types to functions
    chunking_methods: Dict[str, ChunkerFunc] = {
        "paragraph": paragraph,
        "sentence": sentence,
        "word": word,
        "fixed_length": fixed_length,
    }
    
    return chunking_methods[chunker_type](text=input_text)


def paragraph(text: str) -> List[str]:
    """Chunk the text by paragraphs."""
    # paragraphs = text.split("\n\n")
    # return [p.strip() for p in paragraphs if p.strip()]
    return [text]


def sentence(text: str) -> List[str]:
    """Chunk the text by sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def word(text: str) -> List[str]:
    """Chunk the text by words."""
    words = text.split()
    return [w.strip() for w in words if w.strip()]


def fixed_length(text: str) -> List[str]:
    """Chunk the text into fixed-length chunks."""
    return [text[i : i + 600] for i in range(0, len(text), 600)]
