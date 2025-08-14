"""
Text chunking module.

This module provides functions for splitting text into smaller chunks
for processing and embedding.
"""

import re
from typing import List

from src.types.chunker import ChunkerType
from src.utils.logging_utils import get_logger
from src.utils.utils import download_from_url

# Get module logger
logger = get_logger(__name__)


def chunk_from_url(
    chunker_type: ChunkerType, input_url: str, user_temp_dir: str = "./tmp"
) -> List[str]:
    """Chunk based on the specified chunking type."""
    download_path = download_from_url(url=input_url, output_folder=user_temp_dir)

    with open(download_path, "r") as file:
        input_text = file.read()

    return chunk(chunker_type=chunker_type, input_text=input_text)


def chunk(chunker_type: ChunkerType, input_text: str) -> List[str]:
    """Chunk based on the specified chunking type."""

    # Mapping chunking types to functions
    chunking_methods = {
        "fixed_length": fixed_length,
        "recursive": recursive_character,
        "markdown_aware": markdown_aware,
        "semantic_split": semantic_split,
    }

    return chunking_methods[chunker_type](text=input_text)


def fixed_length(text: str, chunk_size: int = 600) -> List[str]:
    """Chunk the text into fixed-length chunks with smart boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to find a good breaking point (sentence, then paragraph, then word)
        break_point = end

        # Look for sentence boundaries within last 100 chars
        sentence_break = text.rfind(".", start, end)
        if sentence_break > start + chunk_size - 100:
            break_point = sentence_break + 1
        else:
            # Look for paragraph breaks
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size - 200:
                break_point = para_break + 2
            else:
                # Look for word boundaries
                word_break = text.rfind(" ", start, end)
                if word_break > start + chunk_size - 50:
                    break_point = word_break + 1

        chunks.append(text[start:break_point].strip())
        start = break_point

    return [chunk for chunk in chunks if chunk]


def recursive_character(
    text: str, chunk_size: int = 1000, overlap: int = 100
) -> List[str]:
    """Recursively split text using multiple separators in order of preference."""

    # Separators in order of preference
    separators = [
        "\n\n",  # Paragraph breaks
        "\n",  # Line breaks
        ". ",  # Sentence ends
        "! ",  # Exclamation ends
        "? ",  # Question ends
        "; ",  # Semicolon
        ", ",  # Comma
        " ",  # Space
        "",  # Character level (last resort)
    ]

    def _split_text(text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if len(text) <= chunk_size:
            return [text]

        if not separators:
            # Last resort: split by character
            return [
                text[i: i + chunk_size]
                for i in range(0, len(text), chunk_size - overlap)
            ]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level split
            return [
                text[i: i + chunk_size]
                for i in range(0, len(text), chunk_size - overlap)
            ]

        splits = text.split(separator)

        # If we can't split further with this separator, try the next one
        if len(splits) == 1:
            return _split_text(text, remaining_separators)

        # Combine splits that are too small
        chunks = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) + len(separator) <= chunk_size:
                if current_chunk:
                    current_chunk += separator + split
                else:
                    current_chunk = split
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # If this split is still too big, recursively split it
                if len(split) > chunk_size:
                    chunks.extend(_split_text(split, remaining_separators))
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    chunks = _split_text(text, separators)

    # Add overlap between chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = (
                    prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                )
                overlapped_chunks.append(overlap_text + " " + chunk)
        return overlapped_chunks

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def markdown_aware(text: str, chunk_size: int = 1000) -> List[str]:
    """Chunk text while preserving markdown structure."""

    # Split by markdown sections (headers)
    sections = re.split(r"\n(#{1,6}\s+.*?\n)", text)

    chunks = []
    current_chunk = ""
    current_header = ""

    for i, section in enumerate(sections):
        if re.match(r"#{1,6}\s+", section):
            # This is a header
            if current_chunk and len(current_chunk) > chunk_size:
                # Current chunk is too big, split it
                chunks.extend(recursive_character(current_chunk, chunk_size))
                current_chunk = ""
            elif current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            current_header = section.strip()
            current_chunk = current_header
        else:
            # This is content
            if len(current_chunk + section) <= chunk_size:
                current_chunk += section
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If section is too big, split it
                if len(section) > chunk_size:
                    section_chunks = recursive_character(section, chunk_size)
                    # Add header to first chunk
                    if section_chunks:
                        section_chunks[0] = current_header + "\n" + section_chunks[0]
                    chunks.extend(section_chunks)
                    current_chunk = ""
                else:
                    current_chunk = current_header + "\n" + section

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()]


def semantic_split(
    text: str, min_chunk_size: int = 200, max_chunk_size: int = 1000
) -> List[str]:
    """Split text at semantic boundaries (sections, lists, etc.)."""

    # Patterns that indicate semantic boundaries
    semantic_patterns = [
        r"\n#{1,6}\s+.*\n",  # Headers
        r"\n\d+\.\s+",  # Numbered lists
        r"\n[*-]\s+",  # Bullet points
        r"\n\|\s*.*\s*\|",  # Tables
        r"\n```.*?```\n",  # Code blocks
        r"\n>\s+",  # Blockquotes
        r"\n---+\n",  # Horizontal rules
    ]

    # Find all semantic boundaries
    boundaries = [0]
    for pattern in semantic_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            boundaries.append(match.start())

    boundaries.append(len(text))
    boundaries = sorted(set(boundaries))

    chunks = []
    current_chunk = ""

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        segment = text[start:end]

        if len(current_chunk + segment) <= max_chunk_size:
            current_chunk += segment
        else:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = segment
            else:
                # Current chunk is too small, but adding segment makes it too big
                # Split the segment further
                if len(segment) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    # Split large segment using recursive chunking
                    chunks.extend(recursive_character(segment, max_chunk_size))
                else:
                    current_chunk += segment

    if current_chunk and len(current_chunk.strip()) >= min_chunk_size:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()]
