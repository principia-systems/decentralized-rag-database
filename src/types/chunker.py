"""
Type definitions and interfaces for the chunker module.
"""

from typing import Literal

# Type definitions
ChunkerType = Literal["fixed_length", "recursive", "markdown_aware", "semantic_split"]
