"""
Type definitions and interfaces for the converter module.
"""

from typing import Literal, Callable

# Type definitions
ConverterType = Literal["marker", "openai", "markitdown"]

# Function type for all converter implementations
ConverterFunc = Callable[[str], str]  # Input path -> Output text