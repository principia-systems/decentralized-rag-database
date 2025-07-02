"""
Type definitions and interfaces for the embedder module.
"""

from typing import List, Literal, Callable

# Type definitions - 4 selected embedding models with <2B parameters
EmbedderType = Literal[
    "openai", 
    "bge",
    "nomic",
    "instructor"
] 
Embedding = List[float] 

# Function type for all embedder implementations
EmbedderFunc = Callable[[str], Embedding]