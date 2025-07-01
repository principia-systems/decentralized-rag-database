"""
Type definitions and interfaces for the embedder module.
"""

from typing import List, Literal, Callable

# Type definitions
EmbedderType = Literal["openai", "nvidia", "bge"] 
Embedding = List[float] 

# Function type for all embedder implementations
EmbedderFunc = Callable[[str], Embedding]