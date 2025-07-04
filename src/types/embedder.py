"""
Type definitions and interfaces for the embedder module.
"""

from typing import List, Literal, Callable, Optional

# Type definitions - 4 selected embedding models with <2B parameters
EmbedderType = Literal[
    "openai", 
    "bge",
    "nvidia",
    "bgelarge",
    "e5large"
] 
Embedding = List[float] 

# Function type for all embedder implementations
EmbedderFunc = Callable[[str], Embedding]

# Function type for batch embedder implementations
BatchEmbedderFunc = Callable[[List[str], int, Optional[str]], List[Embedding]]