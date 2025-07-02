"""
Type definitions and interfaces for the embedder module.
"""

from typing import List, Literal, Callable

# Type definitions - expanded to include GPU-based and heavier models
EmbedderType = Literal[
    "openai", 
    "nvidia", 
    "bge-small", 
    "bge-base", 
    "bge-large",
    "e5-large",
    "gte-large",
    "instructor-xl",
    "stella-1.5b",
    "e5-mistral-7b",
    "nomic-embed-text",
    "jina-embeddings-v2-base-en"
] 
Embedding = List[float] 

# Function type for all embedder implementations
EmbedderFunc = Callable[[str], Embedding]