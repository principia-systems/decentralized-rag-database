"""
Text embedding module.

This module provides functions for generating embeddings from text chunks
using various embedding models including OpenAI's API.
"""

import os
from functools import lru_cache
from typing import List, Dict

import torch
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.types.embedder import EmbedderType, Embedding, BatchEmbedderFunc
from src.utils.logging_utils import get_logger, get_user_logger
from src.utils.utils import download_from_url

# Get module logger
logger = get_logger(__name__)

load_dotenv(override=True)


def embed_from_url(embeder_type: EmbedderType, input_url: str, user_temp_dir: str = "./tmp", user_email: str = None) -> Embedding:
    """Embed based on the specified embedding type."""
    donwload_path = download_from_url(url=input_url, output_folder=user_temp_dir)

    with open(donwload_path, "r") as file:
        input_text = file.read()

    return embed(embeder_type=embeder_type, input_text=input_text, user_email=user_email)


def embed(embeder_type: EmbedderType, input_text: str, user_email: str = None) -> Embedding:
    """Embed a single text using the specified embedding type."""
    embeddings = embed_batch(embeder_type=embeder_type, input_texts=[input_text], batch_size=1, user_email=user_email)
    return embeddings[0]


def embed_batch(embeder_type: EmbedderType, input_texts: List[str], batch_size: int = 32, user_email: str = None) -> List[Embedding]:
    """Embed multiple texts in batches based on the specified embedding type."""
    
    # Create user-specific logger
    embed_logger = get_user_logger(user_email, "embedder") if user_email else get_logger(__name__ + ".embed_batch")
    
    batch_embedding_methods: Dict[str, BatchEmbedderFunc] = {
        "openai": openai_batch,
        "nvidia": nvidia_batch,
        "bge": bge_batch,
        "bgelarge": bgelarge_batch,
        "e5large": e5large_batch
    }
    
    embed_logger.info(f"Starting batch embedding with {embeder_type} for {len(input_texts)} texts (batch_size={batch_size})")
    
    return batch_embedding_methods[embeder_type](texts=input_texts, batch_size=batch_size, user_email=user_email)


def openai_batch(texts: List[str], batch_size: int = 32, user_email: str = None) -> List[Embedding]:
    """Embed multiple texts using the OpenAI embedding API in batches."""
    embed_logger = get_user_logger(user_email, "embedder") if user_email else get_logger(__name__ + ".openai_batch")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch = texts[i:i + batch_size]
        embed_logger.info(f"Processing OpenAI batch {batch_num}/{num_batches} ({len(batch)} texts)")
        
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        batch_embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(batch_embeddings)
    
    embed_logger.info(f"Completed OpenAI batch embedding for {len(texts)} texts")
    return all_embeddings


def nvidia_batch(texts: List[str], batch_size: int = 32, user_email: str = None) -> List[Embedding]:
    """Embed multiple texts using NVIDIA embeddings in batches."""
    embed_logger = get_user_logger(user_email, "embedder") if user_email else get_logger(__name__ + ".nvidia_batch")
    
    embed_logger.warning("NVIDIA embeddings not implemented yet")
    # Implementation not available yet
    return [[] for _ in texts]  # Return empty list for each text


@lru_cache(maxsize=1)
def _load_bge() -> SentenceTransformer:
    model_name = "BAAI/bge-small-en"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def bge_batch(texts: List[str], batch_size: int = 32, user_email: str = None) -> List[Embedding]:
    """Embed multiple texts using BGE model in batches."""
    embed_logger = get_user_logger(user_email, "embedder") if user_email else get_logger(__name__ + ".bge_batch")
    
    model = _load_bge()
    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch = texts[i:i + batch_size]
        embed_logger.info(f"Processing BGE batch {batch_num}/{num_batches} ({len(batch)} texts)")
        
        batch_embeddings = model.encode(batch, show_progress_bar=False, convert_to_tensor=False)
        all_embeddings.extend(batch_embeddings.tolist())
    
    embed_logger.info(f"Completed BGE batch embedding for {len(texts)} texts")
    return all_embeddings


@lru_cache(maxsize=1)
def _load_bge_large() -> SentenceTransformer:
    model_name = "BAAI/bge-large-en-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def bgelarge_batch(texts: List[str], batch_size: int = 32, user_email: str = None) -> List[Embedding]:
    """Embed multiple texts using BGE Large model in batches."""
    embed_logger = get_user_logger(user_email, "embedder") if user_email else get_logger(__name__ + ".bgelarge_batch")
    
    model = _load_bge_large()
    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch = texts[i:i + batch_size]
        embed_logger.info(f"Processing BGE-Large batch {batch_num}/{num_batches} ({len(batch)} texts)")
        
        batch_embeddings = model.encode(batch, show_progress_bar=False, convert_to_tensor=False)
        all_embeddings.extend(batch_embeddings.tolist())
    
    embed_logger.info(f"Completed BGE-Large batch embedding for {len(texts)} texts")
    return all_embeddings


@lru_cache(maxsize=1)
def _load_e5_large() -> SentenceTransformer:
    model_name = "intfloat/e5-large-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def e5large_batch(texts: List[str], batch_size: int = 32, user_email: str = None) -> List[Embedding]:
    """Embed multiple texts using E5 Large model in batches."""
    embed_logger = get_user_logger(user_email, "embedder") if user_email else get_logger(__name__ + ".e5large_batch")
    
    model = _load_e5_large()
    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch = texts[i:i + batch_size]
        embed_logger.info(f"Processing E5-Large batch {batch_num}/{num_batches} ({len(batch)} texts)")
        
        # E5 models require passage prefix for documents
        prefixed_batch = [f"passage: {text}" for text in batch]
        batch_embeddings = model.encode(prefixed_batch, show_progress_bar=False, convert_to_tensor=False)
        all_embeddings.extend(batch_embeddings.tolist())
    
    embed_logger.info(f"Completed E5-Large batch embedding for {len(texts)} texts")
    return all_embeddings


# Individual embedding functions for backward compatibility
def openai(text: str) -> Embedding:
    """Embed text using the OpenAI embedding API. Returns a list."""
    return openai_batch([text], batch_size=1)[0]


def bge(text: str) -> Embedding:
    """Embed text using BGE model. Returns a list."""
    return bge_batch([text], batch_size=1)[0]


def nvidia(text: str) -> Embedding:
    """Embed text using NVIDIA embeddings. Returns a list."""
    return nvidia_batch([text], batch_size=1)[0]


def bgelarge(text: str) -> Embedding:
    """Embed text using BGE Large model with 1024 dimensions. Returns a list."""
    return bgelarge_batch([text], batch_size=1)[0]


def e5large(text: str) -> Embedding:
    """Embed text using E5 Large model with 1024 dimensions. Returns a list."""
    return e5large_batch([text], batch_size=1)[0]