"""
Text embedding module.

This module provides functions for generating embeddings from text chunks
using various embedding models including OpenAI's API and GPU-accelerated models.
"""

import os
import torch
from functools import lru_cache
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.types.embedder import EmbedderType, Embedding, EmbedderFunc
from src.utils.logging_utils import get_logger
from src.utils.utils import download_from_url

# Get module logger
logger = get_logger(__name__)

load_dotenv(override=True)


# Embedding dimensions for each model type
EMBEDDING_DIMENSIONS = {
    "openai": 1536,     # text-embedding-3-small
    "bge": 1024,        # BAAI/bge-large-en-v1.5
    "nomic": 768,       # nomic-ai/nomic-embed-text-v1.5
    "instructor": 768,  # hkunlp/instructor-xl
}


def get_embedding_dimension(embedder_type: EmbedderType) -> int:
    """
    Get the embedding dimension for a specific embedder type.
    
    Args:
        embedder_type: The type of embedder
        
    Returns:
        int: The embedding dimension for the embedder
    """
    return EMBEDDING_DIMENSIONS.get(embedder_type, 768)  # Default to 768 if unknown


def get_device() -> str:
    """
    Determine the best available device for computations.
    Returns 'cuda' if GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        logger.info("Using CPU for embeddings")
    return device


def embed_from_url(embeder_type: EmbedderType, input_url: str, user_temp_dir: str = "./tmp") -> Embedding:
    """Embed based on the specified embedding type."""
    donwload_path = download_from_url(url=input_url, output_folder=user_temp_dir)

    with open(donwload_path, "r") as file:
        input_text = file.read()

    return embed(embeder_type=embeder_type, input_text=input_text)


def embed(embeder_type: EmbedderType, input_text: str) -> Embedding:
    """Embed based on the specified embedding type."""

    embedding_methods: Dict[str, EmbedderFunc] = {
        "openai": openai,
        "bge": bge_large,
        "nomic": nomic_embed_text,
        "instructor": instructor_xl,
    }

    if embeder_type not in embedding_methods:
        raise ValueError(f"Unsupported embedder type: {embeder_type}")

    try:
        return embedding_methods[embeder_type](text=input_text)
    except Exception as e:
        logger.error(f"Error in {embeder_type} embedder: {e}")
        raise


def openai(text: str) -> Embedding:
    """Embed text using the OpenAI embedding API. Returns a list."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(model="text-embedding-3-small", input=[text])
    embedding = response.data[0].embedding
    return embedding


@lru_cache(maxsize=1)
def _load_bge_large() -> SentenceTransformer:
    """Load BGE large model with GPU support."""
    model_name = "BAAI/bge-large-en-v1.5"
    device = get_device()
    logger.info(f"Loading BGE-large model on {device}")
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def _load_instructor_xl() -> SentenceTransformer:
    """Load Instructor XL model with GPU support."""
    model_name = "hkunlp/instructor-xl"
    device = get_device()
    logger.info(f"Loading Instructor-XL model on {device}")
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def _load_nomic_embed_text() -> SentenceTransformer:
    """Load Nomic Embed Text model with GPU support - efficient model with good performance."""
    model_name = "nomic-ai/nomic-embed-text-v1.5"
    device = get_device()
    logger.info(f"Loading Nomic Embed Text model on {device}")
    try:
        return SentenceTransformer(model_name, device=device, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load Nomic model: {e}")
        logger.info("Trying to load without trust_remote_code...")
        try:
            return SentenceTransformer(model_name, device=device)
        except Exception as e2:
            logger.error(f"Failed to load Nomic model without trust_remote_code: {e2}")
            raise


def bge_large(text: str) -> Embedding:
    """Embed text using BGE large model with GPU support."""
    try:
        model = _load_bge_large()
        result = model.encode(text, show_progress_bar=False, convert_to_tensor=False).tolist()
        logger.debug(f"BGE embedding dimension: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"Error in BGE embedder: {e}")
        raise


def instructor_xl(text: str) -> Embedding:
    """Embed text using Instructor XL model with GPU support."""
    try:
        model = _load_instructor_xl()
        # Instructor models expect instruction-based input
        instruction = "Represent the document for retrieval: "
        result = model.encode([[instruction, text]], show_progress_bar=False, convert_to_tensor=False)[0].tolist()
        logger.debug(f"Instructor embedding dimension: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"Error in Instructor embedder: {e}")
        raise


def nomic_embed_text(text: str) -> Embedding:
    """Embed text using Nomic Embed Text model with GPU support."""
    try:
        model = _load_nomic_embed_text()
        result = model.encode(text, show_progress_bar=False, convert_to_tensor=False).tolist()
        logger.debug(f"Nomic embedding dimension: {len(result)}")
        return result
    except Exception as e:
        logger.error(f"Error in Nomic embedder: {e}")
        raise
