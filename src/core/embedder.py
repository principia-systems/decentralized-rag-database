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
        "bge-large": bge_large,
        "nomic-embed-text": nomic_embed_text,
        "instructor-xl": instructor_xl,
    }

    if embeder_type not in embedding_methods:
        raise ValueError(f"Unsupported embedder type: {embeder_type}")

    return embedding_methods[embeder_type](text=input_text)


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
    return SentenceTransformer(model_name, device=device, trust_remote_code=True)


def bge_large(text: str) -> Embedding:
    """Embed text using BGE large model with GPU support."""
    model = _load_bge_large()
    return model.encode(text, show_progress_bar=False, convert_to_tensor=False).tolist()


def instructor_xl(text: str) -> Embedding:
    """Embed text using Instructor XL model with GPU support."""
    model = _load_instructor_xl()
    # Instructor models expect instruction-based input
    instruction = "Represent the document for retrieval: "
    return model.encode([[instruction, text]], show_progress_bar=False, convert_to_tensor=False)[0].tolist()


def nomic_embed_text(text: str) -> Embedding:
    """Embed text using Nomic Embed Text model with GPU support."""
    model = _load_nomic_embed_text()
    return model.encode(text, show_progress_bar=False, convert_to_tensor=False).tolist()
