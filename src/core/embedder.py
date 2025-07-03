"""
Text embedding module.

This module provides functions for generating embeddings from text chunks
using various embedding models including OpenAI's API.
"""

import os
from functools import lru_cache
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.types.embedder import EmbedderType, Embedding, EmbedderFunc
from src.utils.logging_utils import get_logger
from src.utils.utils import download_from_url

# Get module logger
logger = get_logger(__name__)

load_dotenv(override=True)


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
        "nvidia": nvidia,
        "bge": bge,
        "bgelarge": bgelarge,
        "e5large": e5large
    }

    return embedding_methods[embeder_type](text=input_text)


def openai(text: str) -> Embedding:
    """Embed text using the OpenAI embedding API. Returns a list."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(model="text-embedding-3-small", input=[text])
    embedding = response.data[0].embedding
    return embedding


def nvidia(text: str) -> Embedding:
    """Embed text using NVIDIA embeddings. Returns a list."""
    # Implementation not available yet
    return []  # Return empty list for now


@lru_cache(maxsize=1)
def _load_bge() -> SentenceTransformer:
    model_name = "BAAI/bge-small-en"
    return SentenceTransformer(model_name, device="cuda")


def bge(text: str) -> Embedding:
    model = _load_bge()
    return model.encode(text, show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def _load_bge_large() -> SentenceTransformer:
    model_name = "BAAI/bge-large-en-v1.5"
    return SentenceTransformer(model_name, device="cuda")


def bgelarge(text: str) -> Embedding:
    """Embed text using BGE Large model with 1024 dimensions. Returns a list."""
    model = _load_bge_large()
    return model.encode(text, show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def _load_e5_large() -> SentenceTransformer:
    model_name = "intfloat/e5-large-v2"
    return SentenceTransformer(model_name, device="cuda")


def e5large(text: str) -> Embedding:
    """Embed text using E5 Large model with 1024 dimensions. Returns a list."""
    model = _load_e5_large()
    # E5 models require passage prefix for documents
    prefixed_text = f"passage: {text}"
    return model.encode(prefixed_text, show_progress_bar=False).tolist()