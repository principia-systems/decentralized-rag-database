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
        "bge": stella_1_5b,
        "e5-large": e5_large,
        "gte-large": gte_large,
        "instructor-xl": instructor_xl,
        "stella-1.5b": stella_1_5b,
        "e5-mistral-7b": e5_mistral_7b,
        "nomic-embed-text": nomic_embed_text,
        "jina-embeddings-v2-base-en": jina_embeddings_v2_base_en,
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
def _load_bge_small() -> SentenceTransformer:
    """Load BGE small model with GPU support."""
    model_name = "BAAI/bge-small-en-v1.5"
    device = get_device()
    logger.info(f"Loading BGE-small model on {device}")
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def _load_bge_base() -> SentenceTransformer:
    """Load BGE base model with GPU support."""
    model_name = "BAAI/bge-base-en-v1.5"
    device = get_device()
    logger.info(f"Loading BGE-base model on {device}")
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def _load_bge_large() -> SentenceTransformer:
    """Load BGE large model with GPU support."""
    model_name = "BAAI/bge-large-en-v1.5"
    device = get_device()
    logger.info(f"Loading BGE-large model on {device}")
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def _load_e5_large() -> SentenceTransformer:
    """Load E5 large model with GPU support."""
    model_name = "intfloat/e5-large-v2"
    device = get_device()
    logger.info(f"Loading E5-large model on {device}")
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def _load_gte_large() -> SentenceTransformer:
    """Load GTE large model with GPU support."""
    model_name = "thenlper/gte-large"
    device = get_device()
    logger.info(f"Loading GTE-large model on {device}")
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def _load_instructor_xl() -> SentenceTransformer:
    """Load Instructor XL model with GPU support."""
    model_name = "hkunlp/instructor-xl"
    device = get_device()
    logger.info(f"Loading Instructor-XL model on {device}")
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def _load_stella_1_5b() -> SentenceTransformer:
    """Load Stella 1.5B model with GPU support - heavy model for better GPU utilization."""
    model_name = "dunzhang/stella_en_1.5B_v5"
    device = get_device()
    logger.info(f"Loading Stella-1.5B model (~1.5B parameters) on {device}")
    return SentenceTransformer(model_name, device=device)


@lru_cache(maxsize=1)
def _load_e5_mistral_7b() -> SentenceTransformer:
    """Load E5-Mistral-7B model with GPU support - very heavy model for maximum GPU utilization."""
    model_name = "intfloat/e5-mistral-7b-instruct"
    device = get_device()
    logger.info(f"Loading E5-Mistral-7B model (~7B parameters) on {device}")
    return SentenceTransformer(model_name, device=device, trust_remote_code=True)


@lru_cache(maxsize=1)
def _load_nomic_embed_text() -> SentenceTransformer:
    """Load Nomic Embed Text model with GPU support - efficient model with good performance."""
    model_name = "nomic-ai/nomic-embed-text-v1.5"
    device = get_device()
    logger.info(f"Loading Nomic Embed Text model on {device}")
    return SentenceTransformer(model_name, device=device, trust_remote_code=True)


@lru_cache(maxsize=1)
def _load_jina_embeddings_v2_base_en() -> SentenceTransformer:
    """Load Jina Embeddings v2 Base EN model with GPU support."""
    model_name = "jinaai/jina-embeddings-v2-base-en"
    device = get_device()
    logger.info(f"Loading Jina Embeddings v2 Base EN model on {device}")
    return SentenceTransformer(model_name, device=device, trust_remote_code=True)


def bge_small(text: str) -> Embedding:
    """Embed text using BGE small model with GPU support."""
    model = _load_bge_small()
    return model.encode(text, show_progress_bar=False, convert_to_tensor=False).tolist()


def bge_base(text: str) -> Embedding:
    """Embed text using BGE base model with GPU support."""
    model = _load_bge_base()
    return model.encode(text, show_progress_bar=False, convert_to_tensor=False).tolist()


def bge_large(text: str) -> Embedding:
    """Embed text using BGE large model with GPU support."""
    model = _load_bge_large()
    return model.encode(text, show_progress_bar=False, convert_to_tensor=False).tolist()


def e5_large(text: str) -> Embedding:
    """Embed text using E5 large model with GPU support."""
    model = _load_e5_large()
    # E5 models expect the text to be prefixed with "query: " for queries
    prefixed_text = f"query: {text}"
    return model.encode(prefixed_text, show_progress_bar=False, convert_to_tensor=False).tolist()


def gte_large(text: str) -> Embedding:
    """Embed text using GTE large model with GPU support."""
    model = _load_gte_large()
    return model.encode(text, show_progress_bar=False, convert_to_tensor=False).tolist()


def instructor_xl(text: str) -> Embedding:
    """Embed text using Instructor XL model with GPU support."""
    model = _load_instructor_xl()
    # Instructor models expect instruction-based input
    instruction = "Represent the document for retrieval: "
    return model.encode([[instruction, text]], show_progress_bar=False, convert_to_tensor=False)[0].tolist()


def stella_1_5b(text: str) -> Embedding:
    """Embed text using Stella 1.5B model with GPU support - heavy model for better GPU utilization."""
    model = _load_stella_1_5b()
    return model.encode(text, show_progress_bar=False, convert_to_tensor=False).tolist()


def e5_mistral_7b(text: str) -> Embedding:
    """Embed text using E5-Mistral-7B model with GPU support - very heavy model for maximum GPU utilization."""
    model = _load_e5_mistral_7b()
    # E5-Mistral models expect the text to be prefixed with "query: " for queries
    prefixed_text = f"query: {text}"
    return model.encode(prefixed_text, show_progress_bar=False, convert_to_tensor=False).tolist()


def nomic_embed_text(text: str) -> Embedding:
    """Embed text using Nomic Embed Text model with GPU support."""
    model = _load_nomic_embed_text()
    return model.encode(text, show_progress_bar=False, convert_to_tensor=False).tolist()


def jina_embeddings_v2_base_en(text: str) -> Embedding:
    """Embed text using Jina Embeddings v2 Base EN model with GPU support."""
    model = _load_jina_embeddings_v2_base_en()
    return model.encode(text, show_progress_bar=False, convert_to_tensor=False).tolist()
