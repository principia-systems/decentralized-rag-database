"""
Text embedding module.

This module provides functions for generating embeddings from text chunks
using various embedding models including OpenAI's API with multi-GPU support.
"""

import os
import time
import math
from functools import lru_cache
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.types.embedder import EmbedderType, Embedding, BatchEmbedderFunc
from src.utils.logging_utils import get_logger, get_user_logger
from src.utils.file_lock import file_lock, PROJECT_ROOT
from src.utils.utils import download_from_url

# Get module logger
logger = get_logger(__name__)

load_dotenv(override=True)


def setup_gpu_config():
    """Setup GPU configuration based on GPU_SPLIT environment variable."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available. Using CPU.")
        return {
            "device": torch.device("cpu"),
            "gpu_devices": [],
            "use_multi_gpu": False,
        }

    # Get total number of GPUs
    total_gpus = torch.cuda.device_count()
    logger.info(f"Found {total_gpus} GPU(s)")

    if total_gpus == 0:
        return {
            "device": torch.device("cpu"),
            "gpu_devices": [],
            "use_multi_gpu": False,
        }

    # Get GPU_SPLIT from environment (default 75%)
    gpu_split = float(os.getenv("GPU_SPLIT", "0.75"))
    logger.info(f"GPU_SPLIT set to {gpu_split:.0%}")

    # Calculate GPUs to use based on split, starting from the end (higher indices)
    if total_gpus == 1:
        gpus_to_use = 1
        gpu_start_idx = 0
    else:
        gpus_to_use = min(total_gpus, max(1, int(total_gpus * gpu_split)))
        gpu_start_idx = total_gpus - gpus_to_use  # Use end GPUs

    # Create list of GPU devices to use
    gpu_devices = []
    for i in range(gpu_start_idx, gpu_start_idx + gpus_to_use):
        device = torch.device(f"cuda:{i}")
        gpu_devices.append(device)
        gpu_name = torch.cuda.get_device_name(i)
        memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"Using GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")

    use_multi_gpu = len(gpu_devices) > 1
    primary_device = gpu_devices[0]

    logger.info(f"Using {len(gpu_devices)} GPU(s) for processing")

    return {
        "device": primary_device,
        "gpu_devices": gpu_devices,
        "use_multi_gpu": use_multi_gpu,
    }


def _compute_gpu_indices_from_split() -> List[int]:
    """Compute list of GPU indices to be used based on GPU_SPLIT, preferring higher indices.

    Returns an empty list if CUDA is unavailable or no GPUs are visible.
    """
    if not torch.cuda.is_available():
        return []

    total_gpus = torch.cuda.device_count()
    if total_gpus <= 0:
        return []

    gpu_split = float(os.getenv("GPU_SPLIT", "0.75"))
    # Determine how many GPUs to expose via locking (round UP for embedder share)
    gpus_to_use = (
        1
        if total_gpus == 1
        else min(total_gpus, max(1, math.ceil(total_gpus * gpu_split)))
    )
    gpu_start_idx = 0 if total_gpus == 1 else total_gpus - gpus_to_use
    return list(range(gpu_start_idx, gpu_start_idx + gpus_to_use))


def _encode_on_device(
    model_loader_func, texts: List[str], batch_size: int, device: torch.device
) -> List[List[float]]:
    """Helper to encode texts on a specific device, returning CPU lists."""
    model = model_loader_func(device)

    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        embeddings = model.encode(
            batch, show_progress_bar=False, convert_to_tensor=True
        )
        if torch.is_tensor(embeddings):
            all_embeddings.extend(embeddings.cpu().tolist())
        else:
            all_embeddings.extend(
                embeddings.tolist()
                if hasattr(embeddings, "tolist")
                else list(embeddings)
            )
    return all_embeddings


def single_gpu_batch_encode(
    model_loader_func, texts: List[str], batch_size: int, user_email: str = None
) -> List[Embedding]:
    """
    Encode texts using a SINGLE GPU selected via a lock file.

    - Creates up to N lock files (N derived from GPU_SPLIT and visible CUDA devices)
    - Acquires exactly one lock (one GPU index) for the duration of this call
    - Falls back to CPU if no GPU available or lock cannot be acquired within timeout
    """
    embed_logger = get_user_logger(user_email, "embedder")

    gpu_indices = _compute_gpu_indices_from_split()
    locks_dir = PROJECT_ROOT / "temp" / "gpu_locks"
    locks_dir.mkdir(parents=True, exist_ok=True)

    total_timeout_sec = int(os.getenv("GPU_LOCK_TOTAL_TIMEOUT", "600"))
    retry_sleep_sec = float(os.getenv("GPU_LOCK_RETRY_SLEEP", "0.2"))

    # If no CUDA GPUs, run on CPU directly
    if not gpu_indices:
        embed_logger.info("No GPUs available; running on CPU")
        return _encode_on_device(
            model_loader_func, texts, batch_size, torch.device("cpu")
        )

    start_time = time.time()
    while True:
        # Try to acquire any available GPU lock non-blockingly
        for gpu_idx in gpu_indices:
            lock_path = locks_dir / f"gpu_{gpu_idx}.lock"
            try:
                with file_lock(lock_path, timeout=0):
                    # Lock acquired for this gpu_idx
                    gpu_name = torch.cuda.get_device_name(gpu_idx)
                    memory_gb = (
                        torch.cuda.get_device_properties(gpu_idx).total_memory
                        / 1024**3
                    )
                    embed_logger.info(
                        f"Acquired GPU lock -> idx={gpu_idx}, name={gpu_name} ({memory_gb:.1f} GB)"
                    )
                    device = torch.device(f"cuda:{gpu_idx}")
                    return _encode_on_device(
                        model_loader_func, texts, batch_size, device
                    )
            except TimeoutError:
                # Lock busy; try next GPU
                continue

        # If none were available, check timeout and retry
        if time.time() - start_time > total_timeout_sec:
            embed_logger.warning(
                "Timed out acquiring any GPU lock; falling back to CPU"
            )
            return _encode_on_device(
                model_loader_func, texts, batch_size, torch.device("cpu")
            )

        time.sleep(retry_sleep_sec)


def process_batch_on_gpu(
    model_loader_func, batch_texts: List[str], device: torch.device, batch_idx: int
) -> Dict:
    """Process a single batch on a specific GPU."""
    try:
        # Load model on the specific device
        model = model_loader_func(device)

        # Process the batch - first get embeddings as tensors on GPU
        embeddings = model.encode(
            batch_texts, show_progress_bar=False, convert_to_tensor=True
        )

        # Move embeddings to CPU and convert to list to avoid device conflicts
        if torch.is_tensor(embeddings):
            embeddings_cpu = embeddings.cpu().tolist()
        else:
            # If it's already numpy/list, ensure it's a list
            embeddings_cpu = (
                embeddings.tolist()
                if hasattr(embeddings, "tolist")
                else list(embeddings)
            )

        return {
            "batch_idx": batch_idx,
            "device": str(device),
            "embeddings": embeddings_cpu,
            "success": True,
            "num_texts": len(batch_texts),
        }

    except Exception as e:
        logger.error(f"Error processing batch {batch_idx} on {device}: {e}")
        return {
            "batch_idx": batch_idx,
            "device": str(device),
            "error": str(e),
            "success": False,
            "num_texts": len(batch_texts),
        }


def multi_gpu_batch_encode(
    model_loader_func, texts: List[str], batch_size: int, user_email: str = None
) -> List[Embedding]:
    """
    Process texts using multiple GPUs when available.

    This function distributes batches across available GPUs in round-robin fashion.
    All embeddings are moved to CPU before collection to avoid device conflicts.
    """
    embed_logger = get_user_logger(user_email, "embedder")

    gpu_config = setup_gpu_config()
    gpu_devices = gpu_config["gpu_devices"]
    use_multi_gpu = gpu_config["use_multi_gpu"]

    # Split texts into batches
    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        batches.append(batch)

    embed_logger.info(
        f"Processing {len(batches)} batches across {len(gpu_devices)} GPU(s)"
    )

    all_embeddings = [None] * len(batches)

    if use_multi_gpu and len(gpu_devices) > 1:
        # Multi-GPU processing
        with ThreadPoolExecutor(max_workers=len(gpu_devices)) as executor:
            future_to_batch = {}

            # Submit batches to GPUs in round-robin fashion
            for batch_idx, batch in enumerate(batches):
                gpu_idx = batch_idx % len(gpu_devices)
                device = gpu_devices[gpu_idx]

                future = executor.submit(
                    process_batch_on_gpu, model_loader_func, batch, device, batch_idx
                )
                future_to_batch[future] = batch_idx

            # Collect results
            successful_batches = 0
            failed_batches = 0

            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                result = future.result()

                if result["success"]:
                    all_embeddings[batch_idx] = result["embeddings"]
                    successful_batches += 1
                else:
                    embed_logger.error(f"Failed batch {batch_idx}: {result['error']}")
                    failed_batches += 1

            embed_logger.info(
                f"Completed: {successful_batches} batches, Failed: {failed_batches} batches"
            )
    else:
        # Single GPU or CPU processing
        device = gpu_devices[0] if gpu_devices else torch.device("cpu")
        model = model_loader_func(device)

        for batch_idx, batch in enumerate(batches):
            # Get embeddings as tensors first, then move to CPU
            embeddings = model.encode(
                batch, show_progress_bar=False, convert_to_tensor=True
            )

            # Move to CPU and convert to list for consistency
            if torch.is_tensor(embeddings):
                embeddings_cpu = embeddings.cpu().tolist()
            else:
                embeddings_cpu = (
                    embeddings.tolist()
                    if hasattr(embeddings, "tolist")
                    else list(embeddings)
                )

            all_embeddings[batch_idx] = embeddings_cpu

    # Flatten and return all embeddings in correct order
    final_embeddings = []
    for batch_embeddings in all_embeddings:
        if batch_embeddings is not None:
            # Embeddings are already converted to lists in process_batch_on_gpu
            if isinstance(batch_embeddings, list):
                final_embeddings.extend(batch_embeddings)
            else:
                # Handle any remaining tensor/numpy cases by moving to CPU first
                if torch.is_tensor(batch_embeddings):
                    batch_embeddings_cpu = batch_embeddings.cpu().tolist()
                else:
                    batch_embeddings_cpu = (
                        batch_embeddings.tolist()
                        if hasattr(batch_embeddings, "tolist")
                        else list(batch_embeddings)
                    )
                final_embeddings.extend(batch_embeddings_cpu)

    embed_logger.info(f"Collected {len(final_embeddings)} embeddings from all batches")
    return final_embeddings


def embed_from_url(
    embeder_type: EmbedderType,
    input_url: str,
    user_temp_dir: str = "./tmp",
    user_email: str = None,
) -> Embedding:
    """Embed based on the specified embedding type."""
    donwload_path = download_from_url(url=input_url, output_folder=user_temp_dir)

    with open(donwload_path, "r") as file:
        input_text = file.read()

    return embed(
        embeder_type=embeder_type, input_text=input_text, user_email=user_email
    )


def embed(
    embeder_type: EmbedderType, input_text: str, user_email: str = None
) -> Embedding:
    """Embed a single text using the specified embedding type."""
    embeddings = embed_batch(
        embeder_type=embeder_type,
        input_texts=[input_text],
        batch_size=1,
        user_email=user_email,
    )
    return embeddings[0]


def embed_batch(
    embeder_type: EmbedderType,
    input_texts: List[str],
    batch_size: int = 32,
    user_email: str = None,
) -> List[Embedding]:
    """Embed multiple texts in batches based on the specified embedding type."""

    # Create user-specific logger
    embed_logger = get_user_logger(user_email, "embedder")

    batch_embedding_methods: Dict[str, BatchEmbedderFunc] = {
        "openai": openai_batch,
        "bge": bge_batch,
        "bgelarge": bgelarge_batch,
    }

    embed_logger.info(
        f"Starting batch embedding with {embeder_type} for {len(input_texts)} texts (batch_size={batch_size})"
    )

    return batch_embedding_methods[embeder_type](
        texts=input_texts, batch_size=batch_size, user_email=user_email
    )


def openai_batch(
    texts: List[str], batch_size: int = 32, user_email: str = None
) -> List[Embedding]:
    """Embed multiple texts using the OpenAI embedding API in batches."""
    embed_logger = get_user_logger(user_email, "embedder")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch = texts[i: i + batch_size]
        embed_logger.info(
            f"Processing OpenAI batch {batch_num}/{num_batches} ({len(batch)} texts)"
        )

        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        batch_embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(batch_embeddings)

    embed_logger.info(f"Completed OpenAI batch embedding for {len(texts)} texts")
    return all_embeddings


@lru_cache(maxsize=8)
def _load_bge(device: torch.device) -> SentenceTransformer:
    """Load BGE model on specified device with caching."""
    model_name = "BAAI/bge-small-en"
    return SentenceTransformer(model_name, device=device)


def bge_batch(
    texts: List[str], batch_size: int = 32, user_email: str = None
) -> List[Embedding]:
    """Embed multiple texts using BGE model in batches with multi-GPU support."""
    embed_logger = get_user_logger(user_email, "embedder")
    embed_logger.info(
        f"Starting BGE batch embedding for {len(texts)} texts (single-GPU lock mode)"
    )
    embeddings = single_gpu_batch_encode(_load_bge, texts, batch_size, user_email)
    embed_logger.info(
        f"Completed BGE batch embedding for {len(texts)} texts (single-GPU lock mode)"
    )
    return embeddings


@lru_cache(maxsize=8)
def _load_bge_large(device: torch.device) -> SentenceTransformer:
    """Load BGE Large model on specified device with caching."""
    model_name = "BAAI/bge-large-en-v1.5"
    return SentenceTransformer(model_name, device=device)


def bgelarge_batch(
    texts: List[str], batch_size: int = 32, user_email: str = None
) -> List[Embedding]:
    """Embed multiple texts using BGE Large model in batches with multi-GPU support."""
    embed_logger = get_user_logger(user_email, "embedder")
    embed_logger.info(
        f"Starting BGE-Large batch embedding for {len(texts)} texts (single-GPU lock mode)"
    )
    embeddings = single_gpu_batch_encode(_load_bge_large, texts, batch_size, user_email)
    embed_logger.info(
        f"Completed BGE-Large batch embedding for {len(texts)} texts (single-GPU lock mode)"
    )
    return embeddings


# Individual embedding functions for backward compatibility
def openai(text: str) -> Embedding:
    """Embed text using the OpenAI embedding API. Returns a list."""
    return openai_batch([text], batch_size=1)[0]


def bge(text: str) -> Embedding:
    """Embed text using BGE model with multi-GPU support. Returns a list."""
    return bge_batch([text], batch_size=1)[0]


def bgelarge(text: str) -> Embedding:
    """Embed text using BGE Large model with 1024 dimensions and multi-GPU support. Returns a list."""
    return bgelarge_batch([text], batch_size=1)[0]
