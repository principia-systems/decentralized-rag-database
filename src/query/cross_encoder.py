"""
Cross-encoder ranking utilities.

This module provides a `CrossEncoderRanker` that accepts a user query and a
list of retrieved items and returns a list of importance scores (relevance).

It supports several Hugging Face cross-encoder reranker models via
`sentence_transformers.CrossEncoder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from sentence_transformers import CrossEncoder

from src.utils.logging_utils import get_logger, get_user_logger
from src.utils.file_lock import file_lock, PROJECT_ROOT


logger = get_logger(__name__)


# Registry of convenient aliases to HF model ids commonly used for reranking
MODEL_PRESETS: Dict[str, str] = {
    # MS MARCO MiniLM models
    "msmarco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "msmarco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    # BAAI bge rerankers
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-large": "BAAI/bge-reranker-large",
    "mxbai-rerank-base-v1": "mixedbread-ai/mxbai-rerank-base-v1",
}


def _preferred_non_cuda_device() -> str:
    """Prefer mps over cpu when CUDA is not available."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _visible_cuda_indices() -> List[int]:
    """Return a list of visible CUDA device indices. Empty if no CUDA."""
    if not torch.cuda.is_available():
        return []
    try:
        count = torch.cuda.device_count()
    except Exception:
        count = 0
    return list(range(count)) if count and count > 0 else []


def _gpu_locks_dir() -> os.PathLike:
    path = PROJECT_ROOT / "temp" / "gpu_locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=16)
def _get_cross_encoder(
    model_id: str,
    device_str: str,
    max_length: Optional[int],
    revision: Optional[str],
    trust_remote_code: bool,
) -> CrossEncoder:
    """Load and cache a CrossEncoder on a specific device."""
    return CrossEncoder(
        model_id,
        device=device_str,
        max_length=max_length,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )


@dataclass
class CrossEncoderConfig:
    """Configuration for `CrossEncoderRanker`."""

    model_name_or_preset: str = "msmarco-MiniLM-L-6-v2"
    device: Optional[str] = None  # "cuda", "cpu", or "mps"; auto if None
    batch_size: int = 32
    max_length: Optional[int] = 512  # Truncation length for sequences
    revision: Optional[str] = None  # HF revision/tag
    trust_remote_code: bool = False


class CrossEncoderRanker:
    """
    Rank retrieved items for a given query using a cross-encoder.

    Usage:
        ranker = CrossEncoderRanker.from_preset("bge-reranker-base")
        scores = ranker.rank(query, items)

    The `rank` method returns a list of floats (importance scores) aligned with
    the input order. Higher is more relevant.
    """

    def __init__(self, config: CrossEncoderConfig, user_email: Optional[str] = None):
        self.config = config
        self.user_email = user_email

        model_id = MODEL_PRESETS.get(
            config.model_name_or_preset, config.model_name_or_preset
        )
        init_logger = (
            get_user_logger(user_email, "cross_encoder") if user_email else logger
        )
        init_logger.info(
            f"Initializing CrossEncoder model='{model_id}' (CUDA round-robin if available) batch_size={config.batch_size}"
        )

        # Defer model instantiation to request time with per-device cache
        self._model_id = model_id

    @classmethod
    def from_preset(
        cls,
        preset: str,
        *,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: Optional[int] = 512,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        user_email: Optional[str] = None,
    ) -> "CrossEncoderRanker":
        """Construct a ranker using a known model preset alias."""
        return cls(
            CrossEncoderConfig(
                model_name_or_preset=preset,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                revision=revision,
                trust_remote_code=trust_remote_code,
            ),
            user_email=user_email,
        )

    def rank(
        self,
        query: str,
        items: Sequence[str],
        *,
        user_email: Optional[str] = None,
    ) -> List[float]:
        """
        Compute importance scores for a list of retrieved strings w.r.t. `query`.

        Returns a list of floats aligned with `items` order; higher is better.
        """

        if not items:
            return []

        # Pair each item text with the query for cross-encoding
        pairs: List[Tuple[str, str]] = [(query, text) for text in items]

        # Use user-level logger if available
        effective_email = user_email or self.user_email
        user_logger = (
            get_user_logger(effective_email, "cross_encoder")
            if effective_email
            else logger
        )
        user_logger.debug(f"Scoring {len(pairs)} pairs with cross-encoder")

        # Request-level GPU binding with lock-based round-robin assignment
        cuda_indices = _visible_cuda_indices()
        total_timeout_sec = int(
            os.getenv("CROSS_ENCODER_GPU_LOCK_TOTAL_TIMEOUT", "600")
        )
        retry_sleep_sec = float(os.getenv("CROSS_ENCODER_GPU_LOCK_RETRY_SLEEP", "0.2"))

        def _predict_on_device(device_str: str) -> List[float]:
            encoder = _get_cross_encoder(
                self._model_id,
                device_str,
                self.config.max_length,
                self.config.revision,
                self.config.trust_remote_code,
            )
            scores_np = encoder.predict(
                pairs,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [float(s) for s in scores_np]

        if cuda_indices:
            # Dynamic round-robin lock over all visible CUDA devices
            locks_dir = _gpu_locks_dir()
            # Always start from GPU index 0
            start_from = 0
            order = list(range(start_from, len(cuda_indices))) + list(
                range(0, start_from)
            )

            start_time = time.time()
            while True:
                for local_idx in order:
                    gpu_idx = cuda_indices[local_idx]
                    lock_path = locks_dir / f"gpu_{gpu_idx}.lock"
                    try:
                        with file_lock(lock_path, timeout=0):
                            gpu_name = torch.cuda.get_device_name(gpu_idx)
                            mem_gb = (
                                torch.cuda.get_device_properties(gpu_idx).total_memory
                                / 1024**3
                            )
                            user_logger.info(
                                f"Acquired GPU lock -> idx={gpu_idx}, name={gpu_name} ({mem_gb:.1f} GB)"
                            )
                            return _predict_on_device(f"cuda:{gpu_idx}")
                    except TimeoutError:
                        continue

                if time.time() - start_time > total_timeout_sec:
                    user_logger.warning(
                        "Timed out acquiring any GPU lock; falling back to non-CUDA device"
                    )
                    break
                time.sleep(retry_sleep_sec)

        # Fallback to non-CUDA device (prefer MPS on Apple Silicon)
        return _predict_on_device(_preferred_non_cuda_device())

        # Note: we return inside device execution paths

    def rank_and_sort(
        self,
        query: str,
        items: Sequence[str],
        *,
        top_k: Optional[int] = None,
        descending: bool = True,
        user_email: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Convenience method to return items paired with scores, sorted by score.
        """
        scores = self.rank(query, items, user_email=user_email)
        paired = list(zip(items, scores))
        paired.sort(key=lambda x: x[1], reverse=descending)
        effective_email = user_email or self.user_email
        user_logger = (
            get_user_logger(effective_email, "cross_encoder")
            if effective_email
            else logger
        )
        if top_k is not None:
            user_logger.debug("Returning top_k=%d scored items", top_k)
        return paired[:top_k] if top_k is not None else paired


__all__ = [
    "CrossEncoderRanker",
    "CrossEncoderConfig",
    "MODEL_PRESETS",
]
