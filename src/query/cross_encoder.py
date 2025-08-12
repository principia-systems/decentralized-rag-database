"""
Cross-encoder ranking utilities.

This module provides a `CrossEncoderRanker` that accepts a user query and a
list of retrieved items and returns a list of importance scores (relevance).

It supports several Hugging Face cross-encoder reranker models via
`sentence_transformers.CrossEncoder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from sentence_transformers import CrossEncoder

from src.utils.logging_utils import get_logger, get_user_logger


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


def _auto_detect_device() -> str:
    """Return best available device string among cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    # mps for Apple Silicon (PyTorch 1.12+)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

        model_id = MODEL_PRESETS.get(config.model_name_or_preset, config.model_name_or_preset)
        device = config.device or _auto_detect_device()

        init_logger = (
            get_user_logger(user_email, "cross_encoder") if user_email else logger
        )
        init_logger.info(
            f"Initializing CrossEncoder model='{model_id}' device='{device}' batch_size={config.batch_size}"
        )

        # sentence_transformers.CrossEncoder handles tokenization and model forward
        # It accepts device as a torch.device string (e.g., "cuda", "cpu", "mps")
        self._encoder = CrossEncoder(
            model_id,
            device=device,
            max_length=self.config.max_length,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
        )

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
            get_user_logger(effective_email, "cross_encoder") if effective_email else logger
        )
        user_logger.debug(f"Scoring {len(pairs)} pairs with cross-encoder")

        # sentence_transformers.CrossEncoder.predict returns a numpy array of scores
        scores = self._encoder.predict(
            pairs,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Convert to plain Python floats for portability/serialization
        float_scores = [float(s) for s in scores]
        user_logger.debug(
            "Completed scoring. Example scores: %s",
            float_scores[:5] if len(float_scores) > 5 else float_scores,
        )
        return float_scores

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
            get_user_logger(effective_email, "cross_encoder") if effective_email else logger
        )
        if top_k is not None:
            user_logger.debug("Returning top_k=%d scored items", top_k)
        return paired[:top_k] if top_k is not None else paired


__all__ = [
    "CrossEncoderRanker",
    "CrossEncoderConfig",
    "MODEL_PRESETS",
]


