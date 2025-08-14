"""
Query functionality module.

This module provides classes and functions for querying vector databases
and evaluating search results.
"""

from src.query.evaluation_agent import EvaluationAgent
from src.query.query_db import query_collection
from src.query.cross_encoder import (
    CrossEncoderRanker,
    CrossEncoderConfig,
    MODEL_PRESETS,
)

__all__ = [
    "EvaluationAgent",
    "query_collection",
    "CrossEncoderRanker",
    "CrossEncoderConfig",
    "MODEL_PRESETS",
]
