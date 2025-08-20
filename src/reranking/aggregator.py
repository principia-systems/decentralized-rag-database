"""
Result Aggregator for Database App.

This module provides an aggregator class that processes results from database_app.py
endpoints before returning them to users. It implements three aggregation mechanisms
to improve result quality and relevance.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from src.utils.logging_utils import get_logger, get_user_logger

logger = get_logger(__name__)


class AggregationStrategy(Enum):
    """Available aggregation strategies."""
    FREQUENCY = "frequency"  # Frequency-based aggregation
    SIMILARITY = "similarity"  # Similarity-based aggregation  
    HYBRID = "hybrid"  # Combined frequency + similarity


@dataclass
class AggregationConfig:
    """Configuration for result aggregation."""
    strategy: AggregationStrategy = AggregationStrategy.HYBRID
    top_k: int = 5  # Number of top results to keep
    similarity_weight: float = 0.7  # Weight for similarity in hybrid mode
    frequency_weight: float = 0.3  # Weight for frequency in hybrid mode
    min_similarity_threshold: float = 0.1  # Minimum similarity to consider


class ResultAggregator:
    """
    Aggregates and processes results from database app endpoints.
    
    This class can process results from:
    1. /api/v1/user/evaluate - Evaluation results from multiple collections
    
    It applies one of three aggregation mechanisms:
    - Frequency: Rank by how often content appears across collections
    - Similarity: Rank by similarity/relevance scores
    - Hybrid: Combine frequency and similarity with configurable weights
    """
    
    def __init__(self, config: Optional[AggregationConfig] = None, user_email: Optional[str] = None):
        """
        Initialize the result aggregator.
        
        Args:
            config: Aggregation configuration. If None, uses defaults.
            user_email: User email for logging purposes.
        """
        self.config = config or AggregationConfig()
        self.user_email = user_email
        self.logger = get_user_logger(user_email, "result_aggregator") if user_email else logger
        
    def aggregate_evaluation_results(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results from /api/v1/user/evaluate endpoint.
        
        Expected input structure:
        {
            "query": "user query",
            "user_email": "user@example.com",
            "total_collections": 3,
            "collection_names": ["collection1", "collection2", "collection3"],
            "collection_results": {
                "collection1": {
                    "query": "user query",
                    "results": [
                        {
                            "document": "content text",
                            "metadata": {"source": "doc.pdf", "page": 1},
                            "distance": 0.2
                        }
                    ]
                }
            }
        }
        
        Args:
            evaluation_results: Raw evaluation results from the endpoint
            
        Returns:
            Aggregated and ranked results
        """
        self.logger.info(f"Aggregating evaluation results using {self.config.strategy.value} strategy")
        
        # Extract all document results across collections
        all_items = self._extract_evaluation_items(evaluation_results)
        
        if not all_items:
            self.logger.warning("No items found to aggregate")
            return self._create_empty_result(evaluation_results.get("query", ""))
        
        # Apply aggregation strategy
        aggregated_items = self._apply_aggregation_strategy(all_items)
        
        # Build final result
        result = {
            "query": evaluation_results.get("query", ""),
            "user_email": evaluation_results.get("user_email"),
            "original_total_collections": evaluation_results.get("total_collections", 0),
            "original_collection_names": evaluation_results.get("collection_names", []),
            "aggregation_strategy": self.config.strategy.value,
            "aggregation_config": {
                "top_k": self.config.top_k,
                "similarity_weight": self.config.similarity_weight,
                "frequency_weight": self.config.frequency_weight,
                "min_similarity_threshold": self.config.min_similarity_threshold,
            },
            "aggregated_results": aggregated_items,
            "total_aggregated_items": len(aggregated_items),
        }
        
        self.logger.info(f"Aggregated {len(all_items)} items into {len(aggregated_items)} results")
        return result
    
    
    def _extract_evaluation_items(self, evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract items from evaluation results across all collections."""
        items = []
        collection_results = evaluation_results.get("collection_results", {})
        
        for collection_name, collection_data in collection_results.items():
            if isinstance(collection_data, dict) and "results" in collection_data:
                for result in collection_data["results"]:
                    if not isinstance(result, dict):
                        continue
                    
                    # First try to get content from metadata, fallback to document field
                    metadata = result.get("metadata", {})
                    content = metadata.get("content", "").strip()
                    if not content:
                        # Fallback to document field if no content in metadata
                        content = result.get("document", "").strip()
                    if not content:
                        continue
                    
                    distance = result.get("distance", 1.0)
                    similarity = 1.0 - distance if distance <= 1.0 else 0.0
                    
                    # Skip items below similarity threshold
                    if similarity < self.config.min_similarity_threshold:
                        continue
                    
                    # Extract root_cid from metadata for citation-based frequency
                    metadata = result.get("metadata", {})
                    root_cid = metadata.get("root_cid")
                    
                    item = {
                        "content": content,
                        "similarity": similarity,
                        "distance": distance,
                        "collection": collection_name,
                        "metadata": metadata,
                        "root_cid": root_cid,
                        "frequency": 1,  # Will be updated during aggregation
                    }
                    items.append(item)
        
        return items
    
    def _apply_aggregation_strategy(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply the configured aggregation strategy to the items."""
        if not items:
            return []
        
        # Calculate frequencies
        
        # Apply strategy-specific scoring
        if self.config.strategy == AggregationStrategy.FREQUENCY:
            items = self._calculate_frequencies(items)
            scored_items = self._score_by_frequency(items)
        elif self.config.strategy == AggregationStrategy.SIMILARITY:
            scored_items = self._score_by_similarity(items)
        else:  # HYBRID
            items = self._calculate_frequencies(items)
            scored_items = self._score_by_hybrid(items)
        
        # Sort by final score (descending)
        scored_items.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        
        # Take top k
        top_items = scored_items[:self.config.top_k]
        
        # Add ranking information
        for i, item in enumerate(top_items, 1):
            item["rank"] = i
        
        return top_items
    
    def _calculate_frequencies(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate frequency scores for items based on root_cid citations."""
        # Count frequencies by root_cid (citation-based), falling back to content if no root_cid
        citation_counts = Counter()
        for item in items:
            root_cid = item.get("root_cid")
            citation_counts[str(root_cid)] += 1
        
        max_frequency = max(citation_counts.values()) if citation_counts else 1
        
        for item in items:
            root_cid = item.get("root_cid")
            citation_key = str(root_cid)
            frequency = citation_counts[citation_key]
            item["frequency"] = frequency
            item["normalized_frequency"] = frequency / max_frequency
        
        return items
    
    def _score_by_frequency(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score items based on frequency."""
        for item in items:
            item["final_score"] = item["normalized_frequency"]
            item["scoring_method"] = "frequency"
        return items
    
    def _score_by_similarity(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score items based on similarity."""
        for item in items:
            item["final_score"] = item["similarity"]
            item["scoring_method"] = "similarity"
        return items
    
    def _score_by_hybrid(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score items using hybrid frequency + similarity approach."""
        for item in items:
            similarity_score = item["similarity"] * self.config.similarity_weight
            frequency_score = item["normalized_frequency"] * self.config.frequency_weight
            item["final_score"] = similarity_score + frequency_score
            item["scoring_method"] = "hybrid"
            item["similarity_component"] = similarity_score
            item["frequency_component"] = frequency_score
        return items
    
    def _create_empty_result(self, query: str) -> Dict[str, Any]:
        """Create an empty result structure."""
        return {
            "query": query,
            "user_email": self.user_email,
            "aggregation_strategy": self.config.strategy.value,
            "aggregated_results": [],
            "total_aggregated_items": 0,
            "message": "No results found to aggregate",
        }


__all__ = [
    "ResultAggregator",
    "AggregationStrategy", 
    "AggregationConfig",
]
