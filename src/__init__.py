"""
Core module for document processing and knowledge graph management.

This module provides a unified interface for processing scientific documents,
managing knowledge graphs, and handling database operations.
"""

from src.core.chunker import chunk
from src.core.converter import convert
from src.core.embedder import embed
from src.core.processor import Processor
from src.db.chroma_client import VectorDatabaseManager
from src.db.db_creator import DatabaseCreator
from src.db.graph_db import IPFSNeo4jGraph
from src.query.query_db import discover_user_collections, query_collection
from src.rewards.token_rewarder import TokenRewarder
from src.utils import (
    IPFSClient,
    compress,
    download_from_url,
    extract,
    get_ipfs_client,
)

__all__ = [
    # Core functions
    "chunk",
    "convert",
    "embed",
    "Processor",
    # Database classes
    "VectorDatabaseManager",
    "DatabaseCreator",
    "IPFSNeo4jGraph",
    # Query functions
    "discover_user_collections",
    "query_collection",
    # Reward system
    "TokenRewarder",
    # Utilities
    "compress",
    "download_from_url",
    "extract",
    "IPFSClient",
    "get_ipfs_client",
]
