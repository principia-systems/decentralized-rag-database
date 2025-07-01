"""
src package for scientific document processing and storage.

This package provides tools for processing, chunking, embedding, and
storing scientific documents in various database systems.
"""

# Import submodules to make them available
import src.core
import src.db
import src.query
import src.rewards
import src.utils
from src.core.chunker import chunk, chunk_from_url
from src.core.converter import convert
from src.core.embedder import embed, embed_from_url

# Core functionality
from src.core.processor import Processor
from src.db.chroma_client import VectorDatabaseManager
from src.db.graph_db import IPFSNeo4jGraph

# Database connectors
from src.db.postgres_db import PostgresDBManager
from src.query.evaluation_agent import EvaluationAgent

# Query functionality
from src.query.query_db import query_collection

# Reward system
from src.rewards.token_rewarder import TokenRewarder
from src.utils.logging_utils import get_logger

# Utility functions
from src.utils.utils import (
    compress,
    download_from_url,
    extract,
    upload_to_lighthouse,
)

__version__ = "0.1.0"
