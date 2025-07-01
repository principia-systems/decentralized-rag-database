"""
Unit tests for the query_db module.

This module contains tests for the query_db.py module which provides
functionality to query ChromaDB collections with natural language queries.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.query.query_db import query_collection


class TestQueryDB:
    """Test suite for query_db module."""

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Set up mock environment variables for testing."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
        return {"OPENAI_API_KEY": "test_api_key"}

    @pytest.fixture
    def mock_openai_response(self):
        """Create a mock OpenAI embedding response."""
        # Create a mock embedding response
        mock_data = Mock()
        mock_data.embedding = [0.1, 0.2, 0.3]

        mock_response = Mock()
        mock_response.data = [mock_data]

        return mock_response

    @pytest.fixture
    def mock_chroma_response(self):
        """Create a mock ChromaDB query response."""
        return {
            "ids": [["id1", "id2"]],
            "documents": [["document1", "document2"]],
            "metadatas": [[{"source": "paper1"}, {"source": "paper2"}]],
            "distances": [[0.1, 0.2]],
        }

    def test_query_collection_success(
        self, mock_env_vars, mock_openai_response, mock_chroma_response
    ):
        """Test successful querying of a collection."""
        with patch("src.query.query_db.OpenAI") as mock_openai_class:
            with patch(
                "src.query.query_db.chromadb.PersistentClient"
            ) as mock_client:
                with patch(
                    "src.query.query_db.embedding_functions.OpenAIEmbeddingFunction"
                ):
                    with patch("src.query.query_db.os.makedirs"):
                        with patch("src.query.query_db.logger") as mock_logger:
                            # Configure OpenAI mock
                            mock_openai_instance = Mock()
                            mock_openai_class.return_value = mock_openai_instance
                            mock_embeddings = Mock()
                            mock_openai_instance.embeddings = mock_embeddings
                            mock_embeddings.create.return_value = mock_openai_response

                            # Configure ChromaDB client mock
                            mock_client_instance = Mock()
                            mock_collection = Mock()
                            mock_client.return_value = mock_client_instance
                            mock_client_instance.get_collection.return_value = (
                                mock_collection
                            )
                            mock_collection.query.return_value = mock_chroma_response

                            # Call the function
                            result = query_collection("test_collection", "test query")

                            # Parse result to check content
                            result_dict = json.loads(result)

                            # Verify OpenAI was called correctly
                            mock_openai_class.assert_called_once()
                            mock_embeddings.create.assert_called_once()

                            # Verify ChromaDB client was initialized correctly
                            mock_client.assert_called_once()
                            mock_client_instance.get_collection.assert_called_once()

                            # Verify collection query was called with correct parameters
                            mock_collection.query.assert_called_once()

                            # Verify the returned result structure
                            assert "query" in result_dict
                            assert "results" in result_dict
                            assert len(result_dict["results"]) == 2
                            assert result_dict["results"][0]["document"] == "document1"
                            assert result_dict["results"][0]["metadata"] == {
                                "source": "paper1"
                            }
                            assert result_dict["results"][0]["distance"] == 0.1

                            # Verify logging
                            mock_logger.info.assert_any_call(
                                "Querying collection 'test_collection' with: 'test query'..."
                            )
                            mock_logger.info.assert_any_call(
                                "Found 2 results for query"
                            )

    def test_query_collection_no_api_key(self):
        """Test behavior when the OpenAI API key is not set."""
        with patch("src.query.query_db.OpenAI"):
            with patch(
                "src.query.query_db.os.getenv", return_value=None
            ) as mock_getenv:
                with patch("src.query.query_db.logger") as mock_logger:
                    # Call the function
                    result = query_collection("test_collection", "test query")

                    # Parse result to check content
                    result_dict = json.loads(result)

                    # Verify error message
                    assert "error" in result_dict
                    assert "API key not configured" in result_dict["error"]

                    # Verify logging
                    mock_logger.error.assert_called_once_with(
                        "OpenAI API key is not set in environment variables"
                    )

                    # Verify getenv was called with the correct parameter
                    mock_getenv.assert_called_with("OPENAI_API_KEY")

    def test_query_collection_no_results(self, mock_env_vars, mock_openai_response):
        """Test behavior when no results are found."""
        empty_response = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        with patch("src.query.query_db.OpenAI") as mock_openai:
            with patch(
                "src.query.query_db.chromadb.PersistentClient"
            ) as mock_client:
                with patch(
                    "src.query.query_db.embedding_functions.OpenAIEmbeddingFunction"
                ):
                    with patch("src.query.query_db.os.makedirs"):
                        with patch("src.query.query_db.logger") as mock_logger:
                            # Configure OpenAI mock
                            mock_openai_instance = Mock()
                            mock_openai.return_value = mock_openai_instance
                            mock_embeddings = Mock()
                            mock_openai_instance.embeddings = mock_embeddings
                            mock_embeddings.create.return_value = mock_openai_response

                            # Configure ChromaDB client mock with empty response
                            mock_client_instance = Mock()
                            mock_collection = Mock()
                            mock_client.return_value = mock_client_instance
                            mock_client_instance.get_collection.return_value = (
                                mock_collection
                            )
                            mock_collection.query.return_value = empty_response

                            # Call the function
                            result = query_collection("test_collection", "test query")

                            # Parse result to check content
                            result_dict = json.loads(result)

                            # Verify empty results
                            assert "query" in result_dict
                            assert "results" in result_dict
                            assert len(result_dict["results"]) == 0

                            # Verify logging
                            mock_logger.warning.assert_called_once_with(
                                "No results found for query: 'test query'..."
                            )

    def test_query_collection_custom_db_path(
        self, mock_env_vars, mock_openai_response, mock_chroma_response
    ):
        """Test using a custom DB path."""
        with patch("src.query.query_db.OpenAI") as mock_openai_class:
            with patch(
                "src.query.query_db.chromadb.PersistentClient"
            ) as mock_client:
                with patch(
                    "src.query.query_db.embedding_functions.OpenAIEmbeddingFunction"
                ):
                    with patch("src.query.query_db.os.makedirs") as mock_makedirs:
                        with patch("src.query.query_db.logger"):
                            # Configure OpenAI mock
                            mock_openai_instance = Mock()
                            mock_openai_class.return_value = mock_openai_instance
                            mock_embeddings = Mock()
                            mock_openai_instance.embeddings = mock_embeddings
                            mock_embeddings.create.return_value = mock_openai_response

                            # Configure ChromaDB client mock
                            mock_client_instance = Mock()
                            mock_collection = Mock()
                            mock_client.return_value = mock_client_instance
                            mock_client_instance.get_collection.return_value = (
                                mock_collection
                            )
                            mock_collection.query.return_value = mock_chroma_response

                            # Use a custom DB path
                            custom_path = "/custom/db/path"

                            # Call the function
                            query_collection(
                                "test_collection", "test query", db_path=custom_path
                            )

                            # Verify makedirs was called with the custom path
                            mock_makedirs.assert_called_once_with(
                                Path(custom_path), exist_ok=True
                            )

                            # Verify ChromaDB client was initialized with the custom path
                            mock_client.assert_called_once_with(path=custom_path)

    def test_query_collection_exception(self, mock_env_vars):
        """Test handling of exceptions during query."""
        with patch("src.query.query_db.OpenAI") as mock_openai_class:
            with patch("src.query.query_db.logger") as mock_logger:
                # Configure OpenAI mock to raise an exception
                mock_openai_class.side_effect = Exception("Test error")

                # Call the function
                result = query_collection("test_collection", "test query")

                # Parse result to check content
                result_dict = json.loads(result)

                # Verify error message
                assert "error" in result_dict
                assert "Test error" in result_dict["error"]

                # Verify logging
                mock_logger.error.assert_called_once_with(
                    "Error querying collection: Test error"
                )
