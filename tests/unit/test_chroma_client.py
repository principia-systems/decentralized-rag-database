"""
Tests for the ChromaDB client module.

This module contains tests for the VectorDatabaseManager class that manages
ChromaDB vector database collections.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.db.chroma_client import VectorDatabaseManager


class TestVectorDatabaseManager:
    """Test suite for VectorDatabaseManager class."""

    def test_init_with_db_names(self):
        """Test initialization with database names."""
        db_names = ["openai_paragraph_openai"]

        with patch("chromadb.PersistentClient") as mock_client:
            with patch("os.makedirs") as mock_makedirs:
                # Setup mock
                mock_instance = mock_client.return_value
                mock_instance.get_or_create_collection.return_value = MagicMock()

                # Initialize the manager
                manager = VectorDatabaseManager(db_names)

                # Verify
                mock_makedirs.assert_called_once()
                assert manager.db_names == ["openai_paragraph_openai"]
                assert mock_instance.get_or_create_collection.call_count == 1

    def test_init_with_empty_db_names_raises_error(self):
        """Test initialization with empty db_names raises error."""
        db_names = []

        with pytest.raises(ValueError) as excinfo:
            VectorDatabaseManager(db_names)

        assert "db_names must be a non-empty list" in str(excinfo.value)

    def test_init_with_invalid_db_names_type_raises_error(self):
        """Test initialization with invalid db_names type raises error."""
        db_names = "not_a_list"

        with pytest.raises(ValueError) as excinfo:
            VectorDatabaseManager(db_names)

        assert "db_names must be a non-empty list" in str(excinfo.value)

    def test_init_with_custom_db_path(self):
        """Test initialization with custom database path."""
        custom_path = "/tmp/test_chromadb"
        db_names = ["openai_paragraph_openai"]

        with patch("chromadb.PersistentClient") as mock_client:
            with patch("os.makedirs") as mock_makedirs:
                # Setup mock
                mock_instance = mock_client.return_value
                mock_instance.get_or_create_collection.return_value = MagicMock()

                # Initialize manager with custom path
                VectorDatabaseManager(db_names, db_path=custom_path)

                # Verify
                mock_makedirs.assert_called_once_with(Path(custom_path), exist_ok=True)
                mock_client.assert_called_once_with(path=custom_path)
                assert mock_instance.get_or_create_collection.call_count == 1

    def test_initialize_databases(self):
        """Test that initialize_databases creates collections for each database name."""
        db_names = [
            "openai_paragraph_openai",
            "markdown_fixed_length_huggingface",
            "marker_sentence_bge"
        ]

        with patch("chromadb.PersistentClient") as mock_client:
            with patch("os.makedirs"):
                # Setup mock
                mock_instance = mock_client.return_value
                mock_instance.get_or_create_collection.return_value = MagicMock()

                # Initialize the manager
                manager = VectorDatabaseManager(db_names)

                # Verify
                assert len(manager.db_names) == 3
                assert mock_instance.get_or_create_collection.call_count == 3

    def test_insert_document_valid_db(self):
        """Test inserting document into a valid database."""
        db_names = ["openai_paragraph_openai"]
        db_name = "openai_paragraph_openai"
        embedding = [0.1, 0.2, 0.3]
        metadata = {"content_cid": "test_cid", "other": "value"}
        doc_id = "test_doc_id"

        with patch("chromadb.PersistentClient") as mock_client:
            with patch("os.makedirs"):
                # Setup mock
                mock_instance = mock_client.return_value
                mock_collection = MagicMock()
                mock_instance.get_collection.return_value = mock_collection
                mock_instance.get_or_create_collection.return_value = MagicMock()

                # Initialize the manager
                manager = VectorDatabaseManager(db_names)

                # Insert document
                manager.insert_document(db_name, embedding, metadata, doc_id)

                # Verify
                mock_instance.get_collection.assert_called_once_with(name=db_name)
                mock_collection.add.assert_called_once_with(
                    documents=[metadata["content_cid"]],
                    embeddings=embedding,
                    ids=[doc_id],
                    metadatas=[metadata],
                )

    def test_insert_document_invalid_db(self):
        """Test inserting document into an invalid database raises error."""
        db_names = ["openai_paragraph_openai"]
        db_name = "nonexistent_db"
        embedding = [0.1, 0.2, 0.3]
        metadata = {"content_cid": "test_cid"}
        doc_id = "test_doc_id"

        with patch("chromadb.PersistentClient") as mock_client:
            with patch("os.makedirs"):
                # Setup mock
                mock_instance = mock_client.return_value
                mock_instance.get_or_create_collection.return_value = MagicMock()

                # Initialize the manager
                manager = VectorDatabaseManager(db_names)

                # Attempt to insert document into nonexistent DB
                with pytest.raises(ValueError) as excinfo:
                    manager.insert_document(db_name, embedding, metadata, doc_id)

                assert f"Database '{db_name}' does not exist" in str(excinfo.value)

    def test_print_all_metadata(self, capsys):
        """Test that print_all_metadata retrieves and prints metadata from all collections."""
        db_names = ["openai_paragraph_openai"]
        db_name = "openai_paragraph_openai"
        metadata = [{"content_cid": "test_cid", "other": "value"}]

        with patch("chromadb.PersistentClient") as mock_client:
            with patch("os.makedirs"):
                # Setup mock
                mock_instance = mock_client.return_value
                mock_collection = MagicMock()
                mock_instance.get_collection.return_value = mock_collection
                mock_instance.get_or_create_collection.return_value = MagicMock()

                # Mock collection.get() to return test metadata
                mock_collection.get.return_value = {"metadatas": metadata}

                # Initialize the manager and call the method
                db_manager = VectorDatabaseManager(db_names)

                # Call the method
                with patch("builtins.print") as mock_print:
                    db_manager.print_all_metadata()

                    # Verify print calls
                    mock_print.assert_any_call(
                        f"\nMetadata for collection '{db_name}':"
                    )
                    mock_print.assert_any_call(metadata[0])

                # Verify collection.get() was called
                mock_collection.get.assert_called_once()
