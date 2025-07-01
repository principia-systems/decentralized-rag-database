"""
Unit tests for the embedder module.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.core.embedder import embed, embed_from_url, openai


class TestEmbedder:
    """Test cases for embedder functions."""

    @patch("src.core.embedder.OpenAI")
    def test_openai_embedder(self, mock_openai):
        """Test the OpenAI embedder function."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        # Create mock embedding data
        mock_embedding = [0.1, 0.2, 0.3]
        mock_data = MagicMock()
        mock_data.embedding = mock_embedding
        mock_response.data = [mock_data]

        # Call the function
        result = openai("Test text")

        # Verify the expected behavior
        mock_openai.assert_called_once()
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=["Test text"]
        )
        assert result == mock_embedding

    @patch("src.core.embedder.openai")
    def test_embed_with_openai(self, mock_openai_func):
        """Test the embed function with OpenAI embedder."""
        # Mock openai function
        expected_embedding = [0.1, 0.2, 0.3]
        mock_openai_func.return_value = expected_embedding

        # Call the function
        result = embed("openai", "Test text")

        # Verify expected behavior
        mock_openai_func.assert_called_once_with(text="Test text")
        assert result == expected_embedding

    def test_embed_with_invalid_type(self):
        """Test the embed function with an invalid embedder type."""
        with pytest.raises(KeyError):
            embed("invalid_embedder", "Test text")

    @patch("src.core.embedder.embed")
    @patch("src.core.embedder.download_from_url")
    def test_embed_from_url(self, mock_download, mock_embed):
        """Test the embed_from_url function."""
        # Mock download
        mock_file_path = "/tmp/downloaded_file.txt"
        mock_download.return_value = mock_file_path

        # Mock file content
        mock_content = "File content"
        m = MagicMock()
        m.__enter__.return_value.read.return_value = mock_content

        # Mock embed function
        expected_embedding = [0.1, 0.2, 0.3]
        mock_embed.return_value = expected_embedding

        # Call the function with mocked open
        with patch("builtins.open", return_value=m):
            result = embed_from_url("openai", "http://example.com/file.txt")

        # Verify expected behavior
        mock_download.assert_called_once_with(url="http://example.com/file.txt")
        mock_embed.assert_called_once_with(
            embeder_type="openai", input_text=mock_content
        )
        assert result == expected_embedding

    @patch("src.core.embedder.OpenAI")
    def test_openai_embedder_with_environment_variable(self, mock_openai):
        """Test the OpenAI embedder function with API key from environment."""
        # Store the original environment variable
        original_api_key = os.environ.get("OPENAI_API_KEY")

        try:
            # Set a test API key
            test_api_key = "test_api_key"
            os.environ["OPENAI_API_KEY"] = test_api_key

            # Mock OpenAI client and response
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_client.embeddings.create.return_value = mock_response

            # Create mock embedding data
            mock_embedding = [0.1, 0.2, 0.3]
            mock_data = MagicMock()
            mock_data.embedding = mock_embedding
            mock_response.data = [mock_data]

            # Call the function
            openai("Test text")

            # Verify the API key was passed correctly
            mock_openai.assert_called_once_with(api_key=test_api_key)

        finally:
            # Restore the original environment variable
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
            else:
                del os.environ["OPENAI_API_KEY"]
