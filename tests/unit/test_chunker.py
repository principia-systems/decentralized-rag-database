"""
Unit tests for the chunker module.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.chunker import (
    chunk,
    chunk_from_url,
    fixed_length,
    paragraph,
    sentence,
    word,
)


class TestChunker:
    """Test cases for chunker functions."""

    def test_paragraph_chunker(self):
        """Test the paragraph chunker functionality."""
        text = "This is paragraph 1.\n\nThis is paragraph 2.\n\nThis is paragraph 3."
        chunks = paragraph(text)
        assert len(chunks) == 3
        assert chunks[0] == "This is paragraph 1."
        assert chunks[1] == "This is paragraph 2."
        assert chunks[2] == "This is paragraph 3."

    def test_paragraph_chunker_with_empty_paragraphs(self):
        """Test the paragraph chunker with empty paragraphs."""
        text = "This is paragraph 1.\n\n\n\nThis is paragraph 2."
        chunks = paragraph(text)
        assert len(chunks) == 2
        assert chunks[0] == "This is paragraph 1."
        assert chunks[1] == "This is paragraph 2."

    def test_sentence_chunker(self):
        """Test the sentence chunker functionality."""
        text = "This is sentence 1. This is sentence 2! This is sentence 3?"
        chunks = sentence(text)
        assert len(chunks) == 3
        assert chunks[0] == "This is sentence 1."
        assert chunks[1] == "This is sentence 2!"
        assert chunks[2] == "This is sentence 3?"

    def test_sentence_chunker_with_complex_punctuation(self):
        """Test the sentence chunker with complex punctuation."""
        text = "This is sentence 1... This is sentence 2? Well, this is 3!"
        chunks = sentence(text)
        assert len(chunks) == 3
        assert chunks[0] == "This is sentence 1..."
        assert chunks[1] == "This is sentence 2?"
        assert chunks[2] == "Well, this is 3!"

    def test_word_chunker(self):
        """Test the word chunker functionality."""
        text = "These are some words"
        chunks = word(text)
        assert len(chunks) == 4
        assert chunks[0] == "These"
        assert chunks[1] == "are"
        assert chunks[2] == "some"
        assert chunks[3] == "words"

    def test_fixed_length_chunker(self):
        """Test the fixed length chunker functionality."""
        text = "0123456789"
        chunks = fixed_length(text, 2)
        assert len(chunks) == 5
        assert chunks[0] == "01"
        assert chunks[1] == "23"
        assert chunks[2] == "45"
        assert chunks[3] == "67"
        assert chunks[4] == "89"

    def test_fixed_length_chunker_with_uneven_chunks(self):
        """Test the fixed length chunker with text that doesn't divide evenly."""
        text = "0123456789X"
        chunks = fixed_length(text, 3)
        assert len(chunks) == 4
        assert chunks[0] == "012"
        assert chunks[1] == "345"
        assert chunks[2] == "678"
        assert chunks[3] == "9X"

    def test_chunk_function_paragraph(self):
        """Test the main chunk function with paragraph chunking."""
        text = "Para 1.\n\nPara 2."
        chunks = chunk("paragraph", text)
        assert len(chunks) == 2
        assert chunks[0] == "Para 1."
        assert chunks[1] == "Para 2."

    def test_chunk_function_sentence(self):
        """Test the main chunk function with sentence chunking."""
        text = "Sentence 1. Sentence 2."
        chunks = chunk("sentence", text)
        assert len(chunks) == 2
        assert chunks[0] == "Sentence 1."
        assert chunks[1] == "Sentence 2."

    def test_chunk_function_word(self):
        """Test the main chunk function with word chunking."""
        text = "Word1 Word2"
        chunks = chunk("word", text)
        assert len(chunks) == 2
        assert chunks[0] == "Word1"
        assert chunks[1] == "Word2"

    def test_chunk_function_fixed_length(self):
        """Test the main chunk function with fixed_length chunking."""
        text = "0123456789"
        chunks = chunk("fixed_length", text, 5)
        assert len(chunks) == 2
        assert chunks[0] == "01234"
        assert chunks[1] == "56789"

    def test_chunk_function_invalid_type(self):
        """Test the main chunk function with an invalid chunker type."""
        text = "Sample text"
        with pytest.raises(KeyError):
            chunk("invalid_chunker", text)

    @patch("src.chunker.download_from_url")
    def test_chunk_from_url(self, mock_download):
        """Test the chunk_from_url function."""
        # Set up mock
        mock_file_path = "mock_file.txt"
        mock_download.return_value = mock_file_path

        # Create a mock file content
        mock_content = "Mock paragraph 1.\n\nMock paragraph 2."

        # Mock open function to return our content
        m = MagicMock()
        m.__enter__.return_value.read.return_value = mock_content

        with patch("builtins.open", return_value=m):
            chunks = chunk_from_url("paragraph", "http://example.com/file.txt")

        # Assert download was called
        mock_download.assert_called_once_with(url="http://example.com/file.txt")

        # Assert expected result
        assert len(chunks) == 2
        assert chunks[0] == "Mock paragraph 1."
        assert chunks[1] == "Mock paragraph 2."
