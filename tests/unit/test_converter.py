"""
Unit tests for the converter module.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.converter import (
    chunk_text,
    convert,
    convert_from_url,
    extract_text_from_pdf,
    openai,
)


class TestConverter:
    """Test cases for converter functions."""

    def test_chunk_text_small_text(self):
        """Test chunk_text with text smaller than chunk size."""
        text = "Small text"
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    @patch("PyPDF2.PdfReader")
    def test_extract_text_from_pdf(self, mock_pdf_reader):
        """Test the extract_text_from_pdf function."""
        # Mock a PDF file with two pages
        mock_pages = [MagicMock(), MagicMock()]
        mock_pages[0].extract_text.return_value = "Page 1 content"
        mock_pages[1].extract_text.return_value = "Page 2 content"

        # Set up the mock PDF reader
        mock_pdf_reader.return_value.pages = mock_pages

        # Call the function
        result = extract_text_from_pdf("test.pdf")

        # Verify the expected behavior
        mock_pdf_reader.assert_called_once_with("test.pdf")
        assert result == "Page 1 contentPage 2 content"

    @patch("src.core.converter.OpenAI")
    @patch("src.core.converter.extract_text_from_pdf")
    @patch("os.path.exists")
    def test_openai_converter(self, mock_exists, mock_extract, mock_openai):
        """Test the OpenAI converter function."""
        # Mock file existence check
        mock_exists.return_value = True

        # Mock PDF text extraction
        mock_extract.return_value = "PDF text content"

        # Mock OpenAI client response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Converted markdown content"

        # Call the function
        result = openai("test.pdf")

        # Verify the expected behavior
        mock_exists.assert_called_once_with("test.pdf")
        mock_extract.assert_called_once_with("test.pdf")
        mock_openai.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()
        assert result == "Converted markdown content"

    @patch("os.path.exists")
    def test_openai_converter_file_not_found(self, mock_exists):
        """Test the OpenAI converter with a file that doesn't exist."""
        # Mock file existence check
        mock_exists.return_value = False

        # Call the function
        result = openai("nonexistent.pdf")

        # Verify the expected behavior
        mock_exists.assert_called_once_with("nonexistent.pdf")
        assert result == ""

    @patch("src.core.converter.openai")
    def test_convert_openai(self, mock_openai):
        """Test the convert function with the OpenAI converter."""
        # Mock the OpenAI function
        mock_openai.return_value = "Converted with OpenAI"

        # Call the function
        result = convert("openai", "test.pdf")

        # Verify the expected behavior
        mock_openai.assert_called_once_with("test.pdf")
        assert result == "Converted with OpenAI"

    def test_convert_invalid_type(self):
        """Test the convert function with an invalid converter type."""
        with pytest.raises(KeyError):
            convert("invalid_converter", "test.pdf")

    @patch("src.core.converter.convert")
    @patch("src.core.converter.extract")
    @patch("src.core.converter.download_from_url")
    def test_convert_from_url(self, mock_download, mock_extract, mock_convert):
        """Test the convert_from_url function with a tar file."""
        # Mock download return value
        mock_download.return_value = "/tmp/download/file.tar"

        # Mock convert return value
        mock_convert.return_value = "Converted content"

        # Call the function
        result = convert_from_url("openai", "http://example.com/file.tar")

        # Verify expected behavior
        mock_download.assert_called_once_with(url="http://example.com/file.tar")
        mock_extract.assert_called_once_with(
            tar_file_path="/tmp/download/file.tar", output_path="/tmp/download"
        )
        mock_convert.assert_called_once_with(
            conversion_type="openai", input_path="/tmp/download"
        )
        assert result == "Converted content"
