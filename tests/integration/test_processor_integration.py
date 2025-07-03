"""
Integration tests for the processor module.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.integration
class TestProcessorIntegration:
    """Integration tests for the Processor workflow."""

    @pytest.fixture
    def mock_environment(self):
        """Set up mock environment variables and dependencies."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "mock_openai_key",
                "NEO4J_URI": "mock_neo4j_uri",
                "NEO4J_USERNAME": "mock_neo4j_username",
                "NEO4J_PASSWORD": "mock_neo4j_password",
                "LIGHTHOUSE_TOKEN": "mock_lighthouse_token",
            },
        ):
            yield

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        return {
            "author": {"public_key": "mock_author_key", "email": "test@example.com"},
            "databases": [
                {
                    "converter": "mock_converter",
                    "chunker": "paragraph",
                    "embedder": "mock_embedder",
                }
            ],
            "storage": {"directory": "mock_storage_dir"},
            "processing": {
                "papers_directory": "papers",
                "max_papers": 1,
            },
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "user": "postgres",
                "password": "postgres",
            },
            "api_keys": {"lighthouse_token": "LIGHTHOUSE_TOKEN"},
        }

    @pytest.fixture
    def mock_response(self):
        """Create a mock HTTP response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"Hash": "mock_cid"}
        return mock_resp

    @pytest.fixture
    def mock_processor_class(self):
        """Create a mock processor class."""
        with patch("src.core.processor_main.Processor") as mock_class:
            # Set up the mock processor instance
            mock_processor = MagicMock()
            mock_class.return_value = mock_processor
            yield mock_class

    @patch("src.core.processor_main.VectorDatabaseManager")
    @patch("src.core.processor_main.PostgresDBManager")
    @patch("src.core.processor_main.TokenRewarder")
    @patch("src.core.processor_main.load_config")
    def test_processor_workflow(
        self,
        mock_load_config,
        mock_token_rewarder_class,
        mock_postgres_class,
        mock_vector_db_class,
        mock_processor_class,
        mock_config,
        mock_environment,
        tmp_path,
    ):
        """Test the processor workflow end to end."""
        # Setup
        mock_processor = mock_processor_class.return_value
        mock_load_config.return_value = mock_config

        # These variables are used by the patched code internally
        # so we don't need to use them directly in our test
        _ = mock_vector_db_class.return_value
        _ = mock_postgres_class.return_value
        _ = mock_token_rewarder_class.return_value

        # Create a temp directory for papers
        papers_dir = tmp_path / "papers"
        os.makedirs(papers_dir, exist_ok=True)

        # Create a sample PDF file
        sample_pdf = papers_dir / "sample.pdf"
        with open(sample_pdf, "wb") as f:
            f.write(b"Sample PDF content")

        # Set up the directory patching
        with patch("pathlib.Path.parent", return_value=tmp_path), patch(
            "os.listdir", return_value=["sample.pdf"]
        ), patch("os.path.join", return_value=str(sample_pdf)), patch(
            "os.makedirs"
        ), patch(
            "os.getcwd", return_value="/tmp"
        ), patch(
            "os.chdir"
        ), patch(
            "subprocess.run"
        ), patch(
            "os.urandom", return_value=b"random"
        ), patch(
            "hashlib.sha256", return_value=MagicMock(hexdigest=lambda: "hash")
        ), patch(
            "time.sleep"
        ):
            # Call the processor function
            from src.core.processor_main import test_processor

            test_processor()

        # Assertions
        mock_load_config.assert_called_once()
        mock_processor_class.assert_called_once()
        mock_processor.process.assert_called_once()
