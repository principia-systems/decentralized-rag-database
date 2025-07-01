"""
Integration tests for the database creator.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.integration
class TestDBCreatorIntegration:
    """Integration tests for the database creator workflow."""

    @pytest.fixture(autouse=True)
    def mock_environment(self):
        """Set up mock environment variables."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "mock_neo4j_uri",
                "NEO4J_USERNAME": "mock_neo4j_username",
                "NEO4J_PASSWORD": "mock_neo4j_password",
            },
        ):
            yield

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        return {
            "neo4j": {
                "uri": "${NEO4J_URI}",
                "username": "${NEO4J_USERNAME}",
                "password": "${NEO4J_PASSWORD}",
            },
            "components": {
                "converter": ["mock_converter"],
                "chunker": ["paragraph"],
                "embedder": ["mock_embedder"],
            },
            "vector_db": {"path": "mock_vector_db_path"},
            "relationships": ["HAS_PDF", "HAS_CHUNKS", "HAS_EMBEDDINGS"],
            "cids_file_paths": ["mock_cids.txt"],
        }

    def mock_getenv(self, key, default=None):
        """Mock os.getenv to return test values."""
        env_values = {
            "NEO4J_URI": "mock_neo4j_uri",
            "NEO4J_USERNAME": "mock_neo4j_username",
            "NEO4J_PASSWORD": "mock_neo4j_password",
        }
        return env_values.get(key, default)

    @patch("src.db.db_creator_main.load_config")
    @patch("src.db.db_creator_main.IPFSNeo4jGraph")
    @patch("src.db.db_creator_main.VectorDatabaseManager")
    @patch("src.db.db_creator_main.DatabaseCreator")
    def test_db_creator_workflow(
        self,
        mock_creator,
        mock_vector_db,
        mock_graph,
        mock_load_config,
        mock_config,
        tmp_path,
    ):
        """Test the end-to-end database creator workflow."""
        # Setup mocks
        mock_load_config.return_value = mock_config
        mock_graph_instance = mock_graph.return_value
        mock_vector_db_instance = mock_vector_db.return_value
        mock_creator_instance = mock_creator.return_value

        # Create a temporary CIDs file
        cids_file = tmp_path / "mock_cids.txt"
        with open(cids_file, "w") as f:
            f.write("mock_cid1\nmock_cid2")

        # Mock getenv to ensure environment variables are correctly used
        with patch("os.getenv", side_effect=self.mock_getenv):
            # Properly patch open to handle different paths
            with patch("builtins.open") as mock_open:
                # Handle file operations without recursion
                def mock_open_side_effect(file_path, *args, **kwargs):
                    if "mock_cids.txt" in str(file_path):
                        mock_file = MagicMock()
                        mock_file.__enter__.return_value = mock_file
                        mock_file.__iter__.return_value = ["mock_cid1", "mock_cid2"]
                        return mock_file

                    # Default for other files
                    mock_file = MagicMock()
                    mock_file.__enter__.return_value = mock_file
                    mock_file.read.return_value = "mock content"
                    return mock_file

                mock_open.side_effect = mock_open_side_effect

                # Mock path exists check
                with patch("pathlib.Path.exists", return_value=True), patch(
                    "os.path.exists", return_value=True
                ):
                    # Import and run the main function
                    from src.db.db_creator_main import main

                    main()

        # Verify the workflow
        mock_graph.assert_called_once_with(
            uri="mock_neo4j_uri",
            username="mock_neo4j_username",
            password="mock_neo4j_password",
        )

        mock_vector_db.assert_called_once()
        mock_creator.assert_called_once_with(
            mock_graph_instance, mock_vector_db_instance
        )

        # Check if process_paths was called for each CID
        assert mock_creator_instance.process_paths.call_count == 2
        mock_creator_instance.process_paths.assert_any_call(
            "mock_cid1",
            ["HAS_PDF", "HAS_CHUNKS", "HAS_EMBEDDINGS"],
            "mock_converter_paragraph_mock_embedder",
        )
