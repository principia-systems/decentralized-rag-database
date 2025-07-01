"""
Integration tests for the token reward system.
"""

import os
from unittest.mock import patch

import pytest


@pytest.mark.integration
class TestTokenRewardIntegration:
    """Integration tests for the token reward workflow."""

    @pytest.fixture
    def mock_environment(self):
        """Set up mock environment variables."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "mock_neo4j_uri",
                "NEO4J_USERNAME": "mock_neo4j_username",
                "NEO4J_PASSWORD": "mock_neo4j_password",
                "PRIVATE_KEY": "0x1234567890abcdef",
                "OWNER_ADDRESS": "0xOwnerAddress",
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
            "databases": [
                {
                    "converter": "mock_converter",
                    "chunker": "paragraph",
                    "embedder": "mock_embedder",
                },
                {
                    "converter": "mock_converter",
                    "chunker": "sentence",
                    "embedder": "mock_embedder",
                },
            ],
            "token_rewarder": {
                "network": "test_network",
                "contract_address": "0xContractAddress",
                "contract_abi_path": "contracts/mock_contract.json",
            },
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "user": "postgres",
                "password": "postgres",
            },
        }

    @pytest.fixture
    def mock_author_jobs(self):
        """Mock author job statistics."""
        return {"0xAuthor1": 5, "0xAuthor2": 3, "0xAuthor3": 7}

    def mock_getenv(self, key, default=None):
        """Mock os.getenv to return test values."""
        env_values = {
            "NEO4J_URI": "mock_neo4j_uri",
            "NEO4J_USERNAME": "mock_neo4j_username",
            "NEO4J_PASSWORD": "mock_neo4j_password",
        }
        return env_values.get(key, default)

    @patch("src.rewards.token_reward_main.load_config")
    @patch("src.rewards.token_reward_main.IPFSNeo4jGraph")
    @patch("src.rewards.token_reward_main.TokenRewarder")
    def test_token_reward_workflow(
        self,
        mock_token_rewarder_class,
        mock_graph_class,
        mock_load_config,
        mock_environment,
        mock_config,
        mock_author_jobs,
        tmp_path,
    ):
        """Test the end-to-end token reward workflow."""
        # Setup mocks
        mock_load_config.return_value = mock_config

        # Mock graph and its methods
        mock_graph = mock_graph_class.return_value
        mock_graph.get_authored_by_stats.return_value = mock_author_jobs

        # Mock token rewarder
        mock_token_rewarder = mock_token_rewarder_class.return_value

        # Mock contract ABI path
        mock_contract_path = tmp_path / "contracts" / "mock_contract.json"
        os.makedirs(os.path.dirname(mock_contract_path), exist_ok=True)
        with open(mock_contract_path, "w") as f:
            f.write('{"abi": []}')

        # Mock getenv to return our test values
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.__truediv__", return_value=mock_contract_path
        ), patch("os.path.exists", return_value=True), patch(
            "os.getenv", side_effect=self.mock_getenv
        ):
            # Import and run the main function
            from src.rewards.token_reward_main import run_reward_users

            run_reward_users()

        # Verify the workflow
        mock_graph_class.assert_called_once_with(
            uri="mock_neo4j_uri",
            username="mock_neo4j_username",
            password="mock_neo4j_password",
        )

        mock_graph.get_authored_by_stats.assert_called_once()

        # Verify TokenRewarder initialization
        mock_token_rewarder_class.assert_called_once()

        # Expected values
        expected_db_names = [
            "mock_converter_paragraph_mock_embedder",
            "mock_converter_sentence_mock_embedder",
        ]

        # Verify database creation was called for each DB
        for db_name in expected_db_names:
            mock_token_rewarder._create_database_and_table.assert_any_call(db_name)

        # Verify add_reward_to_user was called for each author and database
        expected_calls = 0
        for author, jobs in mock_author_jobs.items():
            for db_name in expected_db_names:
                mock_token_rewarder.add_reward_to_user.assert_any_call(
                    author, db_name, jobs
                )
                expected_calls += 1

        assert mock_token_rewarder.add_reward_to_user.call_count == expected_calls
