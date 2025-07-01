"""
Unit tests for the token reward system.

This module contains tests for the token_reward_main.py module which provides
functionality to test token rewards by fetching author statistics.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.rewards.token_reward_main import run_reward_users


class TestTokenReward:
    """Test suite for token_reward_main module."""

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Set up mock environment variables for testing."""
        monkeypatch.setenv("NEO4J_URI", "neo4j://testhost:7687")
        monkeypatch.setenv("NEO4J_USERNAME", "test_user")
        monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
        return {
            "NEO4J_URI": "neo4j://testhost:7687",
            "NEO4J_USERNAME": "test_user",
            "NEO4J_PASSWORD": "test_password",
        }

    def test_run_reward_users(self, mock_env_vars):
        """Test the token reward test function."""
        with patch("src.rewards.token_reward_main.IPFSNeo4jGraph") as mock_graph:
            with patch(
                "src.rewards.token_reward_main.TokenRewarder"
            ) as mock_rewarder:
                with patch("src.rewards.token_reward_main.Path") as mock_path:
                    with patch(
                        "src.rewards.token_reward_main.logger"
                    ) as mock_logger:
                        with patch(
                            "src.rewards.token_reward_main.os.getenv"
                        ) as mock_getenv:
                            # Configure environment variable mock
                            def getenv_side_effect(key):
                                return mock_env_vars.get(key)

                            mock_getenv.side_effect = getenv_side_effect

                            # Configure path mock properly
                            mock_path_instance = MagicMock()
                            mock_path.return_value = mock_path_instance
                            mock_path_instance.parent.parent = "/mock/project/root"

                            # Configure graph mock
                            mock_graph_instance = Mock()
                            mock_graph.return_value = mock_graph_instance
                            mock_graph_instance.get_authored_by_stats.return_value = {
                                "author1@example.com": 10,
                                "author2@example.com": 5,
                            }

                            # Configure TokenRewarder mock
                            mock_rewarder_instance = Mock()
                            mock_rewarder.return_value = mock_rewarder_instance

                            # Call the function
                            run_reward_users()

                            # Verify logger was called for start and completion
                            mock_logger.info.assert_any_call(
                                "Starting token reward test"
                            )
                            mock_logger.info.assert_any_call(
                                "Token reward test completed"
                            )

                            # Verify Neo4j graph was initialized with correct parameters
                            mock_graph.assert_called_once_with(
                                uri=mock_env_vars["NEO4J_URI"],
                                username=mock_env_vars["NEO4J_USERNAME"],
                                password=mock_env_vars["NEO4J_PASSWORD"],
                            )

                            # Verify author stats were fetched
                            mock_graph_instance.get_authored_by_stats.assert_called_once()

                            # Verify TokenRewarder was initialized with correct parameters
                            mock_rewarder.assert_called_once()
                            _, kwargs = mock_rewarder.call_args
                            assert kwargs["network"] == "test_base"
                            assert (
                                kwargs["contract_address"]
                                == "0x3bB10ec2404638c6fB9f98948f8e3730316B7BfA"
                            )
                            assert "db_components" in kwargs

                            # Verify components structure
                            components = kwargs["db_components"]
                            assert "converter" in components
                            assert "chunker" in components
                            assert "embedder" in components

                            # Verify create_database_and_table was called
                            assert (
                                mock_rewarder_instance.create_database_and_table.call_count
                                > 0
                            )

                            # Verify add_reward_to_user was called for each author
                            assert (
                                mock_rewarder_instance.add_reward_to_user.call_count
                                == 2
                            )  # Two authors
                            mock_rewarder_instance.add_reward_to_user.assert_any_call(
                                "openai_paragraph_openai", "author1@example.com", 10
                            )
                            mock_rewarder_instance.add_reward_to_user.assert_any_call(
                                "openai_paragraph_openai", "author2@example.com", 5
                            )

    def test_run_reward_users_no_authors(self):
        """Test behavior when no authors are found."""
        with patch("src.rewards.token_reward_main.IPFSNeo4jGraph") as mock_graph:
            with patch(
                "src.rewards.token_reward_main.TokenRewarder"
            ) as mock_rewarder:
                with patch("src.rewards.token_reward_main.Path"):
                    with patch(
                        "src.rewards.token_reward_main.logger"
                    ) as mock_logger:
                        with patch("src.rewards.token_reward_main.os.getenv"):
                            # Configure graph mock to return empty author stats
                            mock_graph_instance = Mock()
                            mock_graph.return_value = mock_graph_instance
                            mock_graph_instance.get_authored_by_stats.return_value = {}

                            # Configure TokenRewarder mock
                            mock_rewarder_instance = Mock()
                            mock_rewarder.return_value = mock_rewarder_instance

                            # Call the function
                            run_reward_users()

                            # Verify author stats were fetched
                            mock_graph_instance.get_authored_by_stats.assert_called_once()

                            # Verify found 0 authors message
                            mock_logger.info.assert_any_call(
                                "Found 0 authors with contributions"
                            )

                            # Verify add_reward_to_user was not called (no authors)
                            assert (
                                mock_rewarder_instance.add_reward_to_user.call_count
                                == 0
                            )

    def test_neo4j_connection_failure(self):
        """Test behavior when Neo4j connection fails."""
        with patch("src.rewards.token_reward_main.IPFSNeo4jGraph") as mock_graph:
            with patch("src.rewards.token_reward_main.TokenRewarder"):
                with patch("src.rewards.token_reward_main.Path"):
                    with patch(
                        "src.rewards.token_reward_main.logger"
                    ) as mock_logger:
                        with patch("src.rewards.token_reward_main.os.getenv"):
                            # Configure graph mock to raise an exception on initialization
                            mock_graph.side_effect = Exception("Connection failed")

                            # Call the function
                            with pytest.raises(Exception) as excinfo:
                                run_reward_users()

                            # Verify exception
                            assert "Connection failed" in str(excinfo.value)

                            # Verify logger was called for start
                            mock_logger.info.assert_called_with(
                                "Starting token reward test"
                            )

    def test_main_execution(self):
        """Test the script execution when run as __main__."""
        with patch("src.rewards.token_reward_main.run_reward_users") as mock_run:
            # Directly test the __name__ == "__main__" condition by mocking the entire module
            with patch.dict(
                "sys.modules",
                {"src.rewards.token_reward_main": Mock(__name__="__main__")},
            ):
                # Import the token_reward_main module to execute the __name__ == "__main__" block
                import importlib

                try:
                    # This might raise an ImportError since we're mocking the module
                    importlib.reload(
                        importlib.import_module("src.rewards.token_reward_main")
                    )
                except ImportError:
                    pass

                # Verify run_reward_users was called
                mock_run.assert_called_once()
