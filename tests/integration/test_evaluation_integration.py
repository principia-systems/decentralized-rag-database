"""
Integration tests for the evaluation system.
"""

import json
from unittest.mock import ANY, MagicMock, patch

import pytest


@pytest.mark.integration
class TestEvaluationIntegration:
    """Integration tests for the evaluation workflow."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        return {
            "model_name": "mock_model",
            "query": "What is the impact of climate change on biodiversity?",
            "collections": [
                "mock_converter_paragraph_mock_embedder",
                "mock_converter_sentence_mock_embedder",
            ],
            "db_path": "mock_db_path",
            "output_dir": "mock_output_dir",
        }

    @pytest.fixture
    def mock_results(self):
        """Mock query results."""
        return {
            "query": "What is the impact of climate change on biodiversity?",
            "collection_results": {
                "mock_converter_paragraph_mock_embedder": [
                    {
                        "text": "Climate change affects biodiversity.",
                        "metadata": {"source": "doc1"},
                        "score": 0.95,
                    },
                    {
                        "text": "Species loss is accelerating.",
                        "metadata": {"source": "doc2"},
                        "score": 0.85,
                    },
                ],
                "mock_converter_sentence_mock_embedder": [
                    {
                        "text": "Climate change is a major threat to biodiversity.",
                        "metadata": {"source": "doc3"},
                        "score": 0.90,
                    },
                    {
                        "text": "Ecosystems are being disrupted.",
                        "metadata": {"source": "doc4"},
                        "score": 0.80,
                    },
                ],
            },
        }

    @pytest.fixture
    def mock_evaluation(self):
        """Mock evaluation results."""
        return {
            "ranking": [
                {"name": "mock_converter_paragraph_mock_embedder", "score": 9.5},
                {"name": "mock_converter_sentence_mock_embedder", "score": 8.0},
            ],
            "reasoning": "The paragraph chunking database provided more comprehensive answers about climate change impacts.",
        }

    def test_evaluation_workflow(
        self, mock_config, mock_results, mock_evaluation, tmp_path
    ):
        """Test the end-to-end evaluation workflow."""
        # Create a results file with actual content
        results_file = tmp_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(mock_results, f)

        # Set up all patches
        with patch(
            "src.query.evaluation_main.load_config", return_value=mock_config
        ), patch(
            "src.query.evaluation_main.EvaluationAgent"
        ) as mock_agent_class, patch(
            "os.makedirs"
        ) as mock_makedirs, patch(
            "json.dump"
        ) as mock_json_dump, patch(
            "builtins.print"
        ) as mock_print, patch(
            "builtins.open", create=True
        ) as mock_open, patch(
            "json.load", return_value=mock_results
        ):
            # Mock agent and its methods
            mock_agent = mock_agent_class.return_value
            mock_agent.query_collections.return_value = str(results_file)
            mock_agent.evaluate_results.return_value = mock_evaluation

            # Mock file operations
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            # Import and run the main function
            from src.query.evaluation_main import main

            main()

            # Verify the workflow
            mock_agent_class.assert_called_once_with(model_name="mock_model")
            mock_agent.query_collections.assert_called_once_with(
                query="What is the impact of climate change on biodiversity?",
                collection_names=mock_config["collections"],
                db_path=mock_config["db_path"],
            )
            mock_agent.evaluate_results.assert_called_once_with(str(results_file))

            # Verify results were saved
            mock_makedirs.assert_called_once_with("mock_output_dir", exist_ok=True)
            mock_json_dump.assert_called_once_with(mock_evaluation, ANY, indent=2)

            # Verify results were printed
            mock_print.assert_called_once_with(json.dumps(mock_evaluation, indent=2))
