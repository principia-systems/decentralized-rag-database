"""
Main script for running the evaluation agent.

This script loads configuration from a YAML file and executes
the evaluation agent with the specified parameters.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from src.query.evaluation_agent import EvaluationAgent
from src.utils.logging_utils import get_logger

# Get module logger
logger = get_logger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parents[2]
# Config file path relative to project root
CONFIG_PATH = PROJECT_ROOT / "config" / "evaluation.yml"


def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Returns:
        Dictionary containing configuration parameters
    """
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {CONFIG_PATH}")
    if config is None:
        return {}

    # Explicitly cast to the expected return type
    result: Dict[str, Any] = config
    return result


def evaluate_user_queries(user_email: str, query: str = None):
    """
    Run the evaluation agent for a specific user.

    Args:
        user_email: Email of the user for user-specific database evaluation
        query: Optional query string. If None, uses query from config
        auto_discover_collections: If True, auto-discovers all collections for user. If False, uses collections from config
    """
    # Load configuration
    config = load_config()

    # Construct user-specific database path
    base_db_path = PROJECT_ROOT / "src" / "database" / user_email
    user_db_path = str(base_db_path)

    logger.info(f"Running evaluation for user: {user_email}")
    logger.info(f"Using database path: {user_db_path}")

    # Use provided query or get from config
    evaluation_query = query or config.get("query", "No query provided")

    # Initialize evaluation agent
    agent = EvaluationAgent(model_name=config.get("model_name"))

    # Run query on collections with user-specific database path
    results_file = agent.query_collections(
        query=evaluation_query,
        db_path=user_db_path,
        user_email=user_email,
    )

    # Evaluate results
    evaluation = agent.evaluate_results(results_file)

    # Output results
    output_dir = config.get("output_dir")
    if output_dir:
        # Create user-specific output directory
        user_output_dir = Path(output_dir) / user_email
        os.makedirs(user_output_dir, exist_ok=True)
        output_file = user_output_dir / "evaluation_results.json"
        with open(output_file, "w") as f:
            json.dump(evaluation, f, indent=2)
        logger.info(f"Saved evaluation results to {output_file}")

    # Also save to temp directory with user-specific path
    temp_eval_dir = PROJECT_ROOT / "temp" / user_email / "evaluation"
    os.makedirs(temp_eval_dir, exist_ok=True)
    temp_output_file = temp_eval_dir / "evaluation_results.json"
    with open(temp_output_file, "w") as f:
        json.dump(evaluation, f, indent=2)
    logger.info(f"Saved evaluation results to {temp_output_file}")

    return evaluation


def main():
    """Main function to run the evaluation agent using config file."""
    # Load configuration
    config = load_config()

    # Get user email from config
    user_email = config.get("user_email", "user@example.com")

    # Call the user-specific evaluation function with auto-discovery
    evaluation = evaluate_user_queries(user_email)

    # Print results to stdout
    print(json.dumps(evaluation, indent=2))


# Export the function for API usage
__all__ = ["evaluate_user_queries", "main"]


if __name__ == "__main__":
    main()
