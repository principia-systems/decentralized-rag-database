"""
Token testing module.

This module provides test functions for the token reward system,
allowing verification of token distribution functionality.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.db.graph_db import IPFSNeo4jGraph
from src.rewards.token_rewarder import TokenRewarder
from src.utils.logging_utils import get_logger

# Get module logger
logger = get_logger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment variables
load_dotenv(override=True)


def load_config():
    """
    Load configuration from the config file.

    Returns:
        dict: Configuration dictionary
    """
    config_path = PROJECT_ROOT / "config" / "token_test.yml"

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def run_reward_users():
    """
    Run a test of the token reward system.

    This function connects to Neo4j, fetches author contribution statistics,
    and distributes token rewards based on their contributions across different databases.
    """
    logger.info("Starting token reward test")

    # Load configuration
    config = load_config()

    # Extract Neo4j connection parameters
    neo4j_config = config["neo4j"]
    neo4j_uri = os.getenv(neo4j_config["uri"].replace("${", "").replace("}", ""))
    neo4j_username = os.getenv(
        neo4j_config["username"].replace("${", "").replace("}", "")
    )
    neo4j_password = os.getenv(
        neo4j_config["password"].replace("${", "").replace("}", "")
    )

    # Initialize Neo4j graph connection
    graph = IPFSNeo4jGraph(
        uri=neo4j_uri, username=neo4j_username, password=neo4j_password
    )

    # Fetch author job contributions from Neo4j
    logger.info("Fetching author job contributions from Neo4j")
    author_jobs = graph.get_authored_by_stats()
    logger.info(f"Found {len(author_jobs)} authors with contributions")

    # Get database configurations
    databases = config["databases"]

    # Extract components from database configurations
    components = {
        "converter": list(set([db_config["converter"] for db_config in databases])),
        "chunker": list(set([db_config["chunker"] for db_config in databases])),
        "embedder": list(set([db_config["embedder"] for db_config in databases])),
    }

    # Get token rewarder configuration
    token_config = config["token_rewarder"]
    postgres_config = config["postgres"]
    contract_abi_path = PROJECT_ROOT / token_config["contract_abi_path"]

    # Initialize the TokenRewarder
    logger.info("Initializing TokenRewarder")
    rewarder = TokenRewarder(
        network=token_config["network"],
        contract_address=token_config["contract_address"],
        contract_abi_path=str(contract_abi_path),
        db_components=components,
        host=postgres_config["host"],
        port=postgres_config["port"],
        user=postgres_config["user"],
        password=postgres_config["password"],
    )

    # Process the rewards for each database configuration
    for db_config in databases:
        db_name = (
            f"{db_config['converter']}_{db_config['chunker']}_{db_config['embedder']}"
        )
        logger.info(f"Processing database: {db_name}")

        # Create the database if it doesn't exist
        rewarder._create_database_and_table(db_name)

        # Add rewards for each author
        for author, jobs in author_jobs.items():
            logger.info(f"Adding rewards for author {author}: {jobs} jobs")
            rewarder.add_reward_to_user(author, db_name, jobs)

    logger.info("Token reward test completed")


if __name__ == "__main__":
    run_reward_users()
