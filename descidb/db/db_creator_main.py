"""
Database creation entrypoint for DeSciDB.

This module serves as the main entrypoint for creating and populating databases
with scientific document data for the DeSciDB system.
"""

import os
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv

from descidb.db.chroma_client import VectorDatabaseManager
from descidb.db.db_creator import DatabaseCreator
from descidb.db.graph_db import IPFSNeo4jGraph
from descidb.utils.logging_utils import get_logger

# Get module logger
logger = get_logger(__name__)

load_dotenv()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_config():
    """
    Load configuration from the config file.

    Returns:
        dict: Configuration dictionary
    """
    config_path = PROJECT_ROOT / "config" / "db_creator.yml"

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def create_user_database(user_email: str):
    """
    Create and populate the database from IPFS CIDs for a specific user.
    
    Args:
        user_email: Email of the user for creating user-specific database
    """
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

    logger.info(f"Creating databases for user: {user_email}")

    # Initialize Neo4j graph
    graph = IPFSNeo4jGraph(
        uri=neo4j_uri, username=neo4j_username, password=neo4j_password
    )

    # Extract database components
    components = config["components"]

    # Set up user-specific vector database path
    base_db_path = PROJECT_ROOT / config["vector_db"]["path"]
    user_db_path = base_db_path / user_email
    os.makedirs(user_db_path, exist_ok=True)
    logger.info(f"Using database path: {user_db_path}")

    # Initialize database manager
    vector_db_manager = VectorDatabaseManager(components, db_path=str(user_db_path))
    create_db = DatabaseCreator(graph, vector_db_manager)

    # Construct relationship paths from components
    relationship_path = []
    component_type_mapping = {
        "converter": "CONVERTED",
        "chunker": "CHUNKED",
        "embedder": "EMBEDDED"
    }
    
    for component_type, component_list in components.items():
        for component in component_list:
            relationship = f"{component_type_mapping[component_type]}_BY_{component}"
            relationship_path.append(relationship)

    # Define database name
    converter = components["converter"][0]
    chunker = components["chunker"][0]
    embedder = components["embedder"][0]
    db_name = f"{converter}_{chunker}_{embedder}"

    # Look for user-specific CIDs file paths
    temp_dir = PROJECT_ROOT / "temp" / user_email
    cids_file_paths = [
        temp_dir / "cids.txt",  # User-specific temp directory
    ]

    cids_file = None
    for path in cids_file_paths:
        if path.exists():
            cids_file = path
            logger.info(f"Found CIDs file: {cids_file}")
            break

    if cids_file is None:
        logger.error(f"No cids.txt file found for user {user_email}. Please run processor first.")
        logger.error(f"Checked paths: {[str(p) for p in cids_file_paths]}")
        return

    # Process CIDs
    with open(cids_file, "r") as file:
        counter = 0
        for line in file:
            start_cid = line.strip()
            if start_cid:
                logger.info(f"Processing CID #{counter}: {start_cid}")
                create_db.process_paths(start_cid, relationship_path, db_name)
                counter += 1


def main():
    """
    Main function to create and populate the database from IPFS CIDs.
    Uses the user email from config file.
    """
    # Load configuration
    config = load_config()
    
    # Get user email from config
    user_email = config["user"]["email"]
    
    # Call the user-specific database creation function
    create_user_database(user_email)


if __name__ == "__main__":
    main()
