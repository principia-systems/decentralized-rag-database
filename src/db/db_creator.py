"""
Database creation module.

This module provides functionality for creating and populating databases
with scientific document data for the system.
"""

import json
from pathlib import Path
import os

import requests
from dotenv import load_dotenv

from src.utils.ipfs_utils import get_ipfs_client
from src.utils.logging_utils import get_logger, get_user_logger


load_dotenv()


class DatabaseCreator:
    """
    Creates and populates vector databases from graph relationships.

    This class retrieves embeddings and content from IPFS based on graph
    relationships and inserts them into ChromaDB collections.
    """

    def __init__(self, graph, vector_db_manager, user_email=None, light_server_url=None):
        """
        Initialize the DatabaseCreator.

        Args:
            graph: IPFSNeo4jGraph instance for graph database operations
            vector_db_manager: VectorDatabaseManager instance for vector database operations
            user_email: Optional user email for user-specific logging
            light_server_url: URL of the light server for batch retrieval
        """
        self.graph = graph
        self.vector_db_manager = vector_db_manager
        self.user_email = user_email
        self.light_server_url = light_server_url or os.getenv('LIGHT_SERVER_URL', 'http://localhost:5001')
        
        # Initialize IPFS client
        self.ipfs_client = get_ipfs_client()
        
        # Use user-specific logger if user_email is provided
        if user_email:
            self.logger = get_user_logger(user_email, "database_creator")
        else:
            self.logger = get_logger(__name__ + ".DatabaseCreator")

    def query_lighthouse_for_embedding(self, cid):
        """
        Query IPFS for an embedding vector.

        Args:
            cid: IPFS CID of the embedding

        Returns:
            List representation of the embedding vector or None if retrieval fails
        """
        try:
            content = self.ipfs_client.get_content(cid)
            embedding_vector = json.loads(content)
            return embedding_vector
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to retrieve embedding for CID {cid}: {e}")
            return None

    def query_ipfs_content(self, cid):
        try:
            content = self.ipfs_client.get_content(cid)
            return content
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to retrieve IPFS content for CID {cid}: {e}")
            return None

    def batch_retrieve_data(self, embedding_cids, content_cids):
        """
        Batch retrieve embeddings and content from the light server.
        
        Args:
            embedding_cids: List of embedding CIDs to retrieve
            content_cids: List of content CIDs to retrieve
            
        Returns:
            Tuple of (embeddings_dict, contents_dict) or (None, None) if failed
        """
        try:
            request_data = {
                "embedding_cids": embedding_cids,
                "content_cids": content_cids,
                "user_email": self.user_email or "system"
            }
            
            response = requests.post(
                f"{self.light_server_url}/api/ipfs/batch-retrieve",
                json=request_data,
                timeout=600  # 10 minute timeout for batch operations
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Log any failures
            if result["failed_embeddings"]:
                self.logger.warning(f"Failed to retrieve {len(result['failed_embeddings'])} embeddings: {result['failed_embeddings']}")
            if result["failed_contents"]:
                self.logger.warning(f"Failed to retrieve {len(result['failed_contents'])} contents: {result['failed_contents']}")
            
            self.logger.info(f"Successfully batch retrieved {len(result['embeddings'])} embeddings and {len(result['contents'])} contents")
            
            return result["embeddings"], result["contents"]
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to batch retrieve data from light server: {e}")
            return None, None
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Invalid response format from light server: {e}")
            return None, None

    def process_paths(self, start_cid, path, db_name):
        paths = self.graph.recreate_path(start_cid, path)

        if paths is False:
            self.logger.error(f"No valid paths found for CID {start_cid}")
            return

        self.logger.info(f"Found {len(paths)} paths for CID {start_cid}")

        # Collect all CIDs for batch retrieval
        embedding_cids = []
        content_cids = []
        path_metadata = []  # Store path info for later processing

        for path_nodes in paths:
            if len(path_nodes) < 2:
                self.logger.error(f"Path {path_nodes} is too short.")
                continue

            content_cid = path_nodes[-2]
            embedding_cid = path_nodes[-1]
            
            embedding_cids.append(embedding_cid)
            content_cids.append(content_cid)
            path_metadata.append({
                "content_cid": content_cid,
                "embedding_cid": embedding_cid,
                "path_nodes": path_nodes
            })

        if not embedding_cids or not content_cids:
            self.logger.warning("No valid CIDs found for batch retrieval")
            return

        self.logger.info(f"Preparing batch retrieval for {len(embedding_cids)} embeddings and {len(content_cids)} contents")

        # Batch retrieve all data from light server
        embeddings_dict, contents_dict = self.batch_retrieve_data(embedding_cids, content_cids)
        
        if embeddings_dict is None or contents_dict is None:
            self.logger.error("Batch retrieval failed, aborting path processing")
            return

        # Process and insert documents one by one
        successful_inserts = 0
        for path_info in path_metadata:
            content_cid = path_info["content_cid"]
            embedding_cid = path_info["embedding_cid"]
            
            # Check if we have both embedding and content
            if embedding_cid not in embeddings_dict:
                self.logger.error(f"Missing embedding for CID {embedding_cid}, skipping")
                continue
                
            if content_cid not in contents_dict:
                self.logger.error(f"Missing content for CID {content_cid}, skipping")
                continue

            embedding_vector = embeddings_dict[embedding_cid]
            content = contents_dict[content_cid]

            metadata = {
                "content_cid": content_cid,
                "root_cid": start_cid,
                "embedding_cid": embedding_cid,
                "content": content,
            }

            try:
                self.vector_db_manager.insert_document(
                    db_name, embedding_vector, metadata, embedding_cid
                )
                successful_inserts += 1
                self.logger.debug(f"Inserted document into '{db_name}' with CID {embedding_cid}")
            except Exception as e:
                self.logger.error(f"Failed to insert document into '{db_name}': {e}")

        self.logger.info(f"Successfully inserted {successful_inserts}/{len(path_metadata)} documents into '{db_name}'")
