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

from src.utils.logging_utils import get_logger, get_user_logger


load_dotenv()


class DatabaseCreator:
    """
    Creates and populates vector databases from graph relationships.

    This class retrieves embeddings and content from IPFS based on graph
    relationships and inserts them into ChromaDB collections.
    """

    def __init__(self, graph, vector_db_manager, user_email=None):
        """
        Initialize the DatabaseCreator.

        Args:
            graph: IPFSNeo4jGraph instance for graph database operations
            vector_db_manager: VectorDatabaseManager instance for vector database operations
            user_email: Optional user email for user-specific logging
        """
        self.graph = graph
        self.vector_db_manager = vector_db_manager
        self.user_email = user_email
        self.light_server_url = os.getenv('LIGHT_SERVER_URL', 'http://localhost:5001')
        
        # Use user-specific logger if user_email is provided
        if user_email:
            self.logger = get_user_logger(user_email, "database_creator")
        else:
            self.logger = get_logger(__name__ + ".DatabaseCreator")

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

    def get_pdf_metadata(self, pdf_cid):
        """
        Retrieve metadata for a given PDF CID.
        
        Args:
            pdf_cid: The PDF CID to get metadata for
            
        Returns:
            Dictionary containing metadata or None if not found
        """
        try:
            # Get metadata CID from graph database
            metadata_cid = self.graph.get_existing_metadata_cid(pdf_cid)
            
            if not metadata_cid:
                self.logger.warning(f"No metadata node found for PDF CID {pdf_cid}")
                return None
            
            # Retrieve metadata content from light server
            try:
                request_data = {
                    "embedding_cids": [],
                    "content_cids": [metadata_cid],
                    "user_email": self.user_email or "system"
                }
                
                response = requests.post(
                    f"{self.light_server_url}/api/ipfs/batch-retrieve",
                    json=request_data,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                
                if metadata_cid in result.get("contents", {}):
                    metadata_json = result["contents"][metadata_cid]
                    try:
                        metadata = json.loads(metadata_json)
                        self.logger.debug(f"Retrieved metadata for PDF {pdf_cid}: {metadata.get('title', 'Unknown Title')}")
                        return metadata
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse metadata JSON for PDF {pdf_cid}: {e}")
                        return None
                else:
                    self.logger.warning(f"Metadata content not found for CID {metadata_cid}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to retrieve metadata from light server: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving metadata for PDF {pdf_cid}: {e}")
            return None

    def process_paths(self, start_cid, path, db_name):
        paths = self.graph.recreate_path(start_cid, path)

        if paths is False:
            self.logger.error(f"No valid paths found for CID {start_cid}")
            return

        self.logger.info(f"Found {len(paths)} paths for CID {start_cid}")

        # Retrieve PDF metadata once for all embeddings
        self.logger.info(f"Retrieving metadata for PDF CID {start_cid}")
        pdf_metadata = self.get_pdf_metadata(start_cid)
        
        if pdf_metadata:
            self.logger.info(f"Retrieved metadata: '{pdf_metadata.get('title', 'Unknown Title')}' by {pdf_metadata.get('authors', ['Unknown Authors'])}")
        else:
            self.logger.warning(f"No metadata available for PDF {start_cid}, proceeding without metadata")

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

        # Process and prepare documents for batch insertion
        batch_embeddings = []
        batch_metadatas = []
        batch_doc_ids = []
        skipped_count = 0
        
        for path_info in path_metadata:
            content_cid = path_info["content_cid"]
            embedding_cid = path_info["embedding_cid"]
            
            # Check if we have both embedding and content
            if embedding_cid not in embeddings_dict:
                self.logger.error(f"Missing embedding for CID {embedding_cid}, skipping")
                skipped_count += 1
                continue
                
            if content_cid not in contents_dict:
                self.logger.error(f"Missing content for CID {content_cid}, skipping")
                skipped_count += 1
                continue

            embedding_vector = embeddings_dict[embedding_cid]
            content = contents_dict[content_cid]

            # Build metadata including PDF metadata
            metadata = {
                "content_cid": content_cid,
                "root_cid": start_cid,
                "embedding_cid": embedding_cid,
                "content": content,
                **(pdf_metadata if pdf_metadata else {})
            }

            # Add to batch lists
            batch_embeddings.append(embedding_vector)
            batch_metadatas.append(metadata)
            batch_doc_ids.append(embedding_cid)

        # Perform batch insert if we have valid documents
        successful_inserts = 0
        if batch_embeddings:
            try:
                self.vector_db_manager.batch_insert_documents(
                    db_name, batch_embeddings, batch_metadatas, batch_doc_ids
                )
                successful_inserts = len(batch_embeddings)
                self.logger.info(f"Batch inserted {successful_inserts} documents into '{db_name}' with PDF metadata")
            except Exception as e:
                self.logger.error(f"Failed to batch insert documents into '{db_name}': {e}")
        else:
            self.logger.warning(f"No valid documents to insert into '{db_name}'")

        if skipped_count > 0:
            self.logger.warning(f"Skipped {skipped_count} documents due to missing data")

        self.logger.info(f"Successfully inserted {successful_inserts}/{len(path_metadata)} documents into '{db_name}' with PDF metadata")
