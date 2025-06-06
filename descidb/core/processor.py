"""
Document processing module for DeSciDB.

This module provides a Processor class for handling the end-to-end processing
of scientific documents, including conversion, chunking, embedding, and storage.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import certifi
import requests

from descidb.core.chunker import chunk
from descidb.core.converter import convert
from descidb.core.embedder import embed
from descidb.db.chroma_client import VectorDatabaseManager
from descidb.db.graph_db import IPFSNeo4jGraph
from descidb.db.postgres_db import PostgresDBManager
from descidb.rewards.token_rewarder import TokenRewarder
from descidb.utils.logging_utils import get_logger

# Get module logger
logger = get_logger(__name__)


class Processor:
    """Base class for text processing."""

    def __init__(
        self,
        authorPublicKey: str,
        db_manager: VectorDatabaseManager,
        postgres_db_manager: PostgresDBManager,
        metadata_file: str,
        ipfs_api_key: str,
        TokenRewarder: TokenRewarder,
        user_email: str,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize the processor.

        Args:
            authorPublicKey: Public key of the author
            db_manager: Vector database manager instance
            postgres_db_manager: PostgreSQL database manager instance
            metadata_file: Path to metadata file
            ipfs_api_key: API key for Lighthouse IPFS
            TokenRewarder: Token rewarder instance
            user_email: Email of the user for creating user-specific folders
            project_root: Path to project root directory
        """
        self.logger = get_logger(__name__ + ".Processor")
        self.db_manager = db_manager  # Vector Database Manager
        self.TokenRewarder = TokenRewarder
        self.metadata_file = metadata_file
        self.authorPublicKey = authorPublicKey  # Author Public Key
        self.ipfs_api_key = ipfs_api_key  # IPFS API Key
        self.postgres_db_manager = postgres_db_manager  # Postgres DB Manager
        self.convert_cache: Dict[str, str] = {}  # Cache for converted text
        self.chunk_cache: Dict[str, List[str]] = {}  # Cache for chunked text
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.user_email = user_email

        # Create temp directory for temporary files
        self.temp_dir = self.project_root / "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create user-specific folder inside temp directory
        # Sanitize email for use as folder name (replace @ and . with _)
        self.user_temp_dir = self.temp_dir / self.user_email
        os.makedirs(self.user_temp_dir, exist_ok=True)
        
        self.logger.info(f"Using user temp directory: {self.user_temp_dir}")

        # Paths for user-specific temporary files
        self.tmp_file_path = self.user_temp_dir / "tmp.txt"
        self.cids_file_path = self.user_temp_dir / "cids.txt"

        # Set SSL certificate path explicitly
        os.environ["SSL_CERT_FILE"] = certifi.where()

        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        self.graph_db = IPFSNeo4jGraph(
            uri=neo4j_uri, username=neo4j_username, password=neo4j_password
        )

        self.__write_to_file(self.authorPublicKey, str(self.tmp_file_path))
        self.logger.info(
            f"Uploading author public key to Lighthouse: {self.authorPublicKey[:10]}..."
        )
        self.author_cid = self.__upload_text_to_lighthouse(
            str(self.tmp_file_path)
        ).split("ipfs/")[-1]
        self.logger.info(f"Author CID: {self.author_cid}")
        self.graph_db.add_ipfs_node(self.author_cid)

    def __upload_text_to_lighthouse(self, filename: str) -> str:
        """Uploads a string as a file to Lighthouse IPFS and returns the IPFS hash (CID).

        - content: The string content to be uploaded.
        - filename: The name of the file for the uploaded content.
        - Returns: IPFS hash (CID) of the uploaded file.
        """
        url = "https://node.lighthouse.storage/api/v0/add"

        headers = {"Authorization": f"Bearer {self.ipfs_api_key}"}

        with open(filename, "rb") as file:
            files = {"file": file}
            response = requests.post(url, headers=headers, files=files)

        response.raise_for_status()

        # Explicitly cast to string to satisfy type checker
        hash_value: str = response.json()["Hash"]
        return hash_value

    def __create_file_with_ipfs(self, content: str, file_path: str) -> str:
        """Creates a file with the IPFS CID and returns the CID.

        - content: The string content to be uploaded.
        - file_path: The name of the file for the uploaded content.
        - Returns: IPFS hash (CID) of the uploaded file.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as file:
                file.write(content)
            return file_path  # Return the file path as a string
        except Exception as e:
            self.logger.error(f"Error creating file {file_path}: {e}")
            return ""  # Return empty string in case of error

    def __lighthouse_and_commit(self, object: Union[str, Path], git_path: str) -> str:
        """Uploads a file to Lighthouse IPFS and commits the CID to git.

        - object: Path to the object to be uploaded.
        - Returns: IPFS hash (CID) of the uploaded file.
        """
        try:
            object_str = str(object)
            self.logger.info(f"Uploading to Lighthouse: {object_str}")
            ipfs_cid = self.__upload_text_to_lighthouse(object_str)

            hash_value = ipfs_cid.split("ipfs/")[-1]
            self.logger.info(f"Generated IPFS CID: {hash_value}")

            file_path = os.path.join(git_path, f"{hash_value}.txt")

            self.__create_file_with_ipfs(ipfs_cid, file_path)

            subprocess.run(["git", "-C", git_path, "add", file_path], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    git_path,
                    "commit",
                    "-m",
                    f"Added IPFS CID: {hash_value}",
                ],
                check=True,
            )

            return ipfs_cid

        except Exception as e:
            self.logger.error(f"Error during Git commit process: {e}")
            return ""

    def __write_to_file(self, content: str, file_path: Union[str, Path]) -> None:
        """Writes the content to a file.

        - content: The content to be written to the file.
        - file_path: The path to the file to write the content to.
        """
        try:
            path_str = str(file_path)
            os.makedirs(os.path.dirname(path_str), exist_ok=True)
            with open(path_str, "w") as file:
                file.write(content)
        except Exception as e:
            self.logger.error(f"Error writing to file {file_path}: {e}")

    def process(self, pdf_path: str, databases: List[dict], git_path: str) -> None:
        """
        Processes the PDF according to the list of database configurations passed.

        Args:
            pdf_path: Path to the input PDF
            databases: A list of configs, each containing a converter, chunker, and embedder
            git_path: Path to git repository for storing CIDs
        """
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        self.logger.info(f"Processing document: {doc_id}")
        self.convert_cache = {}
        self.chunk_cache = {}

        metadata = self.get_metadata_for_doc(self.metadata_file, doc_id)
        if not metadata:
            self.logger.warning(f"No metadata found for {doc_id}, using default")
            metadata = self.default_metadata(doc_id)

        metadata = {
            key: (
                json.dumps(value)
                if isinstance(value, (list, dict))
                else value
                if value is not None
                else "N/A"
            )
            for key, value in metadata.items()
        }

        self.logger.info(f"Uploading PDF to IPFS: {pdf_path}")
        metadata["pdf_ipfs_cid"] = self.__lighthouse_and_commit(
            object=pdf_path, git_path=git_path
        )

        if not metadata["pdf_ipfs_cid"]:
            self.logger.error(f"Failed to upload PDF to IPFS: {pdf_path}")
            return

        self.logger.info(f"Adding PDF CID to graph: {metadata['pdf_ipfs_cid']}")
        self.graph_db.add_ipfs_node(metadata["pdf_ipfs_cid"])
        self.graph_db.create_relationship(
            metadata["pdf_ipfs_cid"], self.author_cid, "AUTHORED_BY"
        )

        with open(self.cids_file_path, "a") as cid_file:
            cid_file.write(metadata["pdf_ipfs_cid"] + "\n")
            self.logger.debug(f"Recorded CID in {self.cids_file_path}")

        for db_config in databases:
            converter_func = db_config["converter"]
            chunker_func = db_config["chunker"]
            embedder_func = db_config["embedder"]

            # Step 2.1: Conversion
            # Check if markdown conversion already exists for this PDF CID
            converted_text_ipfs_cid = self.graph_db.get_converted_markdown_cid(
                metadata["pdf_ipfs_cid"], converter_func
            )

            # If the conversion already exists, use the existing conversion
            if converted_text_ipfs_cid:
                self.logger.info(
                    f"Found existing markdown conversion: {converted_text_ipfs_cid}"
                )
                # Fetch converted text content from IPFS
                converted_text = self.graph_db._query_ipfs_content(
                    converted_text_ipfs_cid
                )
                if converted_text:
                    self.convert_cache[converter_func] = converted_text
                    self.logger.info("Using existing markdown conversion")
                else:
                    self.logger.warning(
                        "Failed to fetch existing conversion content, performing new conversion"
                    )
                    converted_text_ipfs_cid = None  # Reset to trigger new conversion

            # If no existing conversion was found or content could not be fetched, perform conversion
            if not converted_text_ipfs_cid or converter_func not in self.convert_cache:
                self.logger.info(
                    "No existing conversion found, performing new conversion"
                )
                if converter_func not in self.convert_cache:
                    converted_text = convert(
                        conversion_type=converter_func, input_path=pdf_path
                    )
                    self.convert_cache[converter_func] = converted_text
                else:
                    converted_text = self.convert_cache[converter_func]

                # Upload converted text to IPFS and commit to Git
                self.__write_to_file(converted_text, self.tmp_file_path)

                converted_text_ipfs_cid = self.__lighthouse_and_commit(
                    object=self.tmp_file_path, git_path=git_path
                )

                self.graph_db.add_ipfs_node(converted_text_ipfs_cid)
                self.graph_db.create_relationship(
                    metadata["pdf_ipfs_cid"],
                    converted_text_ipfs_cid,
                    "CONVERTED_BY_" + converter_func,
                )
                self.graph_db.create_relationship(
                    converted_text_ipfs_cid, self.author_cid, "AUTHORED_BY"
                )

            # Step 2.2: Chunking
            converted_text = self.convert_cache[converter_func]
            chunk_cache_key = f"{converter_func}_{chunker_func}"
            if chunk_cache_key not in self.chunk_cache:
                chunked_text = chunk(
                    chunker_type=chunker_func, input_text=converted_text
                )
                self.chunk_cache[chunk_cache_key] = chunked_text
            else:
                chunked_text = self.chunk_cache[chunk_cache_key]

            for chunk_index, chunk_i in enumerate(chunked_text):
                self.__write_to_file(chunk_i, self.tmp_file_path)

                chunk_text_ipfs_cid = self.__lighthouse_and_commit(
                    object=self.tmp_file_path, git_path=git_path
                )

                self.graph_db.add_ipfs_node(chunk_text_ipfs_cid)
                self.graph_db.create_relationship(
                    converted_text_ipfs_cid,
                    chunk_text_ipfs_cid,
                    "CHUNKED_BY_" + chunker_func,
                )
                self.graph_db.create_relationship(
                    chunk_text_ipfs_cid, self.author_cid, "AUTHORED_BY"
                )
                # Step 2.3: Embedding

                embedding = embed(embeder_type=embedder_func, input_text=chunk_i)

                self.__write_to_file(json.dumps(embedding), self.tmp_file_path)

                embedding_ipfs_cid = self.__lighthouse_and_commit(
                    object=self.tmp_file_path, git_path=git_path
                )
                self.graph_db.add_ipfs_node(embedding_ipfs_cid)
                self.graph_db.create_relationship(
                    chunk_text_ipfs_cid,
                    embedding_ipfs_cid,
                    "EMBEDDED_BY_" + embedder_func,
                )
                self.graph_db.create_relationship(
                    embedding_ipfs_cid, self.author_cid, "AUTHORED_BY"
                )

    def get_metadata_for_doc(self, metadata_file: str, doc_id: str) -> Dict[str, Any]:
        """Retrieves metadata for the given document ID from the metadata file.

        - metadata_file: Path to the metadata file.
        - doc_id: Document ID to retrieve metadata for.
        - Returns: Dictionary containing metadata or empty dict if not found.
        """
        with open(metadata_file, "r") as file:
            for line in file:
                try:
                    data = json.loads(line)
                    if data.get("id") == doc_id:
                        return data  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    continue
        return {}

    def default_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Returns default metadata in case None is found.

        The metadata.json file aready exists for Arxiv papers.

        Returns:
            A dictionary with default metadata values
        """
        # Define a dictionary with an explicit type cast
        metadata: Dict[str, Any] = {}

        # Add each field individually
        metadata["title"] = "Unknown Title"
        metadata["authors"] = "Unknown Authors"
        metadata["categories"] = "Unknown Categories"
        metadata["abstract"] = "No abstract available."
        metadata["doi"] = "No DOI available"

        return metadata
