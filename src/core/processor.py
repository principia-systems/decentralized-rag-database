"""
Document processing module.

This module provides a Processor class for handling the end-to-end processing
of scientific documents, including conversion, chunking, embedding, and storage.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import certifi
import requests

from src.core.chunker import chunk
from src.core.converter import convert
from src.core.embedder import embed_batch
from src.db.graph_db import IPFSNeo4jGraph
from src.utils.ipfs_utils import get_ipfs_client
from src.utils.logging_utils import get_logger, get_user_logger


class Processor:
    """Base class for text processing."""

    def __init__(
        self,
        authorPublicKey: str,
        user_email: str,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize the processor.

        Args:
            authorPublicKey: Public key of the author
            user_email: Email of the user for creating user-specific folders
            project_root: Path to project root directory
        """
        # Use user-specific logger
        self.logger = get_user_logger(user_email, "processor") if user_email else get_logger(__name__ + ".Processor")
        
        self.authorPublicKey = authorPublicKey  # Author Public Key
        
        # Initialize IPFS client
        self.ipfs_client = get_ipfs_client()
        
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

        # Set SSL certificate path explicitly
        os.environ["SSL_CERT_FILE"] = certifi.where()

        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        self.graph_db = IPFSNeo4jGraph(
            uri=neo4j_uri, username=neo4j_username, password=neo4j_password
        )

        # Get OpenRouter API key for metadata extraction
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        self.__write_to_file(self.authorPublicKey, str(self.tmp_file_path))
        self.logger.info(
            f"Uploading author public key to IPFS: {self.authorPublicKey[:10]}..."
        )
        self.author_cid = self.ipfs_client.upload_file(
            str(self.tmp_file_path)
        )
        self.logger.info(f"Author CID: {self.author_cid}")
        self.graph_db.add_ipfs_node(self.author_cid)

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

    def __read_mappings(self, mapping_file_path: Union[str, Path]) -> Dict[str, List[str]]:
        """Read mappings from JSON file.
        
        Args:
            mapping_file_path: Path to the mappings JSON file
            
        Returns:
            Dictionary mapping PDF CIDs to list of database combinations
        """
        try:
            if os.path.exists(mapping_file_path):
                with open(mapping_file_path, "r") as file:
                    content = file.read().strip()
                    if not content:
                        self.logger.debug(f"Mappings file {mapping_file_path} is empty, returning empty dict")
                        return {}
                    return json.loads(content)
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error reading mappings from {mapping_file_path}: {e}")
            # Return empty dict and try to fix the file
            try:
                with open(mapping_file_path, "w") as file:
                    json.dump({}, file)
                self.logger.info(f"Reset corrupted mappings file {mapping_file_path}")
            except Exception as write_error:
                self.logger.error(f"Failed to reset mappings file: {write_error}")
            return {}
        except Exception as e:
            self.logger.error(f"Error reading mappings from {mapping_file_path}: {e}")
            return {}

    def __write_mappings(self, mappings: Dict[str, List[str]], mapping_file_path: Union[str, Path]) -> None:
        """Write mappings to JSON file.
        
        Args:
            mappings: Dictionary mapping PDF CIDs to list of database combinations
            mapping_file_path: Path to the mappings JSON file
        """
        try:
            os.makedirs(os.path.dirname(str(mapping_file_path)), exist_ok=True)
            with open(mapping_file_path, "w") as file:
                json.dump(mappings, file, indent=2)
            self.logger.debug(f"Updated mappings in {mapping_file_path}")
        except Exception as e:
            self.logger.error(f"Error writing mappings to {mapping_file_path}: {e}")

    def _query_ipfs_content(self, cid):
        """
        Retrieves the content stored in IPFS for a given CID.

        :param cid: The IPFS CID.
        :return: The content of the IPFS file as a string.
        """
        try:
            content = self.ipfs_client.get_content(cid)
            return content.strip()  # Ensure leading/trailing spaces are removed
        except Exception as e:
            self.logger.error(f"Failed to retrieve IPFS content for CID {cid}: {e}")
            return None

    def _extract_metadata_with_openrouter(self, markdown_content: str, model_name: str = "openai/gpt-5") -> Optional[Dict[str, Any]]:
        """
        Extract metadata from markdown content using OpenRouter API.
        
        Args:
            markdown_content: The markdown content to extract metadata from
            model_name: The OpenRouter model to use for extraction
            
        Returns:
            Dictionary containing extracted metadata or None if extraction fails
        """
        if not self.openrouter_api_key:
            self.logger.warning("OpenRouter API key not available. Cannot extract metadata.")
            return None
            
        # Create a prompt for metadata extraction
        system_prompt = """You are a helpful assistant that extracts metadata from academic papers in markdown format. 
            Extract the following information and return it as a valid JSON object:
            - title: The paper title
            - authors: List of author names (as an array)
=            - categories: Research categories/fields (as an array)
            - doi: DOI if available
            - keywords: Key terms/concepts (as an array)
            - publication_date: Publication date if mentioned
            - journal: Journal name if mentioned
            - citation: A properly formatted citation in APA style

            Example JSON structure (use appropriate values from the paper):
            {
            "title": "Deep Learning for Natural Language Processing: A Survey",
            "authors": ["John Smith", "Jane Doe", "Bob Johnson"],
            "categories": ["Computer Science", "Natural Language Processing", "Machine Learning"],
            "doi": "10.1000/182",
            "keywords": ["deep learning", "natural language processing", "neural networks"],
            "publication_date": "2023-05-15",
            "journal": "Journal of Machine Learning Research",
            "citation": "Smith, J., Doe, J., & Johnson, B. (2023). Deep Learning for Natural Language Processing: A Survey. Journal of Machine Learning Research, 24(5), 123-145. https://doi.org/10.1000/182"
            }

            If any field is not found, use appropriate default values like "Unknown Title", ["Unknown Authors"], "No abstract available", etc.
            Return ONLY the JSON object, no additional text."""

        user_prompt = f"Please extract metadata from this academic paper:\n\n{markdown_content[:8000]}"  # Limit content to avoid token limits
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "HTTP-Referer": "https://coophive.com",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000,
                },
                timeout=30,
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                
                # Try to parse the JSON response
                try:
                    metadata = json.loads(content)
                    self.logger.info("Successfully extracted metadata using OpenRouter")
                    return metadata
                except json.JSONDecodeError:
                    # If response isn't valid JSON, try to extract it
                    self.logger.warning("OpenRouter response wasn't valid JSON, trying to parse...")
                    try:
                        # Look for JSON-like content in the response
                        start_idx = content.find('{')
                        end_idx = content.rfind('}') + 1
                        if start_idx != -1 and end_idx != 0:
                            json_str = content[start_idx:end_idx]
                            metadata = json.loads(json_str)
                            self.logger.info("Successfully parsed metadata from OpenRouter response")
                            return metadata
                    except (json.JSONDecodeError, ValueError):
                        pass
                    
                    self.logger.error(f"Failed to parse OpenRouter response as JSON: {content}")
                    return None
            else:
                self.logger.error("No response content from OpenRouter")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error("OpenRouter API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenRouter API request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during metadata extraction: {e}")
            return None

    def _create_or_get_metadata_node(self, pdf_cid: str, converted_text: str, doc_id: Optional[str] = None) -> Optional[str]:
        """
        Create or retrieve a metadata node for the given PDF.
        
        Args:
            pdf_cid: The PDF CID to create/get metadata for
            converted_text: The converted markdown text to extract metadata from
            
        Returns:
            The metadata CID if successful, None otherwise
        """
        # Check if metadata already exists
        existing_metadata_cid = self.graph_db.get_existing_metadata_cid(pdf_cid)
        if existing_metadata_cid:
            self.logger.info(f"Using existing metadata node: {existing_metadata_cid}")
            return existing_metadata_cid
        
        self.logger.info("No existing metadata found, extracting with OpenRouter...")
        
        # Extract metadata using OpenRouter
        extracted_metadata = self._extract_metadata_with_openrouter(converted_text)
        
        if not extracted_metadata:
            self.logger.warning("Failed to extract metadata with OpenRouter, using default metadata")
            extracted_metadata = self.graph_db.default_metadata()
        
        # Add doc_id to metadata if provided
        if doc_id:
            extracted_metadata["pdf_filename"] = doc_id
        else:
            extracted_metadata["pdf_filename"] = "Unknown"
        
        # Serialize metadata as JSON
        try:
            metadata_json = json.dumps(extracted_metadata, indent=2)
            
            # Write metadata to temporary file
            self.__write_to_file(metadata_json, self.tmp_file_path)
            
            # Upload metadata to IPFS
            metadata_cid = self.ipfs_client.upload_file(str(self.tmp_file_path))
            
            if not metadata_cid:
                self.logger.error("Failed to upload metadata to IPFS")
                return None
                
            self.logger.info(f"Created new metadata node: {metadata_cid}")
            
            # Create metadata node and relationship using graph database
            if self.graph_db.create_metadata_node(pdf_cid, metadata_cid):
                return metadata_cid
            else:
                self.logger.error("Failed to create metadata node in graph database")
                return None
            
        except Exception as e:
            self.logger.error(f"Error creating metadata node: {e}")
            return None

    def __update_mappings(self, pdf_cid: str, db_combination: str) -> None:
        """Update both global and user-specific mappings.
        
        Args:
            pdf_cid: The PDF CID that was processed
            db_combination: The database combination in format "converter_chunker_embedder"
        """
        # Global mappings file
        global_mappings_path = self.temp_dir / "mappings.json"
        global_mappings = self.__read_mappings(global_mappings_path)
        
        if pdf_cid not in global_mappings:
            global_mappings[pdf_cid] = []
        if db_combination not in global_mappings[pdf_cid]:
            global_mappings[pdf_cid].append(db_combination)
        
        self.__write_mappings(global_mappings, global_mappings_path)
        
        # User-specific mappings file
        user_mappings_path = self.user_temp_dir / "mappings.json"
        user_mappings = self.__read_mappings(user_mappings_path)
        
        if pdf_cid not in user_mappings:
            user_mappings[pdf_cid] = []
        if db_combination not in user_mappings[pdf_cid]:
            user_mappings[pdf_cid].append(db_combination)
        
        self.__write_mappings(user_mappings, user_mappings_path)

    def process(self, pdf_path: str, databases: List[dict]) -> None:
        """
        Processes the PDF according to the list of database configurations passed.

        Args:
            pdf_path: Path to the input PDF
            databases: A list of configs, each containing a converter, chunker, and embedder
        """
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        self.logger.info(f"Processing document: {doc_id}")
        self.convert_cache = {}
        self.chunk_cache = {}

        metadata = {}

        self.logger.info(f"Uploading PDF to IPFS: {pdf_path}")
        metadata["pdf_ipfs_cid"] = self.ipfs_client.upload_file(pdf_path)

        if not metadata["pdf_ipfs_cid"]:
            self.logger.error(f"Failed to upload PDF to IPFS: {pdf_path}")
            return

        self.logger.info(f"Adding PDF CID to graph: {metadata['pdf_ipfs_cid']}")
        self.graph_db.add_ipfs_node(metadata["pdf_ipfs_cid"])
        
        # Handle metadata extraction and node creation
        metadata_cid = None
        
        for db_config in databases:
            
            converter_func = db_config["converter"]
            chunker_func = db_config["chunker"]
            embedder_func = db_config["embedder"]
            db_combination = f"{converter_func}_{chunker_func}_{embedder_func}"

            # Check if this PDF + database combination already exists in global mappings
            global_mappings_path = self.temp_dir / "mappings.json"
            global_mappings = self.__read_mappings(global_mappings_path)
            
            self.logger.debug(f"Checking if {db_combination} already exists for PDF CID {metadata['pdf_ipfs_cid']}")
            
            if metadata["pdf_ipfs_cid"] in global_mappings and db_combination in global_mappings[metadata["pdf_ipfs_cid"]]:
                self.logger.info(f"Skipping {db_combination} - already processed for this PDF")
                self.__update_mappings(metadata["pdf_ipfs_cid"], db_combination)
                continue

            self.logger.info(f"Processing new combination: {db_combination}")

            # Collect all nodes that need AUTHORED_BY relationships
            all_authored_nodes = []

            # Step 2.1: Conversion
            # Check if markdown conversion already exists for this PDF CID
            converted_text_ipfs_cid = self.graph_db.get_converted_markdown_cid(
                metadata["pdf_ipfs_cid"], converter_func
            )

            # If the conversion already exists, use the existing conversion
            if converted_text_ipfs_cid:
                # Fetch converted text content from IPFS
                converted_text = self._query_ipfs_content(
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

                converted_text_ipfs_cid = self.ipfs_client.upload_file(self.tmp_file_path)

                self.graph_db.add_ipfs_node(converted_text_ipfs_cid)
                self.graph_db.create_relationship(
                    metadata["pdf_ipfs_cid"],
                    converted_text_ipfs_cid,
                    "CONVERTED_BY_" + converter_func,
                )
                all_authored_nodes.append(converted_text_ipfs_cid)

            # Step 2.1.5: Handle metadata creation (only once per document)
            if metadata_cid is None:
                self.logger.info("Creating or retrieving metadata node...")
                metadata_cid = self._create_or_get_metadata_node(
                    metadata["pdf_ipfs_cid"], 
                    self.convert_cache[converter_func],
                    doc_id
                )
                if metadata_cid:
                    all_authored_nodes.append(metadata_cid)
                    self.logger.info(f"Metadata node ready: {metadata_cid}")
                else:
                    self.logger.warning("Failed to create metadata node")

            # Step 2.2: Chunking
            self.logger.info(f"Starting chunking process for {converter_func}_{chunker_func}")
            converted_text = self.convert_cache[converter_func]
            chunk_cache_key = f"{converter_func}_{chunker_func}"
            if chunk_cache_key not in self.chunk_cache:
                self.logger.debug(f"Chunking text with {chunker_func}")
                chunked_text = chunk(
                    chunker_type=chunker_func, input_text=converted_text
                )
                self.chunk_cache[chunk_cache_key] = chunked_text
            else:
                chunked_text = self.chunk_cache[chunk_cache_key]
                self.logger.debug(f"Using cached chunks for {chunk_cache_key}")

            # Step 2.2.1: Process all chunks and get their CIDs
            chunk_cids = []
            self.logger.info(f"Processing {len(chunked_text)} chunks...")
            
            for chunk_i in chunked_text:
                self.__write_to_file(chunk_i, self.tmp_file_path)

                chunk_text_ipfs_cid = self.ipfs_client.upload_file(self.tmp_file_path)
                chunk_cids.append(chunk_text_ipfs_cid)

            # Batch add all chunk nodes to graph
            self.logger.info(f"Adding {len(chunk_cids)} chunk nodes to graph...")
            self.graph_db.add_ipfs_nodes_batch(chunk_cids)
            
            # Add chunk CIDs to authored nodes list
            all_authored_nodes.extend(chunk_cids)
            
            # Create CHUNKED_BY relationships for chunks
            chunk_relationships = []
            for chunk_cid in chunk_cids:
                chunk_relationships.append((converted_text_ipfs_cid, chunk_cid, f"CHUNKED_BY_{chunker_func}"))
            
            self.logger.info(f"Creating {len(chunk_relationships)} CHUNKED_BY relationships...")
            self.graph_db.create_relationships_batch(chunk_relationships)

            # Step 2.3: Batch Embedding
            self.logger.info(f"Batch processing embeddings for {len(chunked_text)} chunks...")
            embeddings = embed_batch(embeder_type=embedder_func, input_texts=chunked_text, batch_size=32, user_email=self.user_email)
            
            # Step 2.4: Create embedding relationships
            embedding_cids = []
            for embedding in embeddings:
                self.__write_to_file(json.dumps(embedding), self.tmp_file_path)

                embedding_ipfs_cid = self.ipfs_client.upload_file(self.tmp_file_path)
                embedding_cids.append(embedding_ipfs_cid)
            
            # Batch add all embedding nodes to graph
            self.logger.info(f"Adding {len(embedding_cids)} embedding nodes to graph...")
            self.graph_db.add_ipfs_nodes_batch(embedding_cids)
            
            # Add embedding CIDs to authored nodes list
            all_authored_nodes.extend(embedding_cids)
            
            # Create EMBEDDED_BY relationships for embeddings
            embedding_relationships = []
            for chunk_cid, embedding_cid in zip(chunk_cids, embedding_cids):
                embedding_relationships.append((chunk_cid, embedding_cid, f"EMBEDDED_BY_{embedder_func}"))
                
            self.logger.info(f"Creating {len(embedding_relationships)} EMBEDDED_BY relationships...")
            self.graph_db.create_relationships_batch(embedding_relationships)
            
            # Create all AUTHORED_BY relationships in one batch
            authored_relationships = [(node_cid, self.author_cid, "AUTHORED_BY") for node_cid in all_authored_nodes]
            self.logger.info(f"Creating {len(authored_relationships)} AUTHORED_BY relationships in one batch...")
            self.graph_db.create_relationships_batch(authored_relationships)
                
            self.__update_mappings(metadata["pdf_ipfs_cid"], db_combination)