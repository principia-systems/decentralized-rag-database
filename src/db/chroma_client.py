"""
This module contains the ChromaDB client for managing vector databases.

ChromaDB is used as the vector database for storing and querying document embeddings.
"""

import os
from pathlib import Path
from typing import Optional, List

import chromadb
import numpy as np


class VectorDatabaseManager:
    """
    Manages ChromaDB vector database collections for multiple embedding pipelines.

    This class handles the creation, initialization, and interaction with ChromaDB
    collections, which are used to store document embeddings.
    """

    def __init__(self, db_names: List[str], db_path: Optional[str] = None):
        """
        Initializes databases based on the provided list of database names.

        Args:
            db_names: A list of database names to create (e.g., ['openai_paragraph_openai', 'marker_sentence_bge']).
            db_path: Optional path to the database directory. If not provided,
                    will use the default 'database' directory in the src package.

        Raises:
            ValueError: If db_names is empty or not a list.
        """
        if not isinstance(db_names, list) or not db_names:
            raise ValueError("db_names must be a non-empty list of database names.")

        self.db_names = db_names

        # Use the provided db_path or create a default path
        if db_path is None:
            # Get the directory where this module is located
            module_dir = Path(__file__).parent
            db_path_obj = module_dir / "database"
        else:
            db_path_obj = Path(db_path)

        # Ensure the database directory exists
        os.makedirs(db_path_obj, exist_ok=True)

        self.db_client = chromadb.PersistentClient(path=str(db_path_obj))
        self.initialize_databases()

    def initialize_databases(self):
        """
        Initializes or checks the existence of all databases (Cartesian product of convert, chunker, and embedder).
        """
        for db_name in self.db_names:
            self.db_client.get_or_create_collection(name=db_name)

    def insert_document(
        self, db_name: str, embedding: list, metadata: dict, doc_id: str
    ):
        """
        Inserts a document into the specified database.

        :param db_name: Name of the database where the document is to be inserted.
        :param embedding: The embedding of the document chunk to insert.
        :param metadata: Metadata associated with the document chunk.
        :param doc_id: Document ID to use for insertion.
        """
        if db_name not in self.db_names:
            raise ValueError(f"Database '{db_name}' does not exist.")

        # Insert document into the database
        collection = self.db_client.get_collection(name=db_name)
        try:
            collection.add(
                documents=[metadata["content_cid"]],
                embeddings=embedding,
                ids=[doc_id],
                metadatas=[metadata],
            )
        except Exception as e:
            print(f"Error inserting document into database '{db_name}': {e}")

    def batch_insert_documents(
        self, db_name: str, embeddings: list, metadatas: list, doc_ids: list
    ):
        """
        Batch inserts multiple documents into the specified database in a single transaction.

        :param db_name: Name of the database where the documents are to be inserted.
        :param embeddings: List of embeddings for the document chunks.
        :param metadatas: List of metadata dicts associated with the document chunks.
        :param doc_ids: List of document IDs to use for insertion.
        """
        if db_name not in self.db_names:
            raise ValueError(f"Database '{db_name}' does not exist.")

        if not (len(embeddings) == len(metadatas) == len(doc_ids)):
            raise ValueError("All input lists must have the same length.")

        if not embeddings:
            return  # Nothing to insert

        # Extract documents from metadata
        documents = [metadata["content_cid"] for metadata in metadatas]

        # Batch insert all documents into the database
        collection = self.db_client.get_collection(name=db_name)
        try:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=doc_ids,
                metadatas=metadatas,
            )
        except Exception as e:
            raise Exception(f"Error batch inserting documents into database '{db_name}': {e}")

    def print_all_metadata(self):
        """
        Retrieves and prints all metadata from every collection.
        """
        for db_name in self.db_names:
            collection = self.db_client.get_collection(name=db_name)
            # Retrieve all entries from the collection.
            # The structure of the returned results is assumed to contain a "metadatas" key.
            results = collection.get()
            metadatas = results.get("metadatas", [])
            print(f"\nMetadata for collection '{db_name}':")
            if not metadatas:
                print("  No metadata found.")
            else:
                for metadata in metadatas:
                    print(metadata)
