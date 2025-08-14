"""
Neo4j graph database manager.

This module provides a Neo4jGraph class for managing graph database connections
and operations related to IPFS content identifiers (CIDs).
"""

import os
from typing import List, Tuple, Optional

import certifi
from neo4j import GraphDatabase

from src.utils.logging_utils import get_logger

# Get module logger
logger = get_logger(__name__)


class IPFSNeo4jGraph:
    """
    Manages Neo4j graph database connections and operations for IPFS CIDs.

    This class handles the creation, querying, and traversal of graph relationships
    between IPFS content identifiers in a Neo4j database.
    """

    def __init__(self, uri=None, username=None, password=None):
        """
        Initialize the Neo4j graph connection with URI and credentials.

        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password

        Raises:
            ValueError: If URI, username, or password is missing
        """
        # Use environment variables as fallback
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")

        self.logger = get_logger(__name__ + ".IPFSNeo4jGraph")

        if not all([self.uri, self.username, self.password]):
            missing = []
            if not self.uri:
                missing.append("NEO4J_URI")
            if not self.username:
                missing.append("NEO4J_USERNAME")
            if not self.password:
                missing.append("NEO4J_PASSWORD")

            error_msg = f"Missing Neo4j connection parameters: {', '.join(missing)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            os.environ["SSL_CERT_FILE"] = certifi.where()

            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password), encrypted=False
            )
            self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the database connection."""
        self.driver.close()
        self.logger.info("Database connection closed.")

    def add_ipfs_node(self, cid):
        """Add an IPFS CID as a node in Neo4j."""
        try:
            with self.driver.session() as session:
                session.run("MERGE (:IPFS {cid: $cid})", cid=cid)
                self.logger.info(f"Node added: {cid}")
        except Exception as e:
            self.logger.error(f"Failed to add node {cid}: {e}")

    def add_ipfs_nodes_batch(self, cids: List[str]):
        """Add multiple IPFS CIDs as nodes in Neo4j in a single transaction."""
        if not cids:
            return

        try:
            with self.driver.session() as session:
                # Use UNWIND to process multiple CIDs in one query
                query = """
                    UNWIND $cids AS cid
                    MERGE (:IPFS {cid: cid})
                """
                session.run(query, cids=cids)
                self.logger.info(f"Batch added {len(cids)} nodes")
        except Exception as e:
            self.logger.error(f"Failed to batch add nodes: {e}")

    def create_relationship(self, cid1, cid2, relationship_type="LINKS_TO"):
        """Create a relationship between two IPFS nodes (avoiding Cartesian Product)."""
        try:
            with self.driver.session() as session:
                query = f"""
                    MERGE (a:IPFS {{cid: $cid1}})
                    MERGE (b:IPFS {{cid: $cid2}})
                    MERGE (a)-[:{relationship_type}]->(b)
                """
                session.run(query, cid1=cid1, cid2=cid2)
                self.logger.info(
                    f"Relationship created: {cid1} - [{relationship_type}] -> {cid2}"
                )
        except Exception as e:
            self.logger.error(
                f"Failed to create relationship {cid1} - [{relationship_type}] -> {cid2}: {e}"
            )

    def create_relationships_batch(self, relationships: List[Tuple[str, str, str]]):
        """
        Create multiple relationships in a single transaction.

        Args:
            relationships: List of tuples (source_cid, target_cid, relationship_type)
        """
        if not relationships:
            return

        # Group relationships by type for more efficient querying
        relationships_by_type = {}
        for source, target, rel_type in relationships:
            if rel_type not in relationships_by_type:
                relationships_by_type[rel_type] = []
            relationships_by_type[rel_type].append((source, target))

        try:
            with self.driver.session() as session:
                for rel_type, pairs in relationships_by_type.items():
                    # Use UNWIND to process multiple relationships of the same type
                    query = f"""
                        UNWIND $pairs AS pair
                        MERGE (a:IPFS {{cid: pair[0]}})
                        MERGE (b:IPFS {{cid: pair[1]}})
                        MERGE (a)-[:{rel_type}]->(b)
                    """
                    session.run(query, pairs=pairs)
                    self.logger.info(
                        f"Batch created {len(pairs)} relationships of type {rel_type}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to batch create relationships: {e}")

    def query_graph(self):
        """Retrieve all nodes and relationships."""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (a)-[r]->(b) RETURN a.cid, TYPE(r), b.cid")
                for record in result:
                    self.logger.info(
                        f"{record['a.cid']} -[{record['TYPE(r)']}]→ {record['b.cid']}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to query graph: {e}")

    def get_converted_markdown_cid(self, cid, markdown_conversion):
        """
        Check if a CID has already been converted using a specific markdown conversion
        and return the resulting CID if it exists.

        Args:
            cid: The original IPFS CID to check
            markdown_conversion: The name of the markdown conversion type

        Returns:
            The CID of the converted markdown if found, None otherwise
        """
        relationship_type = f"CONVERTED_BY_{markdown_conversion}"
        try:
            with self.driver.session() as session:
                query = f"""
                    MATCH (source:IPFS {{cid: $cid}})-[:{relationship_type}]->(converted:IPFS)
                    RETURN converted.cid AS converted_cid
                """
                result = session.run(query, cid=cid)
                record = result.single()

                if record:
                    converted_cid = record["converted_cid"]
                    self.logger.info(
                        f"Found converted CID {converted_cid} for {cid} using {markdown_conversion}"
                    )
                    return converted_cid

                self.logger.info(
                    f"No conversion found for {cid} with {markdown_conversion}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Failed to check for converted markdown: {e}")
            return None

    def recreate_path(self, start_cid, path):
        """
        Given a starting node and an ordered list of relationship steps,
        return all possible paths with intermediate nodes included.

        :param start_cid: The CID of the starting node.
        :param path: Ordered list of relationship types to follow.
        :return: List of lists, where each list represents a path including intermediate nodes.
        """
        if not path:
            return [[start_cid]]

        try:
            with self.driver.session() as session:
                query = "MATCH (start:IPFS {cid: $start_cid})"
                path_return = ["start.cid"]

                for i, rel in enumerate(path):
                    query += f"-[:{rel}]->(n{i}:IPFS)"
                    path_return.append(f"n{i}.cid")

                query += f" RETURN {', '.join(path_return)}"

                result = session.run(query, start_cid=start_cid)
                paths = [[record[key] for key in path_return] for record in result]

                return paths if paths else False
        except Exception as e:
            self.logger.error(
                f"Failed to traverse path from {start_cid} with path {path}: {e}"
            )
            return False

    def traverse_path_end_nodes(self, start_cid, path):
        """
        Given a starting node and an ordered list of relationship steps,
        return all node CIDs that exist at the end of the given path.

        :param start_cid: The CID of the starting node.
        :param path: Ordered list of relationship types to follow.
        :return: List of node CIDs at the end of the path if exists, else False.
        """
        if not path:
            return [start_cid]

        try:
            with self.driver.session() as session:
                query = "MATCH (start:IPFS {cid: $start_cid})"
                for i, rel in enumerate(path):
                    query += f"-[:{rel}]->(n{i}:IPFS)"
                query += f" RETURN n{len(path)-1}.cid AS end_cid"

                result = session.run(query, start_cid=start_cid)
                end_nodes = [record["end_cid"] for record in result]

                return end_nodes if end_nodes else False
        except Exception as e:
            self.logger.error(
                f"Failed to traverse path from {start_cid} with path {path}: {e}"
            )
            return False

    def get_existing_metadata_cid(self, pdf_cid: str) -> Optional[str]:
        """
        Check if metadata already exists for the given PDF CID.

        Args:
            pdf_cid: The PDF CID to check for existing metadata

        Returns:
            The metadata CID if it exists, None otherwise
        """
        try:
            with self.driver.session() as session:
                query = """
                    MATCH (pdf:IPFS {cid: $pdf_cid})-[:HAS_METADATA]->(metadata:IPFS)
                    RETURN metadata.cid as metadata_cid
                """
                result = session.run(query, pdf_cid=pdf_cid)
                record = result.single()
                if record:
                    metadata_cid = record["metadata_cid"]
                    self.logger.info(f"Found existing metadata CID: {metadata_cid}")
                    return metadata_cid
                return None
        except Exception as e:
            self.logger.error(f"Error checking for existing metadata: {e}")
            return None

    def create_metadata_node(self, pdf_cid: str, metadata_cid: str) -> bool:
        """
        Create a metadata node and link it to the PDF node.

        Args:
            pdf_cid: The PDF CID to link metadata to
            metadata_cid: The metadata CID to create and link

        Returns:
            True if successful, False otherwise
        """
        try:
            # Add metadata node to graph
            self.add_ipfs_node(metadata_cid)

            # Create relationship between PDF and metadata
            self.create_relationship(pdf_cid, metadata_cid, "HAS_METADATA")

            self.logger.info(
                f"Created metadata node {metadata_cid} linked to PDF {pdf_cid}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error creating metadata node: {e}")
            return False

    @staticmethod
    def default_metadata():
        """
        Returns default metadata in case extraction fails.

        Returns:
            A dictionary with default metadata values
        """
        return {
            "title": "Unknown Title",
            "authors": ["Unknown Authors"],
            "categories": ["Unknown Categories"],
            "doi": "No DOI available",
            "keywords": ["Unknown Keywords"],
            "publication_date": "Unknown Date",
            "journal": "Unknown Journal",
            "citation": "Unknown Authors. (Unknown Date). Unknown Title. Unknown Journal. No DOI available",
            "pdf_filename": "Unknown Filename",
        }
