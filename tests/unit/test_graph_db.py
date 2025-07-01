"""Tests for Neo4j graph database manager in src."""

from unittest.mock import MagicMock, call, patch

import pytest
import requests

from src.db.graph_db import IPFSNeo4jGraph


class TestIPFSNeo4jGraph:
    """Test suite for IPFSNeo4jGraph class."""

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Set up mock environment variables for testing."""
        monkeypatch.setenv("NEO4J_URI", "neo4j://testhost:7687")
        monkeypatch.setenv("NEO4J_USERNAME", "test_user")
        monkeypatch.setenv("NEO4J_PASSWORD", "test_password")
        return {
            "NEO4J_URI": "neo4j://testhost:7687",
            "NEO4J_USERNAME": "test_user",
            "NEO4J_PASSWORD": "test_password",
        }

    @pytest.fixture
    def mock_driver(self):
        """Set up mock Neo4j driver."""
        with patch("neo4j.GraphDatabase.driver") as mock:
            driver_instance = mock.return_value
            driver_instance.verify_connectivity = MagicMock()
            driver_instance.session = MagicMock()
            driver_instance.close = MagicMock()
            yield mock

    def test_init_with_provided_credentials(self, mock_driver):
        """Test initialization with provided credentials."""
        uri = "neo4j://localhost:7687"
        username = "neo4j"
        password = "password"

        with patch("os.environ"):
            with patch("certifi.where", return_value="/path/to/certifi"):
                graph_instance = IPFSNeo4jGraph(
                    uri=uri, username=username, password=password
                )

                # Assert the graph instance was created successfully
                assert graph_instance is not None

                # Verify driver was initialized with correct parameters
                mock_driver.assert_called_once_with(
                    uri, auth=(username, password), encrypted=True
                )
                driver_instance = mock_driver.return_value
                driver_instance.verify_connectivity.assert_called_once()

    def test_init_with_env_vars(self, mock_env_vars, mock_driver):
        """Test initialization using environment variables."""
        with patch("certifi.where", return_value="/path/to/certifi"):
            # Instead of patching os.environ which might affect other code,
            # we'll use getenv to return our mock values
            with patch("os.getenv") as mock_getenv:

                def getenv_side_effect(key, default=None):
                    return mock_env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                graph_instance = IPFSNeo4jGraph()

                # Assert the graph instance was created successfully
                assert graph_instance is not None

                # Verify driver was initialized with environment variables
                mock_driver.assert_called_once_with(
                    mock_env_vars["NEO4J_URI"],
                    auth=(
                        mock_env_vars["NEO4J_USERNAME"],
                        mock_env_vars["NEO4J_PASSWORD"],
                    ),
                    encrypted=True,
                )

    def test_init_missing_credentials(self):
        """Test initialization with missing credentials raises ValueError."""
        with patch("os.environ", {}):
            with patch("os.getenv", return_value=None):
                with pytest.raises(ValueError) as excinfo:
                    IPFSNeo4jGraph()

                assert "Missing Neo4j connection parameters" in str(excinfo.value)

    def test_connection_error(self, mock_env_vars):
        """Test that connection errors are properly handled."""
        with patch("neo4j.GraphDatabase.driver") as mock_driver:
            with patch("certifi.where", return_value="/path/to/certifi"):
                with patch("os.getenv") as mock_getenv:

                    def getenv_side_effect(key, default=None):
                        return mock_env_vars.get(key, default)

                    mock_getenv.side_effect = getenv_side_effect

                    # Simulate connection error
                    driver_instance = mock_driver.return_value
                    driver_instance.verify_connectivity.side_effect = Exception(
                        "Connection failed"
                    )

                    with pytest.raises(Exception) as excinfo:
                        IPFSNeo4jGraph()

                    assert "Connection failed" in str(excinfo.value)

    def test_close(self, mock_env_vars, mock_driver):
        """Test that close method closes the driver connection."""
        with patch("certifi.where", return_value="/path/to/certifi"):
            with patch("os.getenv") as mock_getenv:

                def getenv_side_effect(key, default=None):
                    return mock_env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                graph = IPFSNeo4jGraph()
                graph.close()

                # Verify driver.close() was called
                driver_instance = mock_driver.return_value
                driver_instance.close.assert_called_once()

    def test_add_ipfs_node(self, mock_env_vars, mock_driver):
        """Test adding an IPFS node to the graph."""
        cid = "QmTest123"
        session_mock = MagicMock()

        with patch("certifi.where", return_value="/path/to/certifi"):
            with patch("os.getenv") as mock_getenv:

                def getenv_side_effect(key, default=None):
                    return mock_env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                # Setup the session mock
                driver_instance = mock_driver.return_value
                driver_instance.session.return_value.__enter__.return_value = (
                    session_mock
                )

                # Create graph and add node
                graph = IPFSNeo4jGraph()
                graph.add_ipfs_node(cid)

                # Verify session.run was called with correct query
                session_mock.run.assert_called_once_with(
                    "MERGE (:IPFS {cid: $cid})", cid=cid
                )

    def test_create_relationship(self, mock_env_vars, mock_driver):
        """Test creating a relationship between two IPFS nodes."""
        cid1 = "QmTest123"
        cid2 = "QmTest456"
        relationship_type = "TEST_LINK"
        session_mock = MagicMock()

        with patch("certifi.where", return_value="/path/to/certifi"):
            with patch("os.getenv") as mock_getenv:

                def getenv_side_effect(key, default=None):
                    return mock_env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                # Setup the session mock
                driver_instance = mock_driver.return_value
                driver_instance.session.return_value.__enter__.return_value = (
                    session_mock
                )

                # Create graph and add relationship
                graph = IPFSNeo4jGraph()
                graph.create_relationship(cid1, cid2, relationship_type)

                # Verify session.run was called with correct query
                session_mock.run.assert_called_once()
                # Check that the query includes the relationship type and parameters
                args, kwargs = session_mock.run.call_args
                assert relationship_type in args[0]
                assert kwargs["cid1"] == cid1
                assert kwargs["cid2"] == cid2

    def test_query_graph(self, mock_env_vars, mock_driver):
        """Test querying all nodes and relationships in the graph."""
        mock_records = [
            {"a.cid": "QmTest1", "TYPE(r)": "LINKS_TO", "b.cid": "QmTest2"},
            {"a.cid": "QmTest1", "TYPE(r)": "AUTHORED_BY", "b.cid": "QmTest3"},
        ]
        session_mock = MagicMock()
        session_mock.run.return_value = mock_records

        with patch("certifi.where", return_value="/path/to/certifi"):
            with patch("os.getenv") as mock_getenv:

                def getenv_side_effect(key, default=None):
                    return mock_env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                # Setup the session mock
                driver_instance = mock_driver.return_value
                driver_instance.session.return_value.__enter__.return_value = (
                    session_mock
                )

                # Create graph and query
                graph = IPFSNeo4jGraph()
                with patch.object(graph, "logger") as mock_logger:
                    graph.query_graph()

                    # Verify session.run was called with correct query
                    session_mock.run.assert_called_once_with(
                        "MATCH (a)-[r]->(b) RETURN a.cid, TYPE(r), b.cid"
                    )

                    # Verify logger.info was called for each record
                    assert mock_logger.info.call_count == 2
                    for record in mock_records:
                        mock_logger.info.assert_any_call(
                            f"{record['a.cid']} -[{record['TYPE(r)']}]→ {record['b.cid']}"
                        )

    def test_recreate_path(self, mock_env_vars, mock_driver):
        """Test recreating a path from a starting node."""
        start_cid = "QmTest1"
        path = ["LINKS_TO", "AUTHORED_BY"]
        expected_result = [["QmTest1", "QmTest2", "QmTest3"]]

        session_mock = MagicMock()
        session_mock.run.return_value = [
            {"start.cid": "QmTest1", "n0.cid": "QmTest2", "n1.cid": "QmTest3"}
        ]

        with patch("certifi.where", return_value="/path/to/certifi"):
            with patch("os.getenv") as mock_getenv:

                def getenv_side_effect(key, default=None):
                    return mock_env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                # Setup the session mock
                driver_instance = mock_driver.return_value
                driver_instance.session.return_value.__enter__.return_value = (
                    session_mock
                )

                # Create graph and recreate path
                graph = IPFSNeo4jGraph()
                result = graph.recreate_path(start_cid, path)

                # Verify session.run was called with correct query and parameters
                session_mock.run.assert_called_once()
                args, kwargs = session_mock.run.call_args
                assert "MATCH (start:IPFS {cid: $start_cid})" in args[0]
                assert "-[:LINKS_TO]->" in args[0]
                assert "-[:AUTHORED_BY]->" in args[0]
                assert kwargs["start_cid"] == start_cid

                # Verify result
                assert result == expected_result

    def test_recreate_path_empty(self, mock_env_vars, mock_driver):
        """Test recreating a path with an empty path list."""
        start_cid = "QmTest1"
        path = []

        with patch("certifi.where", return_value="/path/to/certifi"):
            with patch("os.getenv") as mock_getenv:

                def getenv_side_effect(key, default=None):
                    return mock_env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                # Create graph and recreate path
                graph = IPFSNeo4jGraph()
                result = graph.recreate_path(start_cid, path)

                # Verify result is a list containing just the start CID
                assert result == [[start_cid]]

                # Verify session.run was not called
                driver_instance = mock_driver.return_value
                assert not driver_instance.session.called

    def test_traverse_path_end_nodes(self, mock_env_vars, mock_driver):
        """Test traversing a path and returning end nodes."""
        start_cid = "QmTest1"
        path = ["LINKS_TO", "AUTHORED_BY"]
        expected_result = ["QmTest3", "QmTest4"]

        session_mock = MagicMock()
        session_mock.run.return_value = [{"end_cid": "QmTest3"}, {"end_cid": "QmTest4"}]

        with patch("certifi.where", return_value="/path/to/certifi"):
            with patch("os.getenv") as mock_getenv:

                def getenv_side_effect(key, default=None):
                    return mock_env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                # Setup the session mock
                driver_instance = mock_driver.return_value
                driver_instance.session.return_value.__enter__.return_value = (
                    session_mock
                )

                # Create graph and traverse path
                graph = IPFSNeo4jGraph()
                result = graph.traverse_path_end_nodes(start_cid, path)

                # Verify session.run was called with correct query and parameters
                session_mock.run.assert_called_once()
                args, kwargs = session_mock.run.call_args
                assert "MATCH (start:IPFS {cid: $start_cid})" in args[0]
                assert "-[:LINKS_TO]->" in args[0]
                assert "-[:AUTHORED_BY]->" in args[0]
                assert kwargs["start_cid"] == start_cid

                # Verify result
                assert result == expected_result

    def test_query_ipfs_content(self, mock_env_vars, mock_driver):
        """Test querying IPFS content for a given CID."""
        cid = "QmTest123"
        expected_content = "test content"

        with patch("certifi.where", return_value="/path/to/certifi"):
            with patch("os.getenv") as mock_getenv:
                with patch("requests.get") as mock_get:

                    def getenv_side_effect(key, default=None):
                        return mock_env_vars.get(key, default)

                    mock_getenv.side_effect = getenv_side_effect

                    # Setup the response mock
                    mock_response = MagicMock()
                    mock_response.text = (
                        f"{expected_content}  "  # Add spaces to test stripping
                    )
                    mock_response.raise_for_status = MagicMock()
                    mock_get.return_value = mock_response

                    # Create graph and query IPFS content
                    graph = IPFSNeo4jGraph()
                    result = graph._query_ipfs_content(cid)

                    # Verify requests.get was called with correct URL
                    mock_get.assert_called_once_with(
                        f"https://gateway.lighthouse.storage/ipfs/{cid}"
                    )
                    mock_response.raise_for_status.assert_called_once()

                    # Verify result is stripped
                    assert result == expected_content

    def test_query_ipfs_content_error(self, mock_env_vars, mock_driver):
        """Test error handling when querying IPFS content."""
        cid = "QmTest123"

        with patch("certifi.where", return_value="/path/to/certifi"):
            with patch("os.getenv") as mock_getenv:
                with patch("requests.get") as mock_get:

                    def getenv_side_effect(key, default=None):
                        return mock_env_vars.get(key, default)

                    mock_getenv.side_effect = getenv_side_effect

                    # Setup the response mock to raise an exception
                    mock_get.side_effect = requests.exceptions.RequestException(
                        "Failed to retrieve content"
                    )

                    # Create graph and query IPFS content
                    graph = IPFSNeo4jGraph()
                    with patch.object(graph, "logger") as mock_logger:
                        result = graph._query_ipfs_content(cid)

                        # Verify logger.error was called
                        mock_logger.error.assert_called_once()

                        # Verify result is None
                        assert result is None

    def test_get_authored_by_stats(self, mock_env_vars, mock_driver):
        """Test getting author statistics."""
        mock_records = [
            {"authored_by_cid": "QmAuthor1", "incoming_count": 5},
            {"authored_by_cid": "QmAuthor2", "incoming_count": 3},
        ]
        author_contents = {
            "QmAuthor1": "author1@example.com",
            "QmAuthor2": "author2@example.com",
        }

        session_mock = MagicMock()
        session_mock.run.return_value = mock_records

        with patch("certifi.where", return_value="/path/to/certifi"):
            with patch("os.getenv") as mock_getenv:

                def getenv_side_effect(key, default=None):
                    return mock_env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                # Setup the session mock
                driver_instance = mock_driver.return_value
                driver_instance.session.return_value.__enter__.return_value = (
                    session_mock
                )

                # Create graph
                graph = IPFSNeo4jGraph()

                # Mock _query_ipfs_content to return author email
                with patch.object(graph, "_query_ipfs_content") as mock_query:

                    def side_effect(cid):
                        return author_contents.get(cid)

                    mock_query.side_effect = side_effect

                    # Get author stats
                    result = graph.get_authored_by_stats()

                    # Verify session.run was called with correct query
                    session_mock.run.assert_called_once()

                    # Verify _query_ipfs_content was called for each author CID
                    assert mock_query.call_count == 2
                    mock_query.assert_has_calls(
                        [call("QmAuthor1"), call("QmAuthor2")], any_order=True
                    )

                    # Verify result
                    expected_result = {
                        "author1@example.com": 5,
                        "author2@example.com": 3,
                    }
                    assert result == expected_result
