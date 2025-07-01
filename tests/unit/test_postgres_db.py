"""Tests for PostgreSQL database manager in src."""

from unittest.mock import MagicMock, patch

import pytest

from src.db.postgres_db import PostgresDBManager


class TestPostgresDBManager:
    """Test suite for PostgresDBManager class."""

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Set up mock environment variables for testing."""
        monkeypatch.setenv("POSTGRES_HOST", "test_host")
        monkeypatch.setenv("POSTGRES_PORT", "5432")
        monkeypatch.setenv("POSTGRES_USER", "test_user")
        monkeypatch.setenv("POSTGRES_PASSWORD", "test_password")
        return {
            "POSTGRES_HOST": "test_host",
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_password",
        }

    def test_init_with_provided_params(self):
        """Test initialization with provided parameters."""
        with patch("src.postgres_db.psycopg2.connect") as mock_connect:
            # Configure connect mock
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            # Initialize PostgresDBManager
            manager = PostgresDBManager(
                host="custom_host",
                port="1234",
                user="custom_user",
                password="custom_password",
            )

            # Verify connection parameters
            mock_connect.assert_called_once_with(
                host="custom_host",
                port="1234",
                user="custom_user",
                password="custom_password",
                dbname="postgres",
            )
            assert mock_conn.autocommit is True

            # Verify attribute assignment
            assert manager.host == "custom_host"
            assert manager.port == "1234"
            assert manager.user == "custom_user"
            assert manager.password == "custom_password"

    def test_init_with_env_vars(self, mock_env_vars):
        """Test initialization with environment variables."""
        with patch("src.postgres_db.psycopg2.connect") as mock_connect:
            with patch("src.postgres_db.os.getenv") as mock_getenv:
                # Configure getenv mock
                def getenv_side_effect(key, default=None):
                    return mock_env_vars.get(key, default)

                mock_getenv.side_effect = getenv_side_effect

                # Configure connect mock
                mock_conn = MagicMock()
                mock_connect.return_value = mock_conn

                # Initialize PostgresDBManager
                manager = PostgresDBManager()

                # Verify connection parameters
                mock_connect.assert_called_once_with(
                    host=mock_env_vars["POSTGRES_HOST"],
                    port=mock_env_vars["POSTGRES_PORT"],
                    user=mock_env_vars["POSTGRES_USER"],
                    password=mock_env_vars["POSTGRES_PASSWORD"],
                    dbname="postgres",
                )
                assert mock_conn.autocommit is True

                # Verify attribute assignment
                assert manager.host == mock_env_vars["POSTGRES_HOST"]
                assert manager.port == mock_env_vars["POSTGRES_PORT"]
                assert manager.user == mock_env_vars["POSTGRES_USER"]
                assert manager.password == mock_env_vars["POSTGRES_PASSWORD"]

    def test_init_connection_failure(self):
        """Test handling of connection failure during initialization."""
        with patch("src.postgres_db.psycopg2.connect") as mock_connect:
            # Configure connect mock to raise an exception
            mock_connect.side_effect = Exception("Connection failed")

            # Verify the exception is propagated
            with pytest.raises(Exception) as excinfo:
                PostgresDBManager()

            assert "Connection failed" in str(excinfo.value)

    def test_connect_success(self):
        """Test successful database connection."""
        with patch("src.postgres_db.psycopg2.connect") as mock_connect:
            # Configure primary connection mock
            mock_primary_conn = MagicMock()
            mock_connect.return_value = mock_primary_conn

            # Initialize PostgresDBManager
            manager = PostgresDBManager()

            # Reset mock for testing _connect
            mock_connect.reset_mock()

            # Configure new connection mock
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            # Test _connect method
            result = manager._connect("test_db")

            # Verify connection
            mock_connect.assert_called_once_with(
                host=manager.host,
                port=manager.port,
                user=manager.user,
                password=manager.password,
                dbname="test_db",
            )
            assert result == mock_conn
            assert mock_conn.autocommit is True

    def test_connect_failure(self):
        """Test handling of connection failure in _connect method."""
        with patch("src.postgres_db.psycopg2.connect") as mock_connect:
            # Configure primary connection mock
            mock_primary_conn = MagicMock()
            mock_connect.return_value = mock_primary_conn

            # Initialize PostgresDBManager
            manager = PostgresDBManager()

            # Configure connect mock to raise an exception
            mock_connect.side_effect = Exception("Connection failed")

            # Test _connect method
            with patch.object(manager, "logger") as mock_logger:
                result = manager._connect("test_db")

                # Verify result and logging
                assert result is None
                mock_logger.error.assert_called_once()
                assert "Connection failed" in str(mock_logger.error.call_args[0][0])

    def test_create_databases(self):
        """Test creating multiple databases."""
        with patch("src.postgres_db.psycopg2.connect") as mock_connect:
            # Configure primary connection mock
            mock_primary_conn = MagicMock()
            mock_connect.return_value = mock_primary_conn

            # Initialize PostgresDBManager
            manager = PostgresDBManager()

            # Patch _connect and _create_schema_and_table_in_db
            with patch.object(manager, "_connect") as mock_connect_method:
                with patch.object(
                    manager, "_create_schema_and_table_in_db"
                ) as mock_create_schema:
                    # Configure mock connection and cursor
                    mock_conn = MagicMock()
                    mock_cursor = MagicMock()
                    mock_connect_method.return_value = mock_conn
                    mock_conn.cursor.return_value = mock_cursor

                    # Configure cursor.fetchone to alternate (existing and non-existing DBs)
                    mock_cursor.fetchone.side_effect = [None, (1,)]

                    # Call create_databases
                    db_names = ["db1", "db2"]
                    manager.create_databases(db_names)

                    # Verify database existence checks
                    assert mock_cursor.execute.call_count >= 2

                    # Verify _create_schema_and_table_in_db was called for db1 (non-existing)
                    mock_create_schema.assert_called_once_with("db1")

                    # Verify cursor and connection were closed
                    mock_cursor.close.assert_called_once()
                    mock_conn.close.assert_called_once()

    def test_create_schema_and_table_in_db(self):
        """Test creating schema and table in a database."""
        with patch("src.postgres_db.psycopg2.connect") as mock_connect:
            # Configure primary connection mock
            mock_primary_conn = MagicMock()
            mock_connect.return_value = mock_primary_conn

            # Initialize PostgresDBManager
            manager = PostgresDBManager()

            # Patch _connect
            with patch.object(manager, "_connect") as mock_connect_method:
                # Configure mock connection and cursor
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_connect_method.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor

                # Call _create_schema_and_table_in_db
                manager._create_schema_and_table_in_db("test_db")

                # Verify schema creation
                assert any(
                    "CREATE SCHEMA IF NOT EXISTS default_schema" in str(call)
                    for call in mock_cursor.execute.call_args_list
                )

                # Verify table creation
                assert any(
                    "CREATE TABLE IF NOT EXISTS default_schema.papers" in str(call)
                    for call in mock_cursor.execute.call_args_list
                )

                # Verify cursor and connection were closed
                mock_cursor.close.assert_called_once()
                mock_conn.close.assert_called_once()

    def test_insert_data(self):
        """Test inserting data into a database."""
        with patch("src.postgres_db.psycopg2.connect") as mock_connect:
            # Configure primary connection mock
            mock_primary_conn = MagicMock()
            mock_connect.return_value = mock_primary_conn

            # Initialize PostgresDBManager
            manager = PostgresDBManager()

            # Patch _connect
            with patch.object(manager, "_connect") as mock_connect_method:
                # Configure mock connection and cursor
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_connect_method.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor

                # Create test data
                test_data = [
                    (
                        "author1",
                        "paper1",
                        "markdown1",
                        [0.1, 0.2, 0.3],
                        {"key": "value"},
                        True,
                    ),
                    (
                        "author2",
                        "paper2",
                        "markdown2",
                        [0.4, 0.5, 0.6],
                        {"key": "value2"},
                        False,
                    ),
                ]

                # Call insert_data
                manager.insert_data("test_db", test_data)

                # Verify cursor execution count
                assert mock_cursor.execute.call_count == 2  # One for each record

                # Verify pickle.dumps was used for embedding
                with patch("src.postgres_db.pickle.dumps") as mock_dumps:
                    with patch("src.postgres_db.np.array") as mock_array:
                        # Configure mocks
                        mock_dumps.return_value = b"binary_data"
                        mock_array.return_value = "numpy_array"

                        # Call insert_data again
                        manager.insert_data("test_db", test_data)

                        # Verify pickle.dumps was called for each record
                        assert mock_dumps.call_count == 2
                        assert mock_array.call_count == 2

                # Verify cursor was closed
                mock_cursor.close.assert_called()

    def test_query_select(self):
        """Test executing a SELECT query."""
        with patch("src.postgres_db.psycopg2.connect") as mock_connect:
            # Configure primary connection mock
            mock_primary_conn = MagicMock()
            mock_connect.return_value = mock_primary_conn

            # Initialize PostgresDBManager
            manager = PostgresDBManager()

            # Patch _connect
            with patch.object(manager, "_connect") as mock_connect_method:
                # Configure mock connection and cursor
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_connect_method.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor

                # Configure mock fetchall result
                expected_result = [("row1",), ("row2",)]
                mock_cursor.fetchall.return_value = expected_result

                # Call query with SELECT
                query_string = "SELECT * FROM test_table WHERE id = %s"
                params = (1,)
                result = manager.query("test_db", query_string, params)

                # Verify cursor execution
                mock_cursor.execute.assert_called_once_with(query_string, params)
                mock_cursor.fetchall.assert_called_once()

                # Verify result
                assert result == expected_result

                # Verify cursor and connection were closed
                mock_cursor.close.assert_called_once()
                mock_conn.close.assert_called_once()

    def test_query_non_select(self):
        """Test executing a non-SELECT query."""
        with patch("src.postgres_db.psycopg2.connect") as mock_connect:
            # Configure primary connection mock
            mock_primary_conn = MagicMock()
            mock_connect.return_value = mock_primary_conn

            # Initialize PostgresDBManager
            manager = PostgresDBManager()

            # Patch _connect
            with patch.object(manager, "_connect") as mock_connect_method:
                # Configure mock connection and cursor
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_connect_method.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor

                # Call query with INSERT
                query_string = "INSERT INTO test_table (col) VALUES (%s)"
                params = ("value",)
                result = manager.query("test_db", query_string, params)

                # Verify cursor execution
                mock_cursor.execute.assert_called_once_with(query_string, params)
                mock_conn.commit.assert_called_once()

                # Verify result is None
                assert result is None

                # Verify cursor and connection were closed
                mock_cursor.close.assert_called_once()
                mock_conn.close.assert_called_once()

    def test_query_exception(self):
        """Test handling exceptions during query execution."""
        with patch("src.postgres_db.psycopg2.connect") as mock_connect:
            # Configure primary connection mock
            mock_primary_conn = MagicMock()
            mock_connect.return_value = mock_primary_conn

            # Initialize PostgresDBManager
            manager = PostgresDBManager()

            # Patch _connect
            with patch.object(manager, "_connect") as mock_connect_method:
                # Configure mock connection and cursor
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_connect_method.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor

                # Configure cursor to raise an exception
                mock_cursor.execute.side_effect = Exception("Query failed")

                # Call query
                with patch.object(manager, "logger") as mock_logger:
                    result = manager.query("test_db", "SELECT * FROM test")

                    # Verify error was logged
                    mock_logger.error.assert_called_once()
                    assert "Query failed" in str(mock_logger.error.call_args[0][0])

                    # Verify result is None
                    assert result is None

                    # Verify cursor and connection were closed
                    mock_cursor.close.assert_called_once()
                    mock_conn.close.assert_called_once()
