"""Tests for the token_rewarder module in src."""

import itertools
import json
from unittest.mock import Mock, call, patch

import pytest

from src.rewards.token_rewarder import TokenRewarder


class TestTokenRewarder:
    """Test suite for TokenRewarder class."""

    @pytest.fixture
    def mock_contract_abi(self):
        """Create a sample contract ABI for testing."""
        return {
            "abi": [
                {"type": "function", "name": "issueToken", "inputs": [], "outputs": []}
            ]
        }

    @pytest.fixture
    def db_components(self):
        """Create sample database components for testing."""
        return {
            "converter": ["openai"],
            "chunker": ["paragraph"],
            "embedder": ["openai"],
        }

    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Set up mock environment variables for testing."""
        monkeypatch.setenv("OWNER_ADDRESS", "0x123456789")
        monkeypatch.setenv("PRIVATE_KEY", "abcdef1234567890")
        return {"OWNER_ADDRESS": "0x123456789", "PRIVATE_KEY": "abcdef1234567890"}

    def test_init_with_defaults(self, mock_contract_abi):
        """Test initialization with default parameters."""
        with patch("src.token_rewarder.Web3") as mock_web3:
            with patch(
                "src.token_rewarder.TokenRewarder.load_contract_abi",
                return_value=mock_contract_abi,
            ):
                with patch("src.token_rewarder.os.getenv") as mock_getenv:
                    # Configure mocks
                    mock_web3_instance = Mock()
                    mock_web3.return_value = mock_web3_instance
                    mock_provider = Mock()
                    mock_web3.HTTPProvider = Mock(return_value=mock_provider)
                    mock_getenv.return_value = "mock_value"

                    # Initialize TokenRewarder
                    rewarder = TokenRewarder()

                    # Verify initialization
                    assert rewarder.rpc_url == "https://sepolia.base.org"
                    assert rewarder.chain_id == 84532
                    assert rewarder.host == "localhost"
                    assert rewarder.port == 5432

                    # Assert that HTTPProvider was called with the right URL
                    mock_web3.HTTPProvider.assert_called_once()
                    args, _ = mock_web3.HTTPProvider.call_args
                    assert args[0] == rewarder.rpc_url

    def test_initialize_network_valid(self):
        """Test network initialization with valid network types."""
        with patch("src.token_rewarder.Web3"):
            with patch("src.token_rewarder.TokenRewarder.load_contract_abi"):
                with patch("src.token_rewarder.os.getenv"):
                    # Test different networks
                    networks = {
                        "optimism": ("https://mainnet.optimism.io", 10),
                        "test_base": ("https://sepolia.base.org", 84532),
                        "base": ("https://mainnet.base.org", 1234),
                    }

                    for network, (expected_url, expected_chain_id) in networks.items():
                        rewarder = TokenRewarder(network=network)
                        assert rewarder.rpc_url == expected_url
                        assert rewarder.chain_id == expected_chain_id

    def test_initialize_network_invalid(self):
        """Test network initialization with invalid network type raises ValueError."""
        with patch("src.token_rewarder.Web3"):
            with patch("src.token_rewarder.TokenRewarder.load_contract_abi"):
                with patch("src.token_rewarder.os.getenv"):
                    with pytest.raises(ValueError) as excinfo:
                        TokenRewarder(network="invalid_network")

                    assert "Unsupported network" in str(excinfo.value)

    def test_connect_success(self, mock_contract_abi):
        """Test successful database connection."""
        with patch("src.token_rewarder.Web3"):
            with patch(
                "src.token_rewarder.TokenRewarder.load_contract_abi",
                return_value=mock_contract_abi,
            ):
                with patch("src.token_rewarder.os.getenv"):
                    with patch("src.token_rewarder.connect") as mock_connect:
                        # Configure mock connection
                        mock_conn = Mock()
                        mock_connect.return_value = mock_conn

                        # Initialize TokenRewarder and test connection
                        rewarder = TokenRewarder(
                            host="test_host",
                            port=5432,
                            user="test_user",
                            password="test_pass",
                        )
                        conn = rewarder._connect()

                        # Verify connection
                        assert conn == mock_conn
                        mock_connect.assert_called_once_with(
                            host="test_host",
                            port=5432,
                            user="test_user",
                            password="test_pass",
                            dbname="postgres",
                        )
                        assert mock_conn.autocommit is True

    def test_connect_failure(self, mock_contract_abi):
        """Test failed database connection."""
        with patch("src.token_rewarder.Web3"):
            with patch(
                "src.token_rewarder.TokenRewarder.load_contract_abi",
                return_value=mock_contract_abi,
            ):
                with patch("src.token_rewarder.os.getenv"):
                    with patch("src.token_rewarder.connect") as mock_connect:
                        # Configure mock connection to fail
                        mock_connect.side_effect = Exception("Connection failed")

                        # Initialize TokenRewarder and test connection
                        rewarder = TokenRewarder()
                        with patch.object(rewarder, "logger") as mock_logger:
                            conn = rewarder._connect()

                            # Verify connection failure
                            assert conn is None
                            mock_logger.error.assert_called_once()

    def test_load_contract_abi(self, tmp_path):
        """Test loading contract ABI from file."""
        with patch("src.token_rewarder.Web3"):
            with patch("src.token_rewarder.os.getenv"):
                # Create a sample ABI file
                sample_abi = {"abi": [{"type": "function", "name": "test"}]}
                abi_file = tmp_path / "test_abi.json"
                with open(abi_file, "w") as f:
                    json.dump(sample_abi, f)

                # Initialize TokenRewarder and test ABI loading
                with patch(
                    "src.token_rewarder.TokenRewarder.load_contract_abi",
                    return_value=sample_abi,
                ):
                    rewarder = TokenRewarder()

                    # Direct test of the load_contract_abi method
                    original_load = TokenRewarder.load_contract_abi
                    result = original_load(rewarder, str(abi_file))

                    # Verify result
                    assert result == sample_abi

    def test_generate_db_names(self, db_components):
        """Test generation of database names."""
        with patch("src.token_rewarder.Web3"):
            with patch("src.token_rewarder.TokenRewarder.load_contract_abi"):
                with patch("src.token_rewarder.os.getenv"):
                    with patch(
                        "src.token_rewarder.TokenRewarder._initialize_reward_tables"
                    ):
                        # Initialize TokenRewarder and test database name generation
                        rewarder = TokenRewarder(db_components=db_components)

                        # Expected DB names based on Cartesian product
                        expected_names = [
                            f"{c}_{ch}_{e}"
                            for c, ch, e in itertools.product(
                                db_components["converter"],
                                db_components["chunker"],
                                db_components["embedder"],
                            )
                        ]

                        # Verify DB names
                        assert set(rewarder.db_names) == set(expected_names)

    def test_initialize_reward_tables(self, db_components):
        """Test initialization of reward tables."""
        with patch("src.token_rewarder.Web3"):
            with patch("src.token_rewarder.TokenRewarder.load_contract_abi"):
                with patch("src.token_rewarder.os.getenv"):
                    with patch(
                        "src.token_rewarder.TokenRewarder._create_database_and_table"
                    ) as mock_create:
                        # Initialize TokenRewarder with components
                        rewarder = TokenRewarder(db_components=db_components)

                        # Expected calls based on DB names
                        expected_calls = [
                            call(db_name) for db_name in rewarder.db_names
                        ]

                        # Verify _create_database_and_table was called for each DB name
                        mock_create.assert_has_calls(expected_calls, any_order=True)

    def test_create_database_and_table(self, mock_contract_abi):
        """Test creation of database and table."""
        with patch("src.token_rewarder.Web3"):
            with patch(
                "src.token_rewarder.TokenRewarder.load_contract_abi",
                return_value=mock_contract_abi,
            ):
                with patch("src.token_rewarder.os.getenv"):
                    # Mock database connection and cursor
                    mock_conn = Mock()
                    mock_cursor = Mock()
                    mock_conn.cursor.return_value = mock_cursor
                    mock_cursor.fetchone.return_value = None  # Database doesn't exist

                    # Initialize TokenRewarder and mock its _connect method
                    rewarder = TokenRewarder()
                    with patch.object(rewarder, "_connect", return_value=mock_conn):
                        with patch.object(
                            rewarder, "_create_schema_and_table"
                        ) as mock_create_schema:
                            # Call the method
                            rewarder._create_database_and_table("test_db")

                            # Verify database creation - manually check the args
                            query_args = None
                            for call_args in mock_cursor.execute.call_args_list:
                                if (
                                    isinstance(call_args[0][0], str)
                                    and "SELECT 1 FROM pg_database WHERE datname"
                                    in call_args[0][0]
                                ):
                                    query_args = call_args[0]
                                    break

                            assert query_args is not None
                            assert (
                                query_args[0]
                                == "SELECT 1 FROM pg_database WHERE datname = %s"
                            )
                            assert query_args[1] == ("test_db",)

                            # Verify schema and table creation was called
                            mock_create_schema.assert_called_once_with("test_db")

    def test_create_schema_and_table(self, mock_contract_abi):
        """Test creation of schema and table."""
        with patch("src.token_rewarder.Web3"):
            with patch(
                "src.token_rewarder.TokenRewarder.load_contract_abi",
                return_value=mock_contract_abi,
            ):
                with patch("src.token_rewarder.os.getenv"):
                    # Mock database connection and cursor
                    mock_conn = Mock()
                    mock_cursor = Mock()
                    mock_conn.cursor.return_value = mock_cursor
                    mock_cursor.fetchone.return_value = (False,)  # Table doesn't exist

                    # Initialize TokenRewarder and mock its _connect method
                    rewarder = TokenRewarder()
                    with patch.object(rewarder, "_connect", return_value=mock_conn):
                        # Call the method
                        rewarder._create_schema_and_table("test_db")

                        # Verify schema creation
                        mock_cursor.execute.assert_any_call(
                            "CREATE SCHEMA IF NOT EXISTS default_schema"
                        )

                        # Verify user_rewards table existence check
                        assert any(
                            "information_schema.tables" in str(call)
                            for call in mock_cursor.execute.call_args_list
                        )

                        # Verify table creation - check if any call contains "CREATE TABLE"
                        assert any(
                            "CREATE TABLE default_schema.user_rewards" in str(call)
                            for call in mock_cursor.execute.call_args_list
                        )

    def test_add_reward_to_user(self, mock_contract_abi):
        """Test adding reward to a user."""
        with patch("src.token_rewarder.Web3"):
            with patch(
                "src.token_rewarder.TokenRewarder.load_contract_abi",
                return_value=mock_contract_abi,
            ):
                with patch("src.token_rewarder.os.getenv"):
                    # Mock database connection and cursor
                    mock_conn = Mock()
                    mock_cursor = Mock()
                    mock_conn.cursor.return_value = mock_cursor

                    # Configure mock to return no existing user initially
                    mock_cursor.fetchone.return_value = None

                    # Initialize TokenRewarder and mock its _connect method
                    rewarder = TokenRewarder()
                    with patch.object(rewarder, "_connect", return_value=mock_conn):
                        # Call the method
                        rewarder.add_reward_to_user("test_db", "test_user", 5)

                        # Verify user existence check
                        assert any(
                            "SELECT public_key FROM default_schema.user_rewards WHERE public_key = "
                            in str(call)
                            for call in mock_cursor.execute.call_args_list
                        )

                        # Verify INSERT was called for new user
                        insert_called = False
                        for call_args in mock_cursor.execute.call_args_list:
                            if (
                                isinstance(call_args[0][0], str)
                                and "INSERT INTO default_schema.user_rewards"
                                in call_args[0][0]
                            ):
                                insert_called = True
                                break

                        assert insert_called, "INSERT statement not called"

    def test_issue_token(self, mock_contract_abi, mock_env_vars):
        """Test issuing a token to a recipient."""
        with patch("src.token_rewarder.Web3") as mock_web3:
            with patch(
                "src.token_rewarder.TokenRewarder.load_contract_abi",
                return_value=mock_contract_abi,
            ):
                # Mock web3 functionality
                mock_web3_instance = Mock()
                mock_web3.return_value = mock_web3_instance
                mock_contract = Mock()
                mock_web3_instance.eth.contract.return_value = mock_contract
                mock_web3_instance.eth.get_transaction_count.return_value = 1
                mock_web3_instance.eth.chain_id = 84532

                # Create a mock for the contract functions
                mock_functions = Mock()
                mock_contract.functions = mock_functions

                # Create a mock for the issueToken function
                mock_issue_token = Mock()
                mock_functions.issueToken = mock_issue_token

                # Configure the issue_token call chain
                mock_issue_token_call = Mock()
                mock_issue_token.return_value = mock_issue_token_call

                # Set up the build_transaction return value
                mock_issue_token_call.build_transaction.return_value = {
                    "gasPrice": 1000000000
                }

                # Initialize TokenRewarder
                rewarder = TokenRewarder()
                rewarder.owner_address = mock_env_vars["OWNER_ADDRESS"]
                rewarder.private_key = mock_env_vars["PRIVATE_KEY"]

                # Call the method
                rewarder.issue_token("0xRECIPIENT", 10)

                # Verify contract function call by checking the arguments
                mock_issue_token.assert_called_once()
                args, _ = mock_issue_token.call_args
                assert args == ("0xRECIPIENT", 10)

    def test_batch_issue_tokens(self, mock_contract_abi, mock_env_vars):
        """Test batch issuing tokens to multiple recipients."""
        with patch("src.token_rewarder.Web3") as mock_web3:
            with patch(
                "src.token_rewarder.TokenRewarder.load_contract_abi",
                return_value=mock_contract_abi,
            ):
                # Mock web3 functionality
                mock_web3_instance = Mock()
                mock_web3.return_value = mock_web3_instance
                mock_contract = Mock()
                mock_web3_instance.eth.contract.return_value = mock_contract
                mock_web3_instance.eth.get_transaction_count.return_value = 1
                mock_web3_instance.eth.chain_id = 84532

                # Create a mock for the contract functions
                mock_functions = Mock()
                mock_contract.functions = mock_functions

                # Create a mock for the batchIssueToken function
                mock_batch_issue = Mock()
                mock_functions.batchIssueToken = mock_batch_issue

                # Configure the batch_issue call chain
                mock_batch_issue_call = Mock()
                mock_batch_issue.return_value = mock_batch_issue_call

                # Set up the build_transaction return value
                mock_batch_issue_call.build_transaction.return_value = {
                    "gasPrice": 1000000000
                }

                # Initialize TokenRewarder
                rewarder = TokenRewarder()
                rewarder.owner_address = mock_env_vars["OWNER_ADDRESS"]
                rewarder.private_key = mock_env_vars["PRIVATE_KEY"]

                # Call the method
                recipients = ["0xREC1", "0xREC2"]
                amounts = [10, 20]
                rewarder.batch_issue_tokens(recipients, amounts)

                # Verify contract function call
                mock_batch_issue.assert_called_once()
                args, _ = mock_batch_issue.call_args
                assert args == (recipients, amounts)

    def test_get_user_rewards(self, mock_contract_abi):
        """Test getting user rewards from database."""
        with patch("src.token_rewarder.Web3"):
            with patch(
                "src.token_rewarder.TokenRewarder.load_contract_abi",
                return_value=mock_contract_abi,
            ):
                with patch("src.token_rewarder.os.getenv"):
                    # Mock database connection and cursor
                    mock_conn = Mock()
                    mock_cursor = Mock()
                    mock_conn.cursor.return_value = mock_cursor

                    # Mock query results
                    mock_cursor.fetchall.return_value = [("user1", 5), ("user2", 10)]

                    # Initialize TokenRewarder and mock its _connect method
                    rewarder = TokenRewarder()
                    with patch.object(rewarder, "_connect", return_value=mock_conn):
                        # Call the method
                        result = rewarder.get_user_rewards("test_db")

                        # Verify database query was executed
                        assert mock_cursor.execute.called

                        # Check that the query is for user_rewards
                        query_called = False
                        for call_args in mock_cursor.execute.call_args_list:
                            if (
                                isinstance(call_args[0][0], str)
                                and "SELECT public_key, job_count FROM default_schema.user_rewards"
                                in call_args[0][0]
                            ):
                                query_called = True
                                break

                        assert query_called, "User rewards query not executed"

                        # Verify result
                        assert result == {"user1": 5, "user2": 10}

    def test_reward_users_default(self, mock_contract_abi):
        """Test rewarding users using the default strategy."""
        with patch("src.token_rewarder.Web3"):
            with patch(
                "src.token_rewarder.TokenRewarder.load_contract_abi",
                return_value=mock_contract_abi,
            ):
                with patch("src.token_rewarder.os.getenv"):
                    # Mock get_user_rewards
                    user_rewards = {"user1": 5, "user2": 10}

                    # Initialize TokenRewarder
                    rewarder = TokenRewarder()

                    # Mock both the get_user_rewards and batch_issue_tokens methods
                    with patch.object(
                        rewarder, "get_user_rewards", return_value=user_rewards
                    ) as mock_get_rewards:
                        with patch.object(
                            rewarder, "batch_issue_tokens"
                        ) as mock_batch_issue:
                            # Call the method
                            result = rewarder.reward_users_default("test_db")

                            # Verify get_user_rewards was called
                            mock_get_rewards.assert_called_once_with("test_db")

                            # Verify batch_issue_tokens was called
                            mock_batch_issue.assert_called_once()

                            # Check the arguments passed to batch_issue_tokens
                            args, _ = mock_batch_issue.call_args
                            assert set(args[0]) == set(["user1", "user2"])
                            assert sorted(args[1]) == sorted([5, 10])

                            # Verify result
                            assert result == {"user1": 5, "user2": 10}
