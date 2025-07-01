"""
Token reward module.

This module provides a TokenRewarder class for calculating and distributing
rewards to users based on their contributions to the system.
"""

import itertools
import json
import math
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from psycopg2 import connect, sql
from web3 import Web3

from src.utils.logging_utils import get_logger

# Get module logger
logger = get_logger(__name__)

load_dotenv()


class TokenRewarder:
    """
    Manages token rewards for contributors to the ecosystem.

    This class handles the allocation, tracking, and distribution of blockchain tokens
    to users who contribute to the system by uploading and processing scientific documents.
    """

    def __init__(
        self,
        network="test_base",
        contract_address="0x3bB10ec2404638c6fB9f98948f8e3730316B7BfA",
        contract_abi_path=None,
        db_components=None,
        host="localhost",
        port=5432,
        user="",
        password="",
    ):
        """
        Initialize the TokenRewarder with blockchain and database connections.

        Args:
            network: Blockchain network to connect to ('test_base', 'optimism', or 'base')
            contract_address: Address of the token contract
            contract_abi_path: Path to the contract ABI JSON file. If None, uses default path
            db_components: Dictionary containing converter, chunker, and embedder components
            host: PostgreSQL server hostname
            port: PostgreSQL server port
            user: PostgreSQL username
            password: PostgreSQL password
        """
        self.logger = get_logger(__name__ + ".TokenRewarder")
        self._initialize_network(network)

        # Determine the contract ABI path
        if contract_abi_path is None:
            # Use a relative path from the project root
            project_root = Path(__file__).parent.parent.parent
            contract_abi_path = project_root / "contracts" / "CoopHiveV1.json"

        contract_abi = self.load_contract_abi(contract_abi_path)["abi"]

        # Initialize Web3 and contract
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self.contract_address = contract_address
        self.contract = self.web3.eth.contract(
            address=self.contract_address, abi=contract_abi
        )

        # Blockchain credentials from environment variables
        self.owner_address = os.getenv("OWNER_ADDRESS")
        self.private_key = os.getenv("PRIVATE_KEY")

        # Store PostgreSQL connection details
        self.host = host
        self.port = port
        self.user = user
        self.password = password

        # Generate database names and initialize reward tables
        if db_components:
            self.db_names = self.generate_db_names(db_components)
            self._initialize_reward_tables()
        else:
            self.db_names = []

        self.logger.info(f"Initialized TokenRewarder for network: {network}")
        self.logger.info(f"Contract address: {contract_address}")

    def _initialize_network(self, network):
        """Sets the RPC URL and chain ID based on the specified network."""
        if network == "optimism":
            self.rpc_url = "https://mainnet.optimism.io"
            self.chain_id = 10
        elif network == "test_base":
            self.rpc_url = "https://sepolia.base.org"
            self.chain_id = 84532
        elif network == "base":
            self.rpc_url = "https://mainnet.base.org"
            self.chain_id = 1234
        else:
            raise ValueError(
                "Unsupported network. Choose 'optimism', 'test_base', or 'base'."
            )

    def _connect(self, dbname="postgres"):
        """Establishes a connection to the specified PostgreSQL database."""
        try:
            conn = connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=dbname,
            )
            conn.autocommit = True
            return conn
        except Exception as e:
            self.logger.error(f"Error connecting to the database: {e}")
            return None

    def load_contract_abi(self, abi_path):
        """Loads the contract ABI from the given path."""
        with open(abi_path, "r") as abi_file:
            return json.load(abi_file)

    def generate_db_names(self, components):
        """Generates database names using Cartesian product of components."""
        return [
            f"{c}_{ch}_{e}_token"
            for c, ch, e in itertools.product(
                components["converter"], components["chunker"], components["embedder"]
            )
        ]

    def _initialize_reward_tables(self):
        """Creates reward tables in all generated databases."""
        for db_name in self.db_names:
            self._create_database_and_table(db_name)

    def _create_database_and_table(self, db_name):
        """Creates the database and initializes the reward table."""
        conn = self._connect()
        if conn is None:
            self.logger.error("Unable to connect to PostgreSQL server.")
            return

        cursor = conn.cursor()
        try:
            # Check if the database exists
            cursor.execute(
                sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [db_name]
            )
            if not cursor.fetchone():
                # Create the new database
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                )
                self.logger.info(f"Database '{db_name}' created successfully.")

            # Ensure schema and table are created in the new database
            self._create_schema_and_table(db_name)

        except Exception as e:
            self.logger.error(f"Error creating database or table: {e}")
        finally:
            cursor.close()
            conn.close()

    def _create_schema_and_table(self, db_name):
        """Creates the schema and 'user_rewards' table in the given database, if they don't already exist."""
        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(f"Unable to connect to the database '{db_name}'.")
            return

        cursor = conn.cursor()
        try:
            # Create schema if it doesn't exist
            cursor.execute("CREATE SCHEMA IF NOT EXISTS default_schema")

            # Check if 'user_rewards' table exists
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'default_schema'
                    AND table_name = 'user_rewards'
                )
            """
            )
            table_exists = cursor.fetchone()[0]

            if not table_exists:
                # Create the 'user_rewards' table with an id as primary key
                cursor.execute(
                    """
                    CREATE TABLE default_schema.user_rewards (
                        id SERIAL PRIMARY KEY,
                        public_key TEXT NOT NULL,
                        job_count INT DEFAULT 0,
                        time_stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )
                self.logger.info(f"Initialized 'user_rewards' table in '{db_name}'.")
            else:
                self.logger.info(
                    f"'user_rewards' table already exists in '{db_name}', skipping creation."
                )

        except Exception as e:
            self.logger.error(f"Error creating schema or table: {e}")
        finally:
            cursor.close()
            conn.close()

    def add_reward_to_user(self, public_key, db_name, job_count=1):
        """
        Adds a new entry for the user with a specific job count.

        :param public_key: The public key of the user.
        :param db_name: The database name where the user record exists.
        :param job_count: The number of jobs to add.
        """
        db_name = f"{db_name}_token"

        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(f"❌ Unable to connect to the database '{db_name}'.")
            return

        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO default_schema.user_rewards (public_key, job_count, time_stamp)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (public_key)
                DO UPDATE SET job_count = default_schema.user_rewards.job_count + EXCLUDED.job_count
                """,
                (public_key, job_count),
            )
            self.logger.info(
                f"✅ Added entry for user '{public_key}' with job_count {job_count}."
            )

        except Exception as e:
            self.logger.error(f"❌ Error adding reward entry: {e}")
        finally:
            cursor.close()
            conn.close()

    def issue_token(self, recipient_address, amount=1):
        """Issues tokens to the recipient address."""
        if not self.owner_address:
            self.logger.error("❌ OWNER_ADDRESS is not set!")
            return False

        if not recipient_address:
            self.logger.error("❌ Recipient address is invalid!")
            return False

        try:
            nonce = self.web3.eth.get_transaction_count(self.owner_address, "pending")

            txn = self.contract.functions.transfer(
                str(recipient_address), int(amount * 1e18)
            ).build_transaction(
                {
                    "chainId": self.chain_id,
                    "gas": 100000,
                    "gasPrice": self.web3.eth.gas_price,
                    "nonce": nonce,
                }
            )

            signed_txn = self.web3.eth.account.sign_transaction(txn, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            self.logger.info(f"✅ Transaction sent: {self.web3.to_hex(tx_hash)}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error sending transaction: {e}")
            return False

    def batch_issue_tokens(self, recipients, amounts):
        """Issue tokens to multiple addresses in a single transaction using batchDistribute."""
        if not self.owner_address:
            self.logger.error("❌ OWNER_ADDRESS is not set!")
            return False

        if not recipients or not amounts or len(recipients) != len(amounts):
            self.logger.error("❌ Invalid recipients or amounts for batch distribution")
            return False

        try:
            nonce = self.web3.eth.get_transaction_count(self.owner_address, "pending")
            self.logger.info(
                f"🏦 Batch issuing tokens to {len(recipients)} recipients..."
            )

            # Convert amounts to wei (multiply by 10^18)
            wei_amounts = [int(amount * 1e18) for amount in amounts]

            # Call the batchDistribute function
            txn = self.contract.functions.batchDistribute(
                recipients, wei_amounts
            ).build_transaction(
                {
                    "chainId": self.chain_id,
                    # Base gas + extra for each recipient
                    "gas": 200000 + (70000 * len(recipients)),
                    "gasPrice": self.web3.eth.gas_price,
                    "nonce": nonce,
                }
            )

            signed_txn = self.web3.eth.account.sign_transaction(txn, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
            self.logger.info(f"✅ Batch transaction sent: {self.web3.to_hex(tx_hash)}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error sending batch transaction: {e}")
            return False

    def get_user_rewards(self, db_name):
        """
        Get all user rewards and distribute them in a single batch transaction.

        This method retrieves user rewards from the database and uses the
        batchDistribute function of the ERC20 contract to send all rewards
        in a single transaction, which is more gas-efficient than sending
        individual transactions.

        Args:
            db_name (str): The name of the database to query for user rewards
        """
        user_rewards = self.reward_users_constant(db_name)

        if not user_rewards:
            self.logger.info("No rewards to distribute")
            return

        # Collect all recipients and amounts for batch distribution
        recipients = []
        amounts = []

        for user, amount in user_rewards.items():
            self.logger.info(f"Adding {amount:.2f} tokens for user '{user}'")
            recipients.append(user)
            amounts.append(amount)

        # Use batch distribution
        if recipients and amounts:
            self.logger.info(
                f"Issuing tokens to {len(recipients)} users in a single transaction"
            )
            self.batch_issue_tokens(recipients, amounts)

    def reward_users_after_time(self, db_name, start_time, reward_per_job=1):
        """Rewards users based on a constant reward per job count after a specified time."""
        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(f"Unable to connect to the database '{db_name}'.")
            return

        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT public_key, SUM(job_count) AS total_jobs
                FROM default_schema.user_rewards
                WHERE time_stamp >= %s
                GROUP BY public_key
            """,
                (start_time,),
            )

            user_entries = cursor.fetchall()

            rewards = {}
            for public_key, total_jobs in user_entries:
                rewards[public_key] = total_jobs * reward_per_job

            self.logger.info("\nRewards After Specified Time:")
            for user, reward in rewards.items():
                self.logger.info(f"  User '{user}': {reward:.2f} tokens")

            return rewards

        except Exception as e:
            self.logger.error(f"Error calculating time-based rewards: {e}")
        finally:
            cursor.close()
            conn.close()

    def reward_users_milestone(self, db_name, milestone=10, reward_per_job=1):
        """Rewards users based on a milestone-based reward scheme."""
        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(f"Unable to connect to the database '{db_name}'.")
            return

        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT public_key, SUM(job_count) AS total_jobs
                FROM default_schema.user_rewards
                GROUP BY public_key
                HAVING SUM(job_count) >= %s
            """,
                (milestone,),
            )

            user_entries = cursor.fetchall()

            rewards = {}
            for public_key, total_jobs in user_entries:
                rewards[public_key] = total_jobs * reward_per_job

            self.logger.info("\nMilestone-Based Rewards:")
            for user, reward in rewards.items():
                self.logger.info(f"  User '{user}': {reward:.2f} tokens")

            return rewards

        except Exception as e:
            self.logger.error(f"Error calculating milestone-based rewards: {e}")
        finally:
            cursor.close()
            conn.close()

    def reward_users_with_bonus(
        self, db_name, bonus_threshold=50, bonus=10, reward_per_job=1
    ):
        """Rewards users based on a bonus threshold and bonus amount."""
        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(f"Unable to connect to the database '{db_name}'.")
            return

        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT public_key, SUM(job_count) AS total_jobs
                FROM default_schema.user_rewards
                GROUP BY public_key
            """
            )

            user_entries = cursor.fetchall()

            rewards = {}
            for public_key, total_jobs in user_entries:
                reward = total_jobs * reward_per_job
                if total_jobs >= bonus_threshold:
                    reward += bonus
                rewards[public_key] = reward

            self.logger.info("\nRewards with Bonuses:")
            for user, reward in rewards.items():
                self.logger.info(f"  User '{user}': {reward:.2f} tokens")

            return rewards

        except Exception as e:
            self.logger.error(f"Error calculating rewards with bonuses: {e}")
        finally:
            cursor.close()
            conn.close()

    def reward_users_constant(self, db_name, reward_per_job=1):
        """Rewards users based on a constant reward per job count."""
        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(f"Unable to connect to the database '{db_name}'.")
            return

        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT public_key, SUM(job_count) AS total_jobs
                FROM default_schema.user_rewards
                GROUP BY public_key
            """
            )

            user_entries = cursor.fetchall()

            rewards = {}
            for public_key, total_jobs in user_entries:
                rewards[public_key] = total_jobs * reward_per_job

            self.logger.info("\nConstant Rewards:")
            for user, reward in rewards.items():
                self.logger.info(f"  User '{user}': {reward:.2f} tokens")

            return rewards

        except Exception as e:
            self.logger.error(f"Error calculating constant rewards: {e}")
        finally:
            cursor.close()
            conn.close()

    def reward_users_default(self, db_name):
        """Rewards users based on a default exponential decay reward scheme."""
        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(f"Unable to connect to the database '{db_name}'.")
            return

        cursor = conn.cursor()
        n_buckets = 3

        try:
            cursor.execute(
                """
                SELECT public_key, job_count, time_stamp
                FROM default_schema.user_rewards
                ORDER BY public_key, time_stamp
            """
            )

            user_entries = cursor.fetchall()

            if not user_entries:
                self.logger.info("No user entries found.")
                return

            current_time = datetime.now()
            bucket_duration = (user_entries[-1][2] - user_entries[0][2]) / n_buckets
            start_time = current_time - (bucket_duration * n_buckets)

            global_buckets = [
                start_time + i * bucket_duration for i in range(n_buckets)
            ]

            # Initialize the bucket map
            bucket_map = {bucket_start: {} for bucket_start in global_buckets}

            # Populate each bucket with user contributions
            for public_key, job_count, time_stamp in user_entries:
                for bucket_start in global_buckets:
                    # Define the end time for the current bucket
                    bucket_end = bucket_start + bucket_duration
                    if bucket_start <= time_stamp < bucket_end:
                        if public_key not in bucket_map[bucket_start]:
                            bucket_map[bucket_start][public_key] = 0

                        bucket_map[bucket_start][public_key] += job_count
                        break

            weights = [math.exp(-i) for i in range(n_buckets)]

            weighted_rewards = {}
            for i, (bucket_start, users) in enumerate(reversed(bucket_map.items())):
                weight = weights[i]
                self.logger.info(f"Bucket starting {bucket_start} (Weight: {weight}):")
                for user, count in users.items():
                    weighted_reward = count * weight
                    if user not in weighted_rewards:
                        weighted_rewards[user] = 0
                    weighted_rewards[user] += weighted_reward
                    self.logger.info(
                        f"  User '{user}': {count} contributions, Weighted reward: {weighted_reward:.2f}"
                    )

            self.logger.info("\nTotal Weighted Rewards:")
            for user, total_reward in weighted_rewards.items():
                self.logger.info(
                    f"  User '{user}': {total_reward:.2f} total weighted reward"
                )

            return weighted_rewards

        except Exception as e:
            self.logger.error(f"Error fetching user rewards: {e}")
        finally:
            cursor.close()
            conn.close()

    def reward_users_within_timeframe(
        self, db_name, start_time, end_time, reward_per_job=1
    ):
        """Rewards users who contributed within a specific timeframe."""
        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(f"Unable to connect to the database '{db_name}'.")
            return

        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT public_key, SUM(job_count) AS total_jobs
                FROM default_schema.user_rewards
                WHERE time_stamp >= %s AND time_stamp <= %s
                GROUP BY public_key
            """,
                (start_time, end_time),
            )

            user_entries = cursor.fetchall()

            rewards = {}
            for public_key, total_jobs in user_entries:
                rewards[public_key] = total_jobs * reward_per_job

            self.logger.info("\nTimeframe-Based Rewards:")
            for user, reward in rewards.items():
                self.logger.info(f"  User '{user}': {reward:.2f} tokens")

            return rewards

        except Exception as e:
            self.logger.error(f"Error calculating timeframe-based rewards: {e}")
        finally:
            cursor.close()
            conn.close()

    def reward_users_by_tier(self, db_name, tiers=None):
        """Rewards users based on their tier of contributions."""
        if tiers is None:
            tiers = {
                100: 5,  # Contributions >= 100 get 5 tokens per job
                50: 3,  # Contributions >= 50 get 3 tokens per job
                0: 1,  # Contributions >= 0 get 1 token per job
            }

        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(f"Unable to connect to the database '{db_name}'.")
            return

        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT public_key, SUM(job_count) AS total_jobs
                FROM default_schema.user_rewards
                GROUP BY public_key
            """
            )

            user_entries = cursor.fetchall()

            rewards = {}
            for public_key, total_jobs in user_entries:
                reward_per_job = 0
                for threshold, reward in sorted(tiers.items(), reverse=True):
                    if total_jobs >= threshold:
                        reward_per_job = reward
                        break
                rewards[public_key] = total_jobs * reward_per_job

            self.logger.info("\nTier-Based Rewards:")
            for user, reward in rewards.items():
                self.logger.info(f"  User '{user}': {reward:.2f} tokens")

            return rewards

        except Exception as e:
            self.logger.error(f"Error calculating tier-based rewards: {e}")
        finally:
            cursor.close()
            conn.close()
