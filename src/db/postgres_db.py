"""
PostgreSQL database manager.

This module provides a PostgresDBManager class for managing PostgreSQL databases
used to store document metadata and user statistics.
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
import psycopg2
from psycopg2 import sql

from src.utils.logging_utils import get_logger

# Get module logger
logger = get_logger(__name__)


class PostgresDBManager:
    """
    Manages PostgreSQL database connections and operations.

    This class provides methods to create databases, tables, and perform
    database operations related to document processing and user statistics.
    """

    def __init__(self, host=None, port=None, user=None, password=None):
        """
        Initialize a PostgresDBManager with connection parameters.

        Args:
            host: PostgreSQL server hostname
            port: PostgreSQL server port
            user: PostgreSQL username
            password: PostgreSQL password
        """
        self.logger = get_logger(__name__ + ".PostgresDBManager")
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or os.getenv("POSTGRES_PORT", "5432")
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "")

        try:
            # Connect to the default postgres database first
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname="postgres",
            )
            self.conn.autocommit = True
            self.logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Error connecting to PostgreSQL: {e}")
            raise

    def _connect(self, dbname: str = "postgres"):
        try:
            conn = psycopg2.connect(
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

    def create_databases(self, db_names: List[str]):
        conn = self._connect()
        if conn is None:
            self.logger.error("Unable to connect to the PostgreSQL server.")
            return

        cursor = conn.cursor()
        for db_name in db_names:
            try:
                cursor.execute(
                    sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [db_name]
                )
                exists = cursor.fetchone()

                if not exists:
                    cursor.execute(
                        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                    )
                    self.logger.info(f"Database '{db_name}' created successfully.")

                    self._create_schema_and_table_in_db(db_name)
                else:
                    self.logger.info(
                        f"Database '{db_name}' already exists. Skipping creation."
                    )

            except Exception as e:
                self.logger.error(f"Error creating database '{db_name}': {e}")

        cursor.close()
        conn.close()

    def _create_schema_and_table_in_db(self, db_name: str):
        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(f"Unable to connect to the database '{db_name}'.")
            return

        cursor = conn.cursor()
        try:
            cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS default_schema"))
            self.logger.info(
                f"Schema 'default_schema' created successfully in database '{db_name}'."
            )

            cursor.execute(
                sql.SQL(
                    """
                CREATE TABLE IF NOT EXISTS default_schema.papers (
                    author TEXT,
                    paper_name TEXT,
                    markdown TEXT,
                    embedding BYTEA,
                    metadata JSON,
                    public_key TEXT
                )
            """
                )
            )
            self.logger.info(
                f"Table 'papers' created successfully in schema 'default_schema' of database '{db_name}'."
            )

        except Exception as e:
            self.logger.error(
                f"Error creating schema or table in database '{db_name}': {e}"
            )

        cursor.close()
        conn.close()

    def insert_data(
        self, db_name: str, data: List[Tuple[str, str, str, bytes, dict, bool]]
    ):
        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(
                f"Unable to connect to the database '{db_name}' for data insertion."
            )
            return

        cursor = conn.cursor()
        try:
            insert_query = sql.SQL(
                """
                    INSERT INTO default_schema.papers (author, paper_name, markdown, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """
            )

            for record in data:
                binary_embedding = pickle.dumps(np.array(record[3]))
                new_record = (
                    record[0],
                    record[1],
                    record[2],
                    binary_embedding,
                    record[4],
                    record[5],
                )
                cursor.execute(insert_query, new_record)

        except Exception as e:
            self.logger.error(f"Error inserting data into database '{db_name}': {e}")

        cursor.close()

    def query(self, db_name: str, query_string: str, params: Tuple = ()):
        conn = self._connect(db_name)
        if conn is None:
            self.logger.error(
                f"Unable to connect to the database '{db_name}' for query execution."
            )
            return None

        cursor = conn.cursor()
        try:
            cursor.execute(query_string, params)
            if query_string.strip().lower().startswith("select"):
                result = cursor.fetchall()
                return result
            else:
                conn.commit()
                return None
        except Exception as e:
            self.logger.error(f"Error executing query on database '{db_name}': {e}")
            return None
        finally:
            cursor.close()
            conn.close()
