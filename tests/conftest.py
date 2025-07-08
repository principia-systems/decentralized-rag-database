"""
Configuration module for pytest.

This module contains fixtures and setup/teardown functions for tests.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock

import pytest

# Mock problematic modules to avoid OpenCV import issues


class MockModule(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


# Create mock modules before any tests run
MOCK_MODULES = ["cv2", "marker", "surya"]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MockModule()
    # Also mock submodules that might be imported
    for submod in ["config", "converters", "builders", "layout", "common"]:
        sys.modules[f"{mod_name}.{submod}"] = MockModule()
        sys.modules[f"{mod_name}.{submod}.parser"] = MockModule()
        sys.modules[f"{mod_name}.{submod}.pdf"] = MockModule()
        sys.modules[f"{mod_name}.{submod}.document"] = MockModule()
        sys.modules[f"{mod_name}.{submod}.layout"] = MockModule()
        sys.modules[f"{mod_name}.{submod}.loader"] = MockModule()
        sys.modules[f"{mod_name}.{submod}.donut"] = MockModule()
        sys.modules[f"{mod_name}.{submod}.donut.processor"] = MockModule()

# Set up mock for the convert function
sys.modules["src.core.converter"] = MockModule()
sys.modules["src.core.converter"].convert = MagicMock()


@pytest.fixture
def sample_text():
    """Return a sample text for testing."""
    return """This is paragraph 1. It has multiple sentences. Each should be identified correctly.

This is paragraph 2. It also has multiple sentences! And questions?

This is the final paragraph. Just one more sentence."""


@pytest.fixture
def temp_pdf_file():
    """Create a temporary file with .pdf extension for testing."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("PDF content placeholder")
        yield path
    finally:
        # Cleanup the temp file after test completes
        os.unlink(path)


@pytest.fixture
def temp_tar_file():
    """Create a temporary file with .tar extension for testing."""
    fd, path = tempfile.mkstemp(suffix=".tar")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("TAR content placeholder")
        yield path
    finally:
        # Cleanup the temp file after test completes
        os.unlink(path)


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response with an embedding."""
    mock_response = MagicMock()
    mock_data = MagicMock()
    mock_data.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_response.data = [mock_data]
    return mock_response


@pytest.fixture(scope="session")
def test_env_vars():
    """Set up test environment variables."""
    env_vars = {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "test_password",
        "OPENAI_API_KEY": "test_openai_key",
        "LIGHTHOUSE_TOKEN": "test_lighthouse_token",
        "IPFS_MODE": "lighthouse",  # Set to lighthouse for tests
        "CHROMA_PERSIST_DIR": "./test_chroma_db",
        "POSTGRES_DB": "test_db",
        "POSTGRES_USER": "test_user",
        "POSTGRES_PASSWORD": "test_password",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "DATABASE_URL": "postgresql://test_user:test_password@localhost:5432/test_db",
        "ETH_NODE_URL": "http://localhost:8545",
        "PRIVATE_KEY": "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "CONTRACT_ADDRESS": "0x0000000000000000000000000000000000000000",
    }

    # Save original values and set test values
    for var, value in env_vars.items():
        original_value = os.environ.get(var)
        os.environ[var] = value

    yield env_vars

    # Restore original values
    for var, value in env_vars.items():
        if value is not None:
            os.environ[var] = value
        else:
            if var in os.environ:
                del os.environ[var]
