"""
IPFS utilities module.

This module provides unified IPFS operations supporting both Lighthouse and local IPFS.
"""

import os
import tempfile
import threading
import urllib.parse
from pathlib import Path
from typing import Optional, Union

import requests
import requests_unixsocket

from src.utils.logging_utils import get_logger

# Get logger for this module
logger = get_logger(__name__)


class IPFSClient:
    """Handles IPFS operations supporting both Lighthouse and local IPFS."""

    def __init__(self, mode: str = None, api_key: str = None, socket_path: str = None):
        # Add lock for thread-safe operations
        self._operation_lock = threading.Lock()
        """
        Initialize IPFS client.

        Args:
            mode: IPFS mode - 'lighthouse' or 'local'. Defaults to environment variable IPFS_MODE
            api_key: API key for Lighthouse (required for lighthouse mode)
            socket_path: Unix socket path for local IPFS (required for local mode)
        """
        self.mode = mode or os.getenv("IPFS_MODE", "lighthouse")

        if self.mode == "lighthouse":
            self.api_key = api_key or os.getenv("LIGHTHOUSE_TOKEN")
            if not self.api_key:
                raise ValueError("API key required for Lighthouse mode")
            self.api_url = "https://node.lighthouse.storage/api/v0/add"
            self.gateway_url = "https://gateway.lighthouse.storage/ipfs"

        elif self.mode == "local":
            self.socket_path = socket_path or os.getenv(
                "IPFS_SOCKET_PATH", "/root/.ipfs/api.sock"
            )
            if not os.path.exists(self.socket_path):
                raise ValueError(f"IPFS socket not found at {self.socket_path}")

            # Follow the exact pattern from the user's working example
            encoded = urllib.parse.quote_plus(self.socket_path)
            self.base_url = f"http+unix://{encoded}/api/v0"
            
            # Don't create a shared session - we'll create fresh ones per request
            # to avoid thread-safety issues
            self.gateway_url = os.getenv(
                "IPFS_GATEWAY_URL", "http://localhost:8080/ipfs"
            )

        else:
            raise ValueError(
                f"Invalid IPFS mode: {mode}. Must be 'lighthouse' or 'local'"
            )

    def upload_file(self, filepath: Union[str, Path]) -> str:
        """
        Upload a file to IPFS and return the CID.

        Args:
            filepath: Path to the file to upload

        Returns:
            IPFS CID of the uploaded file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Use lock to prevent concurrent access to the session
        with self._operation_lock:
            if self.mode == "lighthouse":
                with open(filepath, "rb") as f:
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    response = requests.post(
                        self.api_url, headers=headers, files={"file": f}
                    )
            else:  # local mode
                # Follow the exact pattern from the user's working example
                with open(filepath, "rb") as f:
                    file_content = f.read()

                files = {"file": (filepath.name, file_content, "application/octet-stream")}

                # Create a fresh session for each request to avoid thread-safety issues
                session = requests_unixsocket.Session()
s                try:
                    session.mount("http+unix://", requests_unixsocket.UnixAdapter())
                    response = session.post(
                        f"{self.base_url}/add?pin=true", files=files
                    )
                finally:
                    # Properly close the session to prevent resource leaks
                    session.close()

            response.raise_for_status()
            return response.json()["Hash"]

    def upload_text(self, text: str, filename: Optional[str] = None) -> str:
        """
        Upload text content to IPFS and return the CID.

        Args:
            text: Text content to upload
            filename: Optional filename for the temporary file

        Returns:
            IPFS CID of the uploaded content
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            cid = self.upload_file(tmp_path)
            return cid
        finally:
            os.unlink(tmp_path)

    def get_content(self, cid: str) -> str:
        """
        Retrieve content from IPFS by CID.

        Args:
            cid: IPFS CID to retrieve

        Returns:
            Content as string
        """
        # Use lock to prevent concurrent access to the session
        with self._operation_lock:
            if self.mode == "lighthouse":
                url = f"{self.gateway_url}/{cid}"
                response = requests.get(url)
            else:  # local mode
                # Create a fresh session for each request to avoid thread-safety issues
                session = requests_unixsocket.Session()
                try:
                    session.mount("http+unix://", requests_unixsocket.UnixAdapter())
                    response = session.post(f"{self.base_url}/cat?arg={cid}")
                finally:
                    # Properly close the session to prevent resource leaks
                    session.close()

            response.raise_for_status()
            return (
                response.text
                if hasattr(response, "text")
                else response.content.decode("utf-8")
            )

    def get_gateway_url(self, cid: str) -> str:
        """
        Get the gateway URL for a CID.

        Args:
            cid: IPFS CID

        Returns:
            Gateway URL
        """
        return f"{self.gateway_url}/{cid}"


# Singleton instance that can be initialized once and reused
_ipfs_client: Optional[IPFSClient] = None
_ipfs_client_lock = threading.Lock()


def get_ipfs_client(
    mode: str = None, api_key: str = None, socket_path: str = None
) -> IPFSClient:
    """
    Get or create the singleton IPFS client instance (thread-safe).

    Args:
        mode: IPFS mode - 'lighthouse' or 'local'
        api_key: API key for Lighthouse
        socket_path: Unix socket path for local IPFS

    Returns:
        IPFSClient instance
    """
    global _ipfs_client

    if _ipfs_client is None:
        with _ipfs_client_lock:
            # Double-check after acquiring lock
            if _ipfs_client is None:
                _ipfs_client = IPFSClient(mode=mode, api_key=api_key, socket_path=socket_path)

    return _ipfs_client


# Compatibility functions to replace the old upload_to_lighthouse function
def upload_to_lighthouse(filepath: Union[str, Path], ipfs_api_key: str) -> str:
    """
    Uploads a file to IPFS and returns the gateway url.
    This function maintains compatibility with existing code.
    """
    client = get_ipfs_client(mode="lighthouse", api_key=ipfs_api_key)
    cid = client.upload_file(filepath)
    return client.get_gateway_url(cid)
