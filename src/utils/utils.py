"""
Utility functions for src.

This module provides various utility functions used across the src package,
including file handling, URL operations, compression, and IPFS integration.
"""

import mimetypes
import tarfile
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

import requests


def compress(
    list_of_input_paths: List[Union[str, Path]], output_path: Union[str, Path]
):
    """Compresses the given list of input file paths into a tar file."""
    output_path = str(output_path)
    if not output_path.endswith(".tar"):
        output_path += ".tar"

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output_path, "w") as tar:
        for input_path in list_of_input_paths:
            input_path_obj = Path(input_path)
            if not input_path_obj.exists():
                print(f"Warning: {input_path} does not exist and will be skipped.")
                continue
            # Add the file or directory to the tar file
            tar.add(str(input_path_obj), arcname=input_path_obj.name)
            print(f"Added {input_path} as {input_path_obj.name} to {output_path}")


def extract(tar_file_path: Union[str, Path], output_path: Union[str, Path]):
    """Extracts the contents of a tar file to the specified output directory."""
    # Ensure the tar file exists
    tar_path = Path(tar_file_path)
    if not tar_path.exists():
        raise FileNotFoundError(f"The tar file '{tar_file_path}' does not exist.")

    # Ensure the output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the tar file and extract its contents
    with tarfile.open(str(tar_path), "r") as tar:
        print(f"Extracting {tar_file_path} to {output_path}...")
        tar.extractall(path=str(output_dir))
        print("Extraction complete.")

    tar_path.unlink()


def download_from_url(url: str, output_folder: Union[str, Path] = "./tmp"):
    """Downloads a file from a given URL and saves it to the specified output path."""
    # Ensure the output folder exists
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch the file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    parsed_url = urlparse(url)
    file_name = Path(parsed_url.path).name

    content_type = response.headers.get("content-type")
    extension = mimetypes.guess_extension(content_type) if content_type else ".bin"
    file_name = f"{file_name}{extension}"

    # Save the file in the output folder
    output_file = output_dir / file_name
    with open(str(output_file), "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    return str(output_file)
