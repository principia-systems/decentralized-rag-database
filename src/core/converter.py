"""
PDF conversion module.

This module provides functions for converting PDF documents to text
using various methods including OpenAI's API and local tools.
"""

import os
import time
import math
import contextlib
import textwrap
import threading
from typing import List, Optional, Dict

import PyPDF2
from dotenv import load_dotenv
from marker.config.parser import ConfigParser  # type: ignore
from marker.converters.pdf import PdfConverter  # type: ignore
from marker.models import create_model_dict  # type: ignore
from markitdown import MarkItDown
from openai import OpenAI

from src.types.converter import ConverterType, ConverterFunc
from src.utils.logging_utils import get_logger
from src.utils.file_lock import file_lock, PROJECT_ROOT

try:
    import torch
except Exception:
    torch = None

from src.utils.utils import download_from_url, extract

# Get module logger
logger = get_logger(__name__)

load_dotenv(override=True)

# Global lock to prevent concurrent marker model loading
_marker_lock = threading.Lock()
_marker_models = None
_marker_converter = None

# Global lock to prevent concurrent markitdown model loading
_markitdown_lock = threading.Lock()
_markitdown_instance = None


def _compute_converter_gpu_indices_from_split() -> list[int]:
    """Compute converter GPU indices as the complementary set to the embedder.

    - Embedder uses ceil(total_gpus * GPU_SPLIT) GPUs from the high end.
    - Converter uses the remaining GPUs from the low end (round down).
    - This ensures no overlap and that GPU 0 is preferred by converter when available.
    """
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return []

    total_gpus = torch.cuda.device_count()
    if total_gpus <= 0:
        return []

    gpu_split = float(os.getenv("GPU_SPLIT", "0.75"))
    embedder_count = 1 if total_gpus == 1 else min(total_gpus, max(1, math.ceil(total_gpus * gpu_split)))
    converter_count = max(0, total_gpus - embedder_count)

    # Use the lowest indices [0 .. converter_count-1]
    return list(range(converter_count))


@contextlib.contextmanager
def acquire_converter_gpu_lock_with_timeout(logger_prefix: str = "CONVERTER"):
    """Context manager that yields the locked GPU index (or None if not acquired).

    Holds the file lock for the duration of the with-block.
    """
    gpu_indices = _compute_converter_gpu_indices_from_split()
    locks_dir = PROJECT_ROOT / "temp" / "gpu_locks"
    locks_dir.mkdir(parents=True, exist_ok=True)

    total_timeout_sec = int(os.getenv("GPU_LOCK_TOTAL_TIMEOUT", "600"))
    retry_sleep_sec = float(os.getenv("GPU_LOCK_RETRY_SLEEP", "0.2"))

    if not gpu_indices:
        yield None
        return

    start_time = time.time()
    while True:
        for gpu_idx in gpu_indices:
            lock_path = locks_dir / f"gpu_{gpu_idx}.lock"
            try:
                with file_lock(lock_path, timeout=0):
                    # Log which GPU we locked
                    try:
                        if torch and torch.cuda.is_available():
                            gpu_name = torch.cuda.get_device_name(gpu_idx)
                            mem_gb = torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3
                            logger.info(f"[{logger_prefix}] Acquired GPU lock -> idx={gpu_idx}, name={gpu_name} ({mem_gb:.1f} GB)")
                        else:
                            logger.info(f"[{logger_prefix}] Acquired GPU lock -> idx={gpu_idx}")
                    except Exception:
                        logger.info(f"[{logger_prefix}] Acquired GPU lock -> idx={gpu_idx}")
                    yield gpu_idx
                    return
            except TimeoutError:
                continue

        if time.time() - start_time > total_timeout_sec:
            logger.warning(f"[{logger_prefix}] Timed out acquiring converter GPU lock; proceeding without lock")
            yield None
            return

        time.sleep(retry_sleep_sec)


def convert_from_url(conversion_type: ConverterType, input_url: str, user_temp_dir: str = "./tmp") -> str:
    """Convert based on the specified conversion type."""
    download_path = download_from_url(url=input_url, output_folder=user_temp_dir)

    if download_path.endswith(".tar"):
        output_path = download_path[: download_path.rfind("/")]
        extract(tar_file_path=download_path, output_path=output_path)

    return convert(conversion_type=conversion_type, input_path=output_path)


def convert(conversion_type: ConverterType, input_path: str) -> str:
    """Convert based on the specified conversion type."""
    # Mapping conversion types to functions
    conversion_methods: Dict[str, ConverterFunc] = {
        "marker": marker,
        "openai": openai,
        "markitdown": markitdown,
    }

    return conversion_methods[conversion_type](input_path)


def chunk_text(text: str, chunk_size: int = 4000) -> List[str]:
    """Splits text into smaller chunks to fit within token limits."""
    return textwrap.wrap(
        text, width=chunk_size, break_long_words=False, break_on_hyphens=False
    )


def marker(input_path: str) -> str:
    """Convert text using the marker module, where input_path is either a path to pdf file or a path to a folder containing a set of pdf files."""
    global _marker_models, _marker_converter
    
    try:
        # Acquire converter-side GPU lock (prefer low indices; likely 0)
        with acquire_converter_gpu_lock_with_timeout("MARKER") as locked_gpu_idx:
            try:
                if locked_gpu_idx is not None and torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.set_device(locked_gpu_idx)
            except Exception:
                pass

            # Ensure the input_path is a valid file
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input path not found: {input_path}")

            # Check if the path is a file and a PDF
            if os.path.isfile(input_path):
                if input_path.lower().endswith(".pdf"):
                    input_pdf_paths = [input_path]
                else:
                    raise ValueError(f"File at {input_path} is not a PDF.")

            # Check if the path is a folder containing PDFs
            elif os.path.isdir(input_path):
                input_pdf_paths = [
                    os.path.join(input_path, f)
                    for f in os.listdir(input_path)
                    if f.lower().endswith(".pdf")
                ]
                if not input_pdf_paths:
                    raise ValueError(f"No PDF files found in directory: {input_path}")
            else:
                raise ValueError(f"Invalid input path: {input_path}")

            std_out = ""
            
            # Use thread-safe model loading and conversion
            with _marker_lock:
                if _marker_models is None or _marker_converter is None:
                    logger.info("Loading marker models (this may take a moment)...")
                    _marker_models = create_model_dict()
                    config_parser = ConfigParser(
                        {
                            "languages": "en",
                            "output_format": "markdown",
                        }
                    )
                    _marker_converter = PdfConverter(
                        config=config_parser.generate_config_dict(),
                        artifact_dict=_marker_models,
                        processor_list=config_parser.get_processors(),
                        renderer=config_parser.get_renderer(),
                    )
                    logger.info("Marker models loaded successfully")
                
                # Use the cached converter - now conversion happens inside the lock
                converter = _marker_converter
                
                for pdf_path in input_pdf_paths:
                    rendered = converter(pdf_path)
                    rendered_markdown = rendered.markdown
                    std_out += rendered_markdown

            return std_out

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return ""  # Return empty string in case of error
    finally:
        pass


def extract_text_from_pdf(input_path: str) -> str:
    """Extracts text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(input_path)
    text_content = ""

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text_content += page.extract_text()

    return text_content


def openai(input_path: str) -> str:
    """Convert large text to Markdown using OpenAI API with chunking."""
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        pdf_text = extract_text_from_pdf(input_path)
        chunks = chunk_text(pdf_text, chunk_size=4000)

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        markdown_chunks = []

        for chunk in chunks:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"Convert the following text to Markdown:\n\n{chunk}",
                    },
                ],
            )
            if response and response.choices:
                markdown_chunks.append(response.choices[0].message.content)
            else:
                print("Failed to convert a chunk using OpenAI.")
                markdown_chunks.append(chunk)

        filtered_chunks = [chunk for chunk in markdown_chunks if chunk is not None]
        return "\n\n".join(filtered_chunks)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


def markitdown(input_path: str) -> str:
    """Convert PDF to Markdown using the Microsoft MarkItDown library."""
    global _markitdown_instance
    
    try:
        # Acquire converter-side GPU lock (prefer low indices; likely 0)
        with acquire_converter_gpu_lock_with_timeout("MARKITDOWN") as locked_gpu_idx:
            try:
                if locked_gpu_idx is not None and torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.set_device(locked_gpu_idx)
            except Exception:
                pass

            # Ensure the input_path is a valid file
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input path not found: {input_path}")

            # Check if the path is a file and a PDF
            if os.path.isfile(input_path):
                if input_path.lower().endswith(".pdf"):
                    input_pdf_path = input_path
                else:
                    raise ValueError(f"File at {input_path} is not a PDF.")
            elif os.path.isdir(input_path):
                raise ValueError(
                    "Input path is a directory. Please specify a single PDF file path."
                )
            else:
                raise ValueError(f"Invalid input path: {input_path}")

            # Use thread-safe model loading and conversion
            with _markitdown_lock:
                if _markitdown_instance is None:
                    logger.info("Loading MarkItDown instance (this may take a moment)...")
                    _markitdown_instance = MarkItDown(enable_plugins=False)
                    logger.info("MarkItDown instance loaded successfully")
                
                # Use the cached instance
                md = _markitdown_instance
                
                # Perform conversion inside the lock
                logger.info(f"Converting {input_pdf_path} using MarkItDown")
                result = md.convert(input_pdf_path)
                return result.text_content.strip()

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return ""
    except Exception as e:
        logger.error(f"An error occurred with MarkItDown: {e}")
        return ""
    finally:
        pass
