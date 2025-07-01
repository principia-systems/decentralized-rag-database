"""
PDF conversion module.

This module provides functions for converting PDF documents to text
using various methods including OpenAI's API and local tools.
"""

import os
import textwrap
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
from src.utils.utils import download_from_url, extract

# Get module logger
logger = get_logger(__name__)


load_dotenv(override=True)


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
    pass
    try:
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

        models = create_model_dict()
        config_parser = ConfigParser(
            {
                "languages": "en",
                "output_format": "markdown",
            }
        )

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        std_out = ""
        for pdf_path in input_pdf_paths:
            rendered = converter(pdf_path)
            rendered_markdown = rendered.markdown
            std_out += rendered_markdown

        return std_out

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return ""  # Return empty string in case of error


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
    try:
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

        md = MarkItDown(enable_plugins=False)

        logger.info(f"Converting {input_pdf_path} using MarkItDown")
        result = md.convert(input_pdf_path)
        return result.text_content.strip()

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return ""
    except Exception as e:
        logger.error(f"An error occurred with MarkItDown: {e}")
        return ""
