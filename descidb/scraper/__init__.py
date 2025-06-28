"""Research paper scraper module for DeSciDB."""

from .openalex_scraper import OpenAlexScraper
from .config import ScraperConfig

__all__ = ["OpenAlexScraper", "ScraperConfig"] 