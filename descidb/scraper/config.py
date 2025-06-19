"""Configuration for the research paper scraper."""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScraperConfig:
    """Configuration for OpenAlex research paper scraper."""
    
    # API settings
    api_base: str = "https://api.openalex.org/works"
    unpaywall_api: str = "https://api.unpaywall.org/v2"
    user_agent: str = "DeSciDB Research Scraper (mailto:contact@coophive.network)"
    referer: str = "https://coophive.network"
    email: str = "contact@coophive.network"
    
    # Search parameters
    topic: str = ""
    min_citations: Optional[int] = 10
    pages: int = 3
    per_page: int = 25
    
    # Download settings
    workers: int = 4
    outdir: str = ""
    downloads_dir: str = ""
    
    # User-specific settings
    user_email: str = ""
    
    @classmethod
    def from_research_area(cls, research_area: str, user_email: str, output_dir: Optional[str] = None, downloads_dir: Optional[str] = None) -> 'ScraperConfig':
        """Create config from research area and user email."""
        project_root = Path(__file__).parent.parent.parent.parent
        
        if output_dir is None:
            # Create user-specific directory in papers folder for temporary storage
            output_dir = str(project_root / "papers" / user_email / "scraped")
        
        if downloads_dir is None:
            # Create downloads directory for zip files
            downloads_dir = str(project_root / "downloads")
        
        return cls(
            topic=research_area,
            user_email=user_email,
            outdir=output_dir,
            downloads_dir=downloads_dir,
            # Reduce pages for faster scraping in API context
            pages=2,
            per_page=20,
            workers=3
        ) 