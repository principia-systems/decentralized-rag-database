"""OpenAlex research paper scraper for DeSciDB."""

import os
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Tuple
import zipfile
from pathlib import Path
import logging
import uuid
import time

from .config import ScraperConfig

logger = logging.getLogger(__name__)


class OpenAlexScraper:
    """OpenAlex research paper scraper with zip file creation."""
    
    def __init__(self, config: ScraperConfig):
        """Initialize scraper with configuration."""
        self.config = config
        self.session = self._make_session()

    def _make_session(self) -> requests.Session:
        """Create and configure requests session with headers."""
        s = requests.Session()
        s.headers.update({
            "User-Agent": self.config.user_agent,
            "Referer": self.config.referer
        })
        return s

    def fetch_works(self, page: int) -> dict:
        """Query OpenAlex /works with filters and return JSON."""
        filters = ["is_oa:true"]
        if self.config.min_citations is not None:
            citation_threshold = self.config.min_citations - 1
            filters.append(f"cited_by_count:>{citation_threshold}")
        filter_str = ",".join(filters)

        params = {
            "filter": filter_str,
            "search": self.config.topic,
            "per-page": self.config.per_page,
            "page": page,
        }
        
        logger.info(f"Fetching works page {page} for topic: {self.config.topic}")
        r = self.session.get(self.config.api_base, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def extract_entries(self, works_json: dict) -> List[dict]:
        """Extract PDF URLs and metadata from works JSON."""
        entries = []
        for w in works_json.get("results", []):
            oa = w.get("best_oa_location") or {}
            pdf = oa.get("pdf_url")
            if not pdf:
                continue
            
            # Create a safe filename from title and ID
            title = w.get("title", "unknown").replace("/", "_").replace("\\", "_")[:100]
            openalex_id = w.get("id", "").split("/")[-1] if w.get("id") else "unknown"
            
            entries.append({
                "pdf_url": pdf,
                "host_type": oa.get("host_type"),
                "doi": w.get("doi"),
                "title": title,
                "openalex_id": openalex_id,
                "filename": f"{openalex_id}_{title}.pdf"
            })
        return entries

    def fetch_unpaywall(self, doi: str) -> Optional[str]:
        """Query Unpaywall for a public PDF URL via DOI."""
        if not doi or not self.config.email:
            return None
        url = f"{self.config.unpaywall_api}/{doi}"
        try:
            r = requests.get(url, params={"email": self.config.email}, timeout=20)
            if r.status_code != 200:
                return None
            data = r.json()
            loc = data.get("best_oa_location") or {}
            return loc.get("url_for_pdf")
        except Exception as e:
            logger.warning(f"Unpaywall request failed for DOI {doi}: {e}")
            return None

    def download_pdf(self, entry: Dict) -> Optional[str]:
        """Download PDF from entry URL with Unpaywall fallback."""
        os.makedirs(self.config.outdir, exist_ok=True)
        
        # Use the safe filename from the entry
        filename = entry.get("filename", entry["pdf_url"].split("/")[-1].split("?")[0])
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        
        path = os.path.join(self.config.outdir, filename)
        if os.path.exists(path):
            logger.info(f"File already exists: {filename}")
            return path

        try:
            logger.info(f"Downloading: {filename}")
            r = self.session.get(entry["pdf_url"], stream=True, timeout=60)
            r.raise_for_status()
        except requests.HTTPError as e:
            if e.response.status_code in (403, 429):
                fallback = self.fetch_unpaywall(entry.get("doi"))
                if fallback:
                    logger.info(f"Publisher blocked. Retrying via Unpaywall: {fallback}")
                    try:
                        r = self.session.get(fallback, stream=True, timeout=60)
                        r.raise_for_status()
                    except Exception as fallback_error:
                        logger.error(f"Unpaywall fallback failed for {filename}: {fallback_error}")
                        return None
                else:
                    logger.error(f"No Unpaywall fallback available for {filename}")
                    return None
            else:
                logger.error(f"HTTP error downloading {filename}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None

        try:
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            logger.info(f"Successfully downloaded: {filename}")
            return path
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")
            return None

    def create_zip_file(self, downloaded_files: List[str]) -> Optional[str]:
        """Create a zip file from downloaded PDFs."""
        try:
            # Create a unique zip filename
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            safe_topic = "".join(c for c in self.config.topic[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
            zip_filename = f"research_papers_{safe_topic}_{timestamp}_{unique_id}.zip"
            
            # Create downloads directory if it doesn't exist
            downloads_dir = Path(self.config.downloads_dir)
            downloads_dir.mkdir(parents=True, exist_ok=True)
            
            zip_path = downloads_dir / zip_filename
            
            logger.info(f"Creating zip file: {zip_path}")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in downloaded_files:
                    if os.path.exists(file_path):
                        # Add file to zip with just the filename (not full path)
                        arcname = os.path.basename(file_path)
                        zipf.write(file_path, arcname)
                        logger.info(f"Added to zip: {arcname}")
            
            logger.info(f"Successfully created zip file: {zip_path}")
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"Error creating zip file: {e}")
            return None

    def cleanup_downloaded_files(self, downloaded_files: List[str]) -> None:
        """Clean up downloaded PDF files."""
        try:
            for file_path in downloaded_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up: {os.path.basename(file_path)}")
            
            # Also try to remove the download directory if it's empty
            try:
                if os.path.exists(self.config.outdir):
                    os.rmdir(self.config.outdir)
                    logger.info(f"Removed empty directory: {self.config.outdir}")
            except OSError:
                # Directory not empty, that's fine
                pass
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def scrape_and_create_zip(self, cleanup_pdfs: bool = True) -> Tuple[bool, str, List[str], Optional[str]]:
        """
        Scrape research papers and create a downloadable zip file.
        
        Args:
            cleanup_pdfs: Whether to delete the individual PDF files after creating the zip
        
        Returns:
            Tuple of (success, message_or_error, downloaded_files, zip_file_path)
        """
        try:
            logger.info(f"Starting scrape for research area: {self.config.topic}")
            entries = []

            # Fetch papers from OpenAlex
            for pg in range(1, self.config.pages + 1):
                try:
                    js = self.fetch_works(pg)
                    es = self.extract_entries(js)
                    logger.info(f"[Page {pg}] Found {len(es)} PDF entries")
                    entries.extend(es)
                except Exception as e:
                    logger.error(f"Error fetching page {pg}: {e}")
                    continue

            if not entries:
                return False, "No research papers found for the given topic", [], None

            logger.info(f"Total PDF entries found: {len(entries)}")

            # Download papers
            downloaded_files = []
            with ThreadPoolExecutor(max_workers=self.config.workers) as ex:
                futures = [
                    ex.submit(self.download_pdf, e)
                    for e in entries
                ]
                for i, f in enumerate(futures):
                    try:
                        path = f.result()
                        if path:
                            downloaded_files.append(path)
                            logger.info(f"Downloaded ({i+1}/{len(entries)}): {os.path.basename(path)}")
                    except Exception as err:
                        logger.error(f"Failed to download entry {i+1}: {err}")

            if not downloaded_files:
                return False, "Failed to download any research papers", [], None

            logger.info(f"Successfully downloaded {len(downloaded_files)} papers")

            # Create zip file
            zip_path = self.create_zip_file(downloaded_files)
            if zip_path:
                logger.info(f"Successfully created zip file: {zip_path}")
                
                # Clean up individual PDF files if requested
                if cleanup_pdfs:
                    logger.info("Cleaning up downloaded PDF files...")
                    self.cleanup_downloaded_files(downloaded_files)
                
                return True, f"Successfully found and packaged {len(downloaded_files)} research papers", [os.path.basename(f) for f in downloaded_files], zip_path
            else:
                return False, "Failed to create zip file", [os.path.basename(f) for f in downloaded_files], None

        except Exception as e:
            logger.error(f"Error in scrape_and_create_zip: {e}")
            return False, f"Scraping failed: {str(e)}", [], None 