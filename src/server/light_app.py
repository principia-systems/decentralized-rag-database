# Light FastAPI server for quick endpoints (evaluation, status)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from typing import List, Dict
import json
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, Any 
import requests

# Import your entry points
from src.scraper.openalex_scraper import OpenAlexScraper
from src.scraper.config import ScraperConfig
from src.utils.logging_utils import get_user_logger
from src.utils.file_lock import load_jobs_safe
from src.utils.ipfs_utils import get_ipfs_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a thread pool executor for CPU-intensive tasks
_thread_pool = ThreadPoolExecutor(max_workers=2)

# Setup FastAPI app
app = FastAPI(
    title="Light API",
    description="Light API - handles quick queries and status checks",
    version="0.1.0"
)

# Add CORS middleware - Allow all origins for development with ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for ngrok compatibility
    allow_credentials=False,  # Must be False when allow_origins=["*"] 
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
WHITELIST_PATH = Path(__file__).parent / "whitelisted_emails.txt"

def _scrape_papers_sync(scraper, cleanup_pdfs):
    """
    Synchronous helper function to scrape papers.
    This runs in a separate thread to avoid blocking the event loop.
    """
    try:
        return scraper.scrape_and_create_zip(cleanup_pdfs)
    except Exception as e:
        return False, str(e), [], None

def cleanup_zip_file(zip_path: str):
    """Background task to clean up zip file after serving."""
    # Use system logger for cleanup operations
    system_logger = get_user_logger("system", "light_server")
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            system_logger.info(f"Cleaned up zip file: {zip_path}")
    except Exception as e:
        system_logger.warning(f"Error cleaning up zip file {zip_path}: {e}")

# Define request/response models
class UserStatusResponse(BaseModel):
    user_email: str
    total_papers: int
    completed_jobs: int
    completion_percentage: float
    papers_directory: str
    mappings_file_path: str

class EmailValidationRequest(BaseModel):
    email: EmailStr

class EmailValidationResponse(BaseModel):
    isValid: bool

class AddEmailRequest(BaseModel):
    email: EmailStr
    admin_key: str  # Simple admin key for security

class AddEmailResponse(BaseModel):
    success: bool
    message: str
    email: str

class ResearchScrapeRequest(BaseModel):
    research_area: str
    user_email: str

class BatchRetrievalRequest(BaseModel):
    embedding_cids: List[str]
    content_cids: List[str]
    user_email: str

class BatchRetrievalResponse(BaseModel):
    embeddings: Dict[str, List]  # CID -> embedding vector
    contents: Dict[str, str]     # CID -> content string
    failed_embeddings: List[str]  # CIDs that failed to retrieve
    failed_contents: List[str]    # CIDs that failed to retrieve

def load_whitelisted_emails() -> set:
    """Load whitelisted emails from the file"""
    if not WHITELIST_PATH.exists():
        return set()
    
    with open(WHITELIST_PATH, 'r') as f:
        emails = {line.strip() for line in f 
                 if line.strip() and not line.startswith('#')}
    return emails

@app.post("/api/auth/validate-email", response_model=EmailValidationResponse)
async def validate_email(request: EmailValidationRequest):
    """Validate if an email is whitelisted"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.email, "email_validation")
    
    try:
        whitelisted_emails = load_whitelisted_emails()
        email = request.email.lower()
        
        # Check if email is in whitelist
        is_valid = email in whitelisted_emails
        
        user_logger.info(f"Email validation request: {email} - Valid: {is_valid}")
        
        return EmailValidationResponse(isValid=is_valid)
    except Exception as e:
        user_logger.error(f"Error validating email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error validating email: {str(e)}")

@app.post("/api/auth/add-email", response_model=AddEmailResponse)
async def add_email_to_whitelist(request: AddEmailRequest):
    """Add an email to the whitelist (requires admin key)"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.email, "add_email")
    
    try:
        # In a real application, you would validate the admin_key
        # For this example, we'll just check if it's not empty
        if not request.admin_key or request.admin_key != os.getenv("ADMIN_KEY"):
            raise HTTPException(status_code=401, detail="Invalid admin key")

        whitelisted_emails = load_whitelisted_emails()
        email = request.email.lower()
        
        if email in whitelisted_emails:
            return AddEmailResponse(success=False, message=f"Email {email} is already whitelisted.", email=email)
        
        # Append the new email to the file instead of overwriting
        with open(WHITELIST_PATH, 'a') as f:
            f.write(f"\n{email}")
        
        user_logger.info(f"Email {email} added to whitelist.")
        return AddEmailResponse(success=True, message=f"Email {email} added to whitelist.", email=email)
    except HTTPException:
        raise
    except Exception as e:
        user_logger.error(f"Error adding email to whitelist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding email to whitelist: {str(e)}")

@app.post("/api/research/scrape")
async def scrape_research_papers(request: ResearchScrapeRequest, background_tasks: BackgroundTasks):
    """Scrape research papers from OpenAlex and return zip file"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.user_email, "research_scrape")
    
    try:
        user_logger.info(f"Starting research scrape for: {request.research_area}")
        user_logger.info(f"User email: {request.user_email}")
        
        # Validate research area
        if not request.research_area.strip():
            raise HTTPException(status_code=400, detail="Research area cannot be empty")
        
        # Create scraper configuration
        config = ScraperConfig.from_research_area(
            research_area=request.research_area.strip(),
            user_email=request.user_email
        )
        
        # Create scraper instance
        scraper = OpenAlexScraper(config)
        
        # Run scraping in the shared thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        success, result_message, downloaded_files, zip_path = await loop.run_in_executor(
            _thread_pool,
            _scrape_papers_sync,
            scraper,
            True  # cleanup_pdfs=True
        )
        
        if success and zip_path:
            user_logger.info(f"Successfully completed scraping for {request.user_email}")
            user_logger.info(f"Returning zip file: {zip_path}")
            
            # Schedule cleanup of the zip file after response is sent
            background_tasks.add_task(cleanup_zip_file, zip_path)
            
            # Get a clean filename for the download
            safe_topic = "".join(c for c in request.research_area[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
            download_filename = f"research_papers_{safe_topic}.zip"
            
            return FileResponse(
                path=zip_path,
                filename=download_filename,
                media_type='application/zip',
                headers={
                    "Content-Disposition": f"attachment; filename={download_filename}",
                    "X-Papers-Count": str(len(downloaded_files)),
                    "X-Research-Area": request.research_area[:100]
                }
            )
        else:
            user_logger.error(f"Scraping failed for {request.user_email}: {result_message}")
            raise HTTPException(status_code=404, detail=result_message)
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error scraping research papers: {str(e)}"
        user_logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/api/ipfs/batch-retrieve", response_model=BatchRetrievalResponse)
async def batch_retrieve_ipfs_data(request: BatchRetrievalRequest):
    """Batch retrieve embeddings and content from IPFS"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.user_email, "batch_retrieve")
    
    try:
        ipfs_client = get_ipfs_client()
        
        # Initialize response data
        embeddings = {}
        contents = {}
        failed_embeddings = []
        failed_contents = []
        
        user_logger.info(f"Starting batch retrieval for {len(request.embedding_cids)} embeddings and {len(request.content_cids)} contents")
        
        # Retrieve embeddings
        for cid in request.embedding_cids:
            try:
                content = ipfs_client.get_content(cid)
                embedding_vector = json.loads(content)
                embeddings[cid] = embedding_vector
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                user_logger.error(f"Failed to retrieve embedding for CID {cid}: {e}")
                failed_embeddings.append(cid)
        
        # Retrieve contents
        for cid in request.content_cids:
            try:
                content = ipfs_client.get_content(cid)
                contents[cid] = content
            except requests.exceptions.RequestException as e:
                user_logger.error(f"Failed to retrieve content for CID {cid}: {e}")
                failed_contents.append(cid)
        
        user_logger.info(f"Successfully retrieved {len(embeddings)}/{len(request.embedding_cids)} embeddings and {len(contents)}/{len(request.content_cids)} contents")
        
        return BatchRetrievalResponse(
            embeddings=embeddings,
            contents=contents,
            failed_embeddings=failed_embeddings,
            failed_contents=failed_contents
        )
        
    except Exception as e:
        user_logger.error(f"Error in batch retrieval: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch retrieval failed: {str(e)}")
    

@app.get("/api/status")
async def get_user_status(user_email: str):
    """Get processing status for a specific user - fast status check"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(user_email, "status_check")
    
    try:
        # Use thread-safe file locking for reading jobs.json
        jobs = load_jobs_safe()
        total_jobs, completed_jobs = jobs.get(user_email, [0, 0])
        completion_percentage = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
        user_logger.debug(f"Status check: {completed_jobs}/{total_jobs} jobs completed")
        
        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "completion_percentage": completion_percentage
        }
    except Exception as e:
        user_logger.error(f"Error getting user status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "light"}

@app.get("/api/evaluation/stats")
async def get_evaluation_stats(user_email: str):
    """Get evaluation statistics for a user"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(user_email, "evaluation_stats")
    
    try:
        storage_dir = PROJECT_ROOT / "storage"
        stats = {
            "manual": 0,
            "scoring": 0,
            "ranking": 0,
            "total": 0
        }
        
        # Count evaluations for each mode
        for mode, filename in [("manual", "manual_evaluations.json"), 
                               ("scoring", "scoring_evaluations.json"), 
                               ("ranking", "ranking_evaluations.json")]:
            file_path = storage_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        evaluations = json.load(f)
                        user_evaluations = [e for e in evaluations if e.get("user_email") == user_email]
                        stats[mode] = len(user_evaluations)
                        stats["total"] += len(user_evaluations)
                except Exception as e:
                    user_logger.warning(f"Could not read {filename}: {e}")
        
        return stats
        
    except Exception as e:
        user_logger.error(f"Error getting evaluation stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting evaluation stats: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001) 