# Heavy FastAPI server for resource-intensive endpoints (ingestion, processing)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import itertools
import asyncio
import glob
import json
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import requests

# Import your entry points
from src.utils.gdrive_scraper import scrape_gdrive_pdfs
from src.core.processor_main import process_combination
from src.utils.logging_utils import get_user_logger
from src.utils.file_lock import load_jobs_safe, save_jobs_safe, reset_job_tracking_safe
from src.utils.file_lock import increment_job_progress_safe
# Setup FastAPI app
app = FastAPI(
    title="Heavy API",
    description="Heavy API - handles resource-intensive processing tasks",
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

# Configuration
DATABASE_SERVER_URL = os.getenv('DATABASE_SERVER_URL', 'http://localhost:5003')

# Global lock for database creation to prevent concurrent write conflicts
_db_creation_lock = asyncio.Lock()

# Job tracking functions are now handled by src.utils.file_lock module
# These functions are kept for backward compatibility but should not be used
def load_jobs():
    """DEPRECATED: Use load_jobs_safe() from src.utils.file_lock instead"""
    return load_jobs_safe()

def save_jobs(jobs_data):
    """DEPRECATED: Use save_jobs_safe() from src.utils.file_lock instead"""
    print("[HEAVY] WARNING: Using deprecated save_jobs(). Use file_lock.save_jobs_safe() instead.")
    return save_jobs_safe(jobs_data)

def increment_job_progress(user_email: str, increment: int):
    """Increment job progress for a user"""
    logger = get_user_logger(user_email, "job_progress")
    logger.info(f"Incrementing job progress for {user_email} by {increment}")
    return increment_job_progress_safe(user_email, increment)

async def create_database_via_server(user_email: str) -> bool:
    """
    Create database by calling the database server endpoint.
    
    Args:
        user_email: Email of the user for database creation
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = get_user_logger(user_email, "database_server_call")
    
    try:
        # Load mappings from user's temp directory
        user_temp_path = PROJECT_ROOT / "temp" / user_email
        mappings_file_path = user_temp_path / "mappings.json"
        
        if not mappings_file_path.exists():
            logger.error(f"No mappings.json file found for user {user_email}")
            return False
            
        with open(mappings_file_path, "r") as file:
            mappings = json.load(file)
            
        if not mappings:
            logger.warning(f"Empty mappings file for user {user_email}")
            return False
            
        logger.info(f"Loaded {len(mappings)} CID mappings for database creation")
        
        # Prepare request data
        request_data = {
            "user_email": user_email,
            "mappings": mappings,
            "model_name": "openai/gpt-4o-mini"
        }
        
        # Make request to database server
        logger.info(f"Calling database server at {DATABASE_SERVER_URL}/api/database/create")
        response = requests.post(
            f"{DATABASE_SERVER_URL}/api/database/create",
            json=request_data,
            timeout=1800  # 30 minute timeout for database creation
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Database creation successful: {result.get('message', 'Success')}")
        logger.info(f"Processed {result.get('total_cids', 0)} CIDs with {result.get('total_combinations', 0)} combinations")
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call database server: {e}")
        return False
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading mappings file: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in database creation: {e}")
        return False


# Background processing function
async def background_processing(
    processing_combinations: List[tuple],
    downloaded_files: List[str],
    user_papers_dir: str,
    user_email: str
):
    """Handle all processing in the background"""
    # Create user-specific logger for this processing session
    logger = get_user_logger(user_email, "background_processing")
    
    try:
        logger.info(f"Starting background processing for {user_email}")
        
        # Process each combination
        processing_tasks = []
        for converter, chunker, embedder in processing_combinations:
            logger.info(f"Creating task for combination: {converter}_{chunker}_{embedder}")
            task = asyncio.create_task(
                process_combination(
                    converter=converter,
                    chunker=chunker,
                    embedder=embedder,
                    papers_list=downloaded_files,
                    user_papers_dir=user_papers_dir,
                    user_email=user_email
                )
            )
            processing_tasks.append(task)
        
        # Wait for all processing tasks to complete concurrently
        logger.info(f"Running {len(processing_tasks)} combinations concurrently...")
        results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Log any errors from the concurrent processing
        for i, result in enumerate(results):
            converter, chunker, embedder = processing_combinations[i]
            if isinstance(result, Exception):
                logger.error(f"Error processing combination {converter}_{chunker}_{embedder}: {str(result)}")
            else:
                logger.info(f"Successfully completed combination {converter}_{chunker}_{embedder}")
        
        logger.info("All processing combinations completed")
        
        # Clean up: Delete all PDFs from user's papers directory after processing
        try:
            logger.info(f"Cleaning up PDFs from user directory: {user_papers_dir}")
            pdf_files = glob.glob(os.path.join(user_papers_dir, "*.pdf"))
            deleted_count = 0
            for pdf_file in pdf_files:
                try:
                    os.remove(pdf_file)
                    deleted_count += 1
                    logger.debug(f"Deleted: {os.path.basename(pdf_file)}")
                except Exception as delete_error:
                    logger.error(f"Error deleting {pdf_file}: {str(delete_error)}")
            
            logger.info(f"Successfully deleted {deleted_count} PDF files from user directory")
        except Exception as cleanup_error:
            logger.error(f"Error during PDF cleanup: {str(cleanup_error)}")

        # Create databases via database server
        logger.info(f"Starting database creation for {user_email}")
        try:
            # Use the async lock to prevent concurrent database creation issues
            async with _db_creation_lock:
                success = await create_database_via_server(user_email)
                logger.info(f"Database creation successful: {success}")
                increment_job_progress(user_email, 10)
                if success:
                    logger.info(f"Successfully completed database creation for {user_email}")
                else:
                    logger.error(f"Database creation failed for {user_email}")
        except Exception as db_error:
            increment_job_progress(user_email, 10)
            logger.error(f"Error during database creation for {user_email}: {str(db_error)}")
            # Don't raise the exception - processing was successful even if DB creation failed

        logger.info(f"Background processing completed for {user_email}")
        
    except Exception as e:
        increment_job_progress(user_email, 10)
        logger.error(f"Error in background processing for {user_email}: {str(e)}")


# Define request/response models
class IngestGDriveRequest(BaseModel):
    drive_url: str = Field(..., description="Public Google Drive folder URL")
    converters: Optional[List[str]] = Field(
        default=["markitdown"], 
        description="List of converters to use (marker, openai, markitdown)"
    )
    chunkers: Optional[List[str]] = Field(
        default=["recursive"], 
        description="List of chunkers to use (fixed_length, recursive, markdown_aware, semantic_split)"
    )
    embedders: Optional[List[str]] = Field(
        default=["bge"], 
        description="List of embedders to use (openai, nvidia, bge, bgelarge, e5large)"
    )
    user_email: str

class IngestGDriveResponse(BaseModel):
    success: bool
    message: str
    downloaded_files: List[str]
    total_files: int
    processing_combinations: List[str]
    processing_started: bool

class EmbedRequest(BaseModel):
    user_email: str

@app.post("/api/ingest/gdrive", response_model=IngestGDriveResponse)
async def ingest_gdrive_pdfs(request: IngestGDriveRequest):
    """
    Ingest PDFs from a public Google Drive folder.
    
    This endpoint scrapes all PDFs from a public Google Drive folder,
    downloads them to a user-specific papers directory, and starts
    background processing for all converter, chunker, and embedder combinations.
    Returns immediately after starting the processing.
    """
    # Create user-specific logger for this request
    logger = get_user_logger(request.user_email, "gdrive_ingestion")
    
    try:
        logger.info(f"Processing Google Drive ingestion for user: {request.user_email}")
        logger.info(f"Drive URL: {request.drive_url}")
        
        # Validate the Google Drive URL
        if "drive.google.com" not in request.drive_url:
            raise HTTPException(
                status_code=400, 
                detail="Invalid Google Drive URL. Please provide a valid Google Drive folder link."
            )
        
        # Validate component lists
        valid_converters = ["marker", "openai", "markitdown"]
        valid_chunkers = ["fixed_length", "recursive", "markdown_aware", "semantic_split"]
        valid_embedders = ["openai", "nvidia", "bge", "bgelarge", "e5large"]
        
        # Validate requested components
        for converter in request.converters:
            if converter not in valid_converters:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid converter '{converter}'. Valid options: {valid_converters}"
                )
        logger.debug(f"Valid chunkers: {valid_chunkers}")
        for chunker in request.chunkers:
            if chunker not in valid_chunkers:
                logger.error(f"Invalid chunker: {chunker}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid chunker '{chunker}'. Valid options: {valid_chunkers}"
                )
        logger.debug(f"Validated chunkers: {request.chunkers}")
        
        for embedder in request.embedders:
            if embedder not in valid_embedders:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid embedder '{embedder}'. Valid options: {valid_embedders}"
                )
        
        papers_directory = PROJECT_ROOT / "papers"

        # Create user-specific directory structure
        user_papers_dir = os.path.join(str(papers_directory), request.user_email)
        
        # Create directories if they don't exist
        os.makedirs(user_papers_dir, exist_ok=True)
        
        # Generate Cartesian product of all combinations
        processing_combinations = list(itertools.product(
            request.converters,
            request.chunkers,
            request.embedders
        ))
        
        combination_strings = [
            f"{converter}_{chunker}_{embedder}"
            for converter, chunker, embedder in processing_combinations
        ]
        
        logger.info(f"Will process {len(processing_combinations)} combinations")
        
        # Download PDFs from Google Drive to user-specific papers folder
        downloaded_files = scrape_gdrive_pdfs(
            drive_url=request.drive_url,
            download_dir=user_papers_dir,
            user_email=request.user_email
        )
        
        if not downloaded_files:
            logger.warning("No PDF files found or downloaded from the Google Drive folder")
            return IngestGDriveResponse(
                success=False,
                message="No PDF files found or downloaded from the Google Drive folder",
                downloaded_files=[],
                total_files=0,
                processing_combinations=combination_strings,
                processing_started=False
            )
        
        logger.info(f"Downloaded {len(downloaded_files)} files, starting background processing...")
        
        # Reset job tracking for this new batch - using thread-safe version
        total_jobs = (len(processing_combinations) * len(downloaded_files)) + 10
        success = reset_job_tracking_safe(request.user_email, total_jobs)
        if not success:
            logger.warning(f"Failed to reset job tracking for {request.user_email}")
        else:
            logger.info(f"Initialized job tracking for {request.user_email}: 0/{total_jobs}")

        # Start background processing (don't await)
        asyncio.create_task(background_processing(
            processing_combinations=processing_combinations,
            downloaded_files=downloaded_files,
            user_papers_dir=user_papers_dir,
            user_email=request.user_email
        ))
        
        logger.info("Background processing started, returning response immediately")

        # Return immediately
        return IngestGDriveResponse(
            success=True,
            message=f"Processing started for {len(downloaded_files)} PDF files with {len(processing_combinations)} combinations. Check status for progress updates.",
            downloaded_files=downloaded_files,
            total_files=len(downloaded_files),
            processing_combinations=combination_strings,
            processing_started=True
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error ingesting PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ingesting PDFs: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "heavy"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002) 