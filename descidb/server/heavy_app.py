# Heavy FastAPI server for resource-intensive endpoints (ingestion, processing)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import itertools
import asyncio
import glob
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Import your entry points
from descidb.utils.gdrive_scraper import scrape_gdrive_pdfs
from descidb.core.processor_main import process_combination
from descidb.db.db_creator_main import create_user_database

# Setup FastAPI app
app = FastAPI(
    title="DeSciDB Heavy API",
    description="Heavy API for DeSciDB - handles resource-intensive processing tasks",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://coophive-wine.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Global lock for database creation to prevent concurrent write conflicts
_db_creation_lock = asyncio.Lock()

# Define request/response models
class IngestGDriveRequest(BaseModel):
    drive_url: str = Field(..., description="Public Google Drive folder URL")
    converters: Optional[List[str]] = Field(
        default=["markitdown"], 
        description="List of converters to use (marker, openai, markitdown)"
    )
    chunkers: Optional[List[str]] = Field(
        default=["paragraph"], 
        description="List of chunkers to use (paragraph, sentence, word, fixed_length)"
    )
    embedders: Optional[List[str]] = Field(
        default=["bge"], 
        description="List of embedders to use (openai, nvidia, bge)"
    )
    user_email: str

class IngestGDriveResponse(BaseModel):
    success: bool
    message: str
    downloaded_files: List[str]
    total_files: int
    processing_combinations: List[str]
    database_created: bool

class EmbedRequest(BaseModel):
    user_email: str

@app.post("/api/ingest/gdrive", response_model=IngestGDriveResponse)
async def ingest_gdrive_pdfs(request: IngestGDriveRequest):
    """
    Ingest PDFs from a public Google Drive folder.
    
    This endpoint scrapes all PDFs from a public Google Drive folder
    and downloads them to a user-specific papers directory.
    It also creates a Cartesian product of all converter, chunker, and embedder combinations.
    After processing, it automatically creates the user database.
    """
    try:
        print(f"[HEAVY] Processing Google Drive ingestion for user: {request.user_email}")
        print(f"[HEAVY] Drive URL: {request.drive_url}")
        
        # Validate the Google Drive URL
        if "drive.google.com" not in request.drive_url:
            raise HTTPException(
                status_code=400, 
                detail="Invalid Google Drive URL. Please provide a valid Google Drive folder link."
            )
        
        # Validate component lists
        valid_converters = ["marker", "openai", "markitdown"]
        valid_chunkers = ["paragraph", "sentence", "word", "fixed_length"]
        valid_embedders = ["openai", "nvidia", "bge"]
        
        # Validate requested components
        for converter in request.converters:
            if converter not in valid_converters:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid converter '{converter}'. Valid options: {valid_converters}"
                )
        
        for chunker in request.chunkers:
            if chunker not in valid_chunkers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid chunker '{chunker}'. Valid options: {valid_chunkers}"
                )
        
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
        
        print(f"[HEAVY] Will process {len(processing_combinations)} combinations")
        
        # Download PDFs from Google Drive to user-specific papers folder
        downloaded_files = scrape_gdrive_pdfs(
            drive_url=request.drive_url,
            download_dir=user_papers_dir
        )
        
        if not downloaded_files:
            return IngestGDriveResponse(
                success=False,
                message="No PDF files found or downloaded from the Google Drive folder",
                downloaded_files=[],
                total_files=0,
                processing_combinations=combination_strings,
                database_created=False
            )
        
        print(f"[HEAVY] Downloaded {len(downloaded_files)} files, starting processing...")
        
        # Process each combination
        processing_tasks = []
        for converter, chunker, embedder in processing_combinations:
            print(f"[HEAVY] Creating task for combination: {converter}_{chunker}_{embedder}")
            task = asyncio.create_task(
                process_combination(
                    converter=converter,
                    chunker=chunker,
                    embedder=embedder,
                    papers_list=downloaded_files,
                    user_papers_dir=user_papers_dir,
                    user_email=request.user_email
                )
            )
            processing_tasks.append(task)
        
        # Wait for all processing tasks to complete concurrently
        print(f"[HEAVY] Running {len(processing_tasks)} combinations concurrently...")
        results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Log any errors from the concurrent processing
        for i, result in enumerate(results):
            converter, chunker, embedder = processing_combinations[i]
            if isinstance(result, Exception):
                print(f"[HEAVY] Error processing combination {converter}_{chunker}_{embedder}: {str(result)}")
            else:
                print(f"[HEAVY] Successfully completed combination {converter}_{chunker}_{embedder}")
        
        print(f"[HEAVY] All processing combinations completed")
        
        # Clean up: Delete all PDFs from user's papers directory after processing
        try:
            print(f"[HEAVY] Cleaning up PDFs from user directory: {user_papers_dir}")
            pdf_files = glob.glob(os.path.join(user_papers_dir, "*.pdf"))
            deleted_count = 0
            for pdf_file in pdf_files:
                try:
                    os.remove(pdf_file)
                    deleted_count += 1
                    print(f"[HEAVY] Deleted: {os.path.basename(pdf_file)}")
                except Exception as delete_error:
                    print(f"[HEAVY] Error deleting {pdf_file}: {str(delete_error)}")
            
            print(f"[HEAVY] Successfully deleted {deleted_count} PDF files from user directory")
        except Exception as cleanup_error:
            print(f"[HEAVY] Error during PDF cleanup: {str(cleanup_error)}")
            # Don't fail the entire request if cleanup fails

        # Automatically create user database after processing completes
        database_created = False
        try:
            print(f"[HEAVY] Creating database for user: {request.user_email}")
            # Use async lock to prevent concurrent database creation conflicts
            async with _db_creation_lock:
                print(f"[HEAVY] Acquired lock for database creation: {request.user_email}")
                # Run database creation in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,  # Use default thread pool for I/O operations
                    create_user_database,
                    request.user_email
                )
            database_created = True
            print(f"[HEAVY] Successfully created database for user: {request.user_email}")
        except Exception as db_error:
            print(f"[HEAVY] Error creating user database: {str(db_error)}")
            # Don't fail the entire request if database creation fails, but log it

        return IngestGDriveResponse(
            success=True,
            message=f"Successfully processed {len(downloaded_files)} PDF files with {len(processing_combinations)} combinations, cleaned up papers/ directory, and {'created' if database_created else 'failed to create'} user database.",
            downloaded_files=downloaded_files,
            total_files=len(downloaded_files),
            processing_combinations=combination_strings,
            database_created=database_created
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting PDFs: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "heavy"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002) 