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
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent.parent.parent

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

class EmbedRequest(BaseModel):
    user_email: str

@app.post("/api/ingest/gdrive", response_model=IngestGDriveResponse)
async def ingest_gdrive_pdfs(request: IngestGDriveRequest):
    """
    Ingest PDFs from a public Google Drive folder.
    
    This endpoint scrapes all PDFs from a public Google Drive folder
    and downloads them to a user-specific papers directory.
    It also creates a Cartesian product of all converter, chunker, and embedder combinations.
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
                processing_combinations=combination_strings
            )
        
        print(f"[HEAVY] Downloaded {len(downloaded_files)} files, starting processing...")
        
        # Process each combination
        for converter, chunker, embedder in processing_combinations:
            try:
                print(f"[HEAVY] Processing combination: {converter}_{chunker}_{embedder}")
                # Call the processor for this specific combination with user-specific db path
                await process_combination(
                    converter=converter,
                    chunker=chunker,
                    embedder=embedder,
                    papers_list=downloaded_files,
                    user_papers_dir=user_papers_dir,
                    user_email=request.user_email
                )
                
                # Create user database after processing
            except Exception as e:
                # Log the error but continue with other combinations
                print(f"[HEAVY] Error processing combination {converter}_{chunker}_{embedder}: {str(e)}")
        
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

        return IngestGDriveResponse(
            success=True,
            message=f"Successfully processed {len(downloaded_files)} PDF files with {len(processing_combinations)} combinations and cleaned up papers/ directory.",
            downloaded_files=downloaded_files,
            total_files=len(downloaded_files),
            processing_combinations=combination_strings
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting PDFs: {str(e)}")

@app.post("/api/embed")
async def embed_endpoint(request: EmbedRequest):
    """Endpoint to create user database - resource intensive"""
    try:
        print(f"[HEAVY] Creating database for user: {request.user_email}")
        
        # Create user database
        create_user_database(request.user_email)
        
        return {
            "success": True,
            "message": f"Successfully created database for user: {request.user_email}",
            "user_email": request.user_email
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user database: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "heavy"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002) 