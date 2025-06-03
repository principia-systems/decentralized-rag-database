# Create new file: descidb/server/app.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os
import sys
import itertools
from fastapi.middleware.cors import CORSMiddleware

# Import your entry points
from descidb.query.evaluation_agent import EvaluationAgent
from descidb.utils.gdrive_scraper import scrape_gdrive_pdfs

# Setup FastAPI app
app = FastAPI(
    title="DeSciDB API",
    description="API for DeSciDB - a decentralized RAG database",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Define request/response models
class EvaluationRequest(BaseModel):
    query: str
    collections: List[str]
    db_path: Optional[str] = None
    model_name: str = "openai/gpt-3.5-turbo"

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

class IngestGDriveResponse(BaseModel):
    success: bool
    message: str
    downloaded_files: List[str]
    total_files: int
    processing_combinations: List[str]

@app.post("/api/evaluate")
async def evaluate_endpoint(request: EvaluationRequest):
    """Endpoint for evaluation (maps to run_evaluation.sh)"""
    try:
        # Initialize evaluation agent
        agent = EvaluationAgent(model_name=request.model_name)
        # Run query on collections
        results_file = agent.query_collections(
            query=request.query,
            collection_names=request.collections,
            db_path=request.db_path,
        )
        
        # Read the results from the JSON file
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/gdrive", response_model=IngestGDriveResponse)
async def ingest_gdrive_pdfs(request: IngestGDriveRequest):
    """
    Ingest PDFs from a public Google Drive folder.
    
    This endpoint scrapes all PDFs from a public Google Drive folder
    and downloads them to the papers directory in the root.
    It also creates a Cartesian product of all converter, chunker, and embedder combinations.
    """
    try:
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
        
        # Check if all provided converters are valid
        invalid_converters = [c for c in request.converters if c not in valid_converters]
        if invalid_converters:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid converters: {invalid_converters}. Valid options: {valid_converters}"
            )
        
        # Check if all provided chunkers are valid
        invalid_chunkers = [c for c in request.chunkers if c not in valid_chunkers]
        if invalid_chunkers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chunkers: {invalid_chunkers}. Valid options: {valid_chunkers}"
            )
        
        # Check if all provided embedders are valid
        invalid_embedders = [e for e in request.embedders if e not in valid_embedders]
        if invalid_embedders:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid embedders: {invalid_embedders}. Valid options: {valid_embedders}"
            )
        
        # Create Cartesian product of all combinations
        combinations = list(itertools.product(request.converters, request.chunkers, request.embedders))
        processing_combinations = [f"{conv}_{chunk}_{embed}" for conv, chunk, embed in combinations]
        
        # Download PDFs from Google Drive to papers folder
        downloaded_files = scrape_gdrive_pdfs(drive_url=request.drive_url)
        
        if not downloaded_files:
            return IngestGDriveResponse(
                success=False,
                message="No PDF files found or downloaded from the Google Drive folder",
                downloaded_files=[],
                total_files=0,
                processing_combinations=processing_combinations
            )
        
        return IngestGDriveResponse(
            success=True,
            message=f"Successfully downloaded {len(downloaded_files)} PDF files to papers/ directory. Created {len(processing_combinations)} processing combinations.",
            downloaded_files=downloaded_files,
            total_files=len(downloaded_files),
            processing_combinations=processing_combinations
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting PDFs: {str(e)}")

@app.get("/api/papers")
async def list_papers():
    """List all PDF files in the papers directory."""
    try:
        papers_dir = "papers"
        if not os.path.exists(papers_dir):
            return {"papers": [], "total": 0}
        
        pdf_files = [
            f for f in os.listdir(papers_dir) 
            if f.lower().endswith('.pdf')
        ]
        
        return {
            "papers": pdf_files,
            "total": len(pdf_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing papers: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)