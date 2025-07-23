from pydantic import BaseModel
from typing import List, Optional, Dict
from src.utils.logging_utils import get_user_logger
from src.core.agent import EvaluationAgent
from src.core.config import PROJECT_ROOT
from fastapi import FastAPI, HTTPException
import json
from fastapi.middleware.cors import CORSMiddleware
from src.db.db_creator_main import create_user_database
import os

# Setup FastAPI app
app = FastAPI(
    title="Database API",
    description="Database API - handles database operations",
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

class CreateDatabaseRequest(BaseModel):
    user_email: str
    mappings: Dict[str, List[str]]  # CID -> list of database combinations
    model_name: str = "openai/gpt-4o-mini"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "database"}


class EvaluationRequest(BaseModel):
    query: str
    collections: Optional[List[str]] = None  # If None, auto-discovers collections
    db_path: Optional[str] = None
    model_name: str = "openai/gpt-4o-mini"
    user_email: str


@app.post("/api/evaluate")
async def evaluate_endpoint(request: EvaluationRequest):
    """Endpoint for evaluation - fast queries"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.user_email, "evaluation")
    
    try:
        user_logger.info(f"Evaluating query: {request.query}")
        user_logger.debug(f"DB Path: {request.db_path}")
        user_logger.debug(f"Model Name: {request.model_name}")
        user_logger.info(f"User Email: {request.user_email}")
        
        # Construct user-specific database path if not provided
        if request.db_path is None:
            base_db_path = PROJECT_ROOT / "src" / "database" / request.user_email
            user_db_path = str(base_db_path)
        else:
            user_db_path = request.db_path
        
        user_logger.info(f"Using database path: {user_db_path}")
        
        # Initialize evaluation agent
        agent = EvaluationAgent(model_name=request.model_name)

        # Run query on collections with user-specific database path
        results_file = agent.query_collections(
            query=request.query,
            db_path=user_db_path,
            user_email=request.user_email,
        )
        
        # Read the results from the JSON file
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        user_logger.info(f"Successfully completed evaluation query")
        return results
    except Exception as e:
        user_logger.error(f"Error in evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/database/create")
async def create_database(request: CreateDatabaseRequest):
    """Endpoint for creating a database from processing mappings"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.user_email, "database_creation")
    
    try:
        user_logger.info(f"Creating database for user: {request.user_email}")
        user_logger.info(f"Processing {len(request.mappings)} PDF CIDs")
        
        # Import database creation functionality
        
        user_temp_dir = PROJECT_ROOT / "temp" / request.user_email
        os.makedirs(user_temp_dir, exist_ok=True)
        
        mappings_file = user_temp_dir / "mappings.json"
        with open(mappings_file, 'w') as f:
            json.dump(request.mappings, f, indent=2)
        
        user_logger.info(f"Saved mappings to {mappings_file}")
        
        # Create the database using the existing functionality
        create_user_database(request.user_email)
        
        user_logger.info(f"Successfully created database for user: {request.user_email}")
        
        return {
            "success": True,
            "message": f"Database created successfully for {request.user_email}",
            "user_email": request.user_email,
            "total_cids": len(request.mappings),
            "total_combinations": sum(len(combinations) for combinations in request.mappings.values())
        }
    except Exception as e:
        user_logger.error(f"Error creating database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating database: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)