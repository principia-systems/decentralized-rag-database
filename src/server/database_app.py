from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
from src.utils.logging_utils import get_user_logger
from src.query.evaluation_agent import EvaluationAgent
from src.reranking.cross_encoder import CrossEncoderRanker
from fastapi import FastAPI, HTTPException
import json
from fastapi.middleware.cors import CORSMiddleware
from src.db.db_creator_main import create_user_database
import os
import uuid
from datetime import datetime
import math

# Define PROJECT_ROOT
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Setup FastAPI app
app = FastAPI(
    title="Database API",
    description="Database API - handles database operations",
    version="0.1.0",
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


class WhitelistRequest(BaseModel):
    requester_email: str
    target_email: str


class WhitelistResponse(BaseModel):
    success: bool
    message: str
    requester_email: str
    target_email: str


class WhitelistInfoResponse(BaseModel):
    success: bool
    user_email: str
    whitelisted_users: List[str]  # Users this person has whitelisted
    whitelisted_by: List[str]  # Users who have whitelisted this person


class WhitelistRemoveRequest(BaseModel):
    requester_email: str
    target_email: str


class RerankRequest(BaseModel):
    user_email: str
    query: str
    items: List[str]
    model_preset: str = "msmarco-MiniLM-L-6-v2"
    batch_size: int = 16
    max_length: Optional[int] = 256
    top_k: Optional[int] = None
    descending: bool = True
    device: Optional[str] = "cpu"  # default to CPU for consistent behavior


def get_user_temp_dir(user_email: str) -> Path:
    """Get the temp directory path for a user"""
    return PROJECT_ROOT / "temp" / user_email


def ensure_user_temp_dir(user_email: str) -> Path:
    """Ensure user temp directory exists and return its path"""
    user_temp_dir = get_user_temp_dir(user_email)
    os.makedirs(user_temp_dir, exist_ok=True)
    return user_temp_dir


def add_to_whitelist_file(file_path: Path, email: str) -> bool:
    """Add an email to a whitelist file if not already present"""
    existing_emails = []

    # Read existing emails if file exists
    if file_path.exists():
        try:
            with open(file_path, "r") as f:
                existing_emails = [
                    line.strip() for line in f.readlines() if line.strip()
                ]
        except Exception:
            # If file is corrupted, start fresh
            existing_emails = []

    # Add email if not already present
    if email not in existing_emails:
        existing_emails.append(email)

        # Write back to file
        try:
            with open(file_path, "w") as f:
                for email_entry in existing_emails:
                    f.write(f"{email_entry}\n")
            return True
        except Exception:
            return False

    return True  # Email already exists, which is fine


def remove_from_whitelist_file(file_path: Path, email: str) -> bool:
    """Remove an email from a whitelist file if it exists"""
    if not file_path.exists():
        return False  # File doesn't exist, nothing to remove

    try:
        with open(file_path, "r") as f:
            existing_emails = [line.strip() for line in f.readlines() if line.strip()]

        # Filter out the email if it exists
        was_present = email in existing_emails
        updated_emails = [e for e in existing_emails if e != email]

        # Write back to file
        with open(file_path, "w") as f:
            for email_entry in updated_emails:
                f.write(f"{email_entry}\n")

        return was_present
    except Exception:
        return False


def get_whitelist_from_file(file_path: Path) -> List[str]:
    """Get list of emails from whitelist file"""
    if not file_path.exists():
        return []

    try:
        with open(file_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except Exception:
        return []


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    print("PROJECT_ROOT", PROJECT_ROOT)
    return {"status": "healthy", "service": "database"}


@app.post("/api/whitelist/add", response_model=WhitelistResponse)
async def add_to_whitelist(request: WhitelistRequest):
    """
    Add a user to another user's whitelist.
    This creates a bidirectional relationship:
    - Adds target_email to requester's whitelisted_users.txt
    - Adds requester_email to target's whitelisted_by.txt
    """
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.requester_email, "whitelist_add")

    try:
        user_logger.info(
            f"Processing whitelist request: {request.requester_email} -> {request.target_email}"
        )

        # Validate that emails are different
        if request.requester_email == request.target_email:
            raise HTTPException(status_code=400, detail="Cannot whitelist yourself")

        # Ensure both users' temp directories exist
        requester_temp_dir = ensure_user_temp_dir(request.requester_email)
        target_temp_dir = ensure_user_temp_dir(request.target_email)

        # Define file paths
        requester_whitelisted_users_file = requester_temp_dir / "whitelisted_users.txt"
        target_whitelisted_by_file = target_temp_dir / "whitelisted_by.txt"

        # Add target to requester's whitelisted users
        success1 = add_to_whitelist_file(
            requester_whitelisted_users_file, request.target_email
        )
        if not success1:
            raise HTTPException(
                status_code=500, detail="Failed to update requester's whitelist file"
            )

        # Add requester to target's whitelisted by
        success2 = add_to_whitelist_file(
            target_whitelisted_by_file, request.requester_email
        )
        if not success2:
            raise HTTPException(
                status_code=500, detail="Failed to update target's whitelist file"
            )

        user_logger.info(
            f"Successfully whitelisted {request.target_email} for {request.requester_email}"
        )

        return WhitelistResponse(
            success=True,
            message=f"Successfully whitelisted {request.target_email}",
            requester_email=request.requester_email,
            target_email=request.target_email,
        )

    except HTTPException:
        raise
    except Exception as e:
        user_logger.error(f"Error in whitelist operation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing whitelist request: {str(e)}"
        )


@app.get("/api/whitelist/{user_email}", response_model=WhitelistInfoResponse)
async def get_whitelist_info(user_email: str):
    """
    Get whitelist information for a user:
    - Users they have whitelisted
    - Users who have whitelisted them
    """
    # Create user-specific logger for this request
    user_logger = get_user_logger(user_email, "whitelist_info")

    try:
        user_logger.info(f"Retrieving whitelist info for: {user_email}")

        # Ensure user temp directory exists
        user_temp_dir = ensure_user_temp_dir(user_email)

        # Define file paths
        whitelisted_users_file = user_temp_dir / "whitelisted_users.txt"
        whitelisted_by_file = user_temp_dir / "whitelisted_by.txt"

        # Get lists from files
        whitelisted_users = get_whitelist_from_file(whitelisted_users_file)
        whitelisted_by = get_whitelist_from_file(whitelisted_by_file)

        user_logger.info(
            f"Retrieved {len(whitelisted_users)} whitelisted users and {len(whitelisted_by)} whitelisted by for {user_email}"
        )

        return WhitelistInfoResponse(
            success=True,
            user_email=user_email,
            whitelisted_users=whitelisted_users,
            whitelisted_by=whitelisted_by,
        )

    except Exception as e:
        user_logger.error(f"Error retrieving whitelist info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving whitelist info: {str(e)}"
        )


@app.post("/api/whitelist/remove", response_model=WhitelistResponse)
async def remove_from_whitelist(request: WhitelistRemoveRequest):
    """
    Remove a user from another user's whitelist.
    This removes the bidirectional relationship:
    - Removes target_email from requester's whitelisted_users.txt
    - Removes requester_email from target's whitelisted_by.txt
    """
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.requester_email, "whitelist_remove")

    try:
        user_logger.info(
            f"Processing whitelist removal: {request.requester_email} -> {request.target_email}"
        )

        # Validate that emails are different
        if request.requester_email == request.target_email:
            raise HTTPException(
                status_code=400, detail="Cannot remove yourself from whitelist"
            )

        # Get temp directories (don't create if they don't exist)
        requester_temp_dir = get_user_temp_dir(request.requester_email)
        target_temp_dir = get_user_temp_dir(request.target_email)

        # Define file paths
        requester_whitelisted_users_file = requester_temp_dir / "whitelisted_users.txt"
        target_whitelisted_by_file = target_temp_dir / "whitelisted_by.txt"

        # Remove target from requester's whitelisted users
        success1 = remove_from_whitelist_file(
            requester_whitelisted_users_file, request.target_email
        )

        # Remove requester from target's whitelisted by
        success2 = remove_from_whitelist_file(
            target_whitelisted_by_file, request.requester_email
        )

        if success1 or success2:  # Success if we removed from at least one file
            user_logger.info(
                f"Successfully removed {request.target_email} from {request.requester_email}'s whitelist"
            )
            message = f"Successfully removed {request.target_email} from whitelist"
        else:
            message = f"{request.target_email} was not in the whitelist"

        return WhitelistResponse(
            success=True,
            message=message,
            requester_email=request.requester_email,
            target_email=request.target_email,
        )

    except HTTPException:
        raise
    except Exception as e:
        user_logger.error(f"Error in whitelist removal: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing whitelist removal: {str(e)}"
        )


class EvaluationRequest(BaseModel):
    query: str
    collections: Optional[List[str]] = None  # If None, auto-discovers collections
    db_path: Optional[str] = None
    model_name: str = "openai/gpt-4o-mini"
    user_email: str


class EvaluationOption(BaseModel):
    id: str
    content: str
    collection_name: Optional[str] = None
    score: Optional[int] = None  # For scoring mode
    rank: Optional[int] = None  # For ranking mode


class StoreEvaluationRequest(BaseModel):
    user_email: str
    query: str
    mode: str  # "manual", "scoring", or "ranking"
    options: List[EvaluationOption]
    selected_option_id: str
    chat_id: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@app.post("/api/evaluation/store")
async def store_evaluation(request: StoreEvaluationRequest):
    """Store evaluation data for analysis"""
    # Create user-specific logger for this request
    user_logger = get_user_logger(request.user_email, "evaluation_storage")

    try:
        # Create storage directory if it doesn't exist
        storage_dir = PROJECT_ROOT / "storage"
        storage_dir.mkdir(exist_ok=True)

        # Determine which file to use based on mode
        mode_to_file = {
            "manual": "manual_evaluations.json",
            "scoring": "scoring_evaluations.json",
            "ranking": "ranking_evaluations.json",
        }

        if request.mode not in mode_to_file:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

        file_path = storage_dir / mode_to_file[request.mode]

        # Create evaluation record with appropriate schema
        evaluation_record = {
            "id": str(uuid.uuid4()),
            "timestamp": request.timestamp or datetime.utcnow().timestamp(),
            "user_email": request.user_email,
            "chat_id": request.chat_id,
            "query": request.query,
            "mode": request.mode,
            "selected_option_id": request.selected_option_id,
            "options": [],
        }

        # Format options based on mode
        for option in request.options:
            option_data = {
                "id": option.id,
                "content": option.content,  # Truncate long content
                "collection_name": option.collection_name,
            }

            if request.mode == "scoring" and option.score is not None:
                option_data["score"] = option.score
            elif request.mode == "ranking" and option.rank is not None:
                option_data["rank"] = option.rank

            evaluation_record["options"].append(option_data)

        # Add metadata if provided
        if request.metadata:
            evaluation_record["metadata"] = request.metadata

        # Load existing data or create new list
        evaluations = []
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    evaluations = json.load(f)
            except json.JSONDecodeError:
                user_logger.warning(
                    f"Could not parse existing file {file_path}, starting fresh"
                )
                evaluations = []

        # Append new evaluation
        evaluations.append(evaluation_record)

        # Write back to file
        with open(file_path, "w") as f:
            json.dump(evaluations, f, indent=2)

        user_logger.info(
            f"Stored {request.mode} evaluation for user {request.user_email}"
        )

        return {
            "success": True,
            "evaluation_id": evaluation_record["id"],
            "message": f"Evaluation stored successfully in {mode_to_file[request.mode]}",
        }

    except HTTPException:
        raise
    except Exception as e:
        user_logger.error(f"Error storing evaluation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error storing evaluation: {str(e)}"
        )


@app.post("/api/v1/user/evaluate")
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
        with open(results_file, "r") as f:
            results = json.load(f)

        user_logger.info("Successfully completed evaluation query")
        return results
    except Exception as e:
        user_logger.error(f"Error in evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rerank")
async def rerank_endpoint(request: RerankRequest):
    """Rerank a list of items for a query using a cross-encoder model."""
    user_logger = get_user_logger(request.user_email, "rerank")
    try:
        user_logger.info(
            f"Reranking {len(request.items)} items using {request.model_preset}"
        )

        ranker = CrossEncoderRanker.from_preset(
            request.model_preset,
            device=request.device,
            batch_size=request.batch_size,
            max_length=request.max_length,
            user_email=request.user_email,
        )

        ranked_pairs = ranker.rank_and_sort(
            request.query,
            request.items,
            top_k=request.top_k,
            descending=request.descending,
            user_email=request.user_email,
        )

        results = []
        for text, score in ranked_pairs:
            safe_score = float(score)
            if not math.isfinite(safe_score):
                user_logger.warning(
                    "Non-finite score encountered (score=%s). Converting to 0.0 for JSON compliance.",
                    score,
                )
                safe_score = 0.0
            results.append({"item": text, "score": safe_score})
        user_logger.info("Successfully computed reranked results")
        return {"success": True, "results": results}
    except Exception as e:
        user_logger.error(f"Error during rerank: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during rerank: {str(e)}")


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
        with open(mappings_file, "w") as f:
            json.dump(request.mappings, f, indent=2)

        user_logger.info(f"Saved mappings to {mappings_file}")

        # Create the database using the existing functionality
        create_user_database(request.user_email)

        user_logger.info(
            f"Successfully created database for user: {request.user_email}"
        )

        return {
            "success": True,
            "message": f"Database created successfully for {request.user_email}",
            "user_email": request.user_email,
            "total_cids": len(request.mappings),
            "total_combinations": sum(
                len(combinations) for combinations in request.mappings.values()
            ),
        }
    except Exception as e:
        user_logger.error(f"Error creating database: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error creating database: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5003)
