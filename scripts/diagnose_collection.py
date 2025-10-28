#!/usr/bin/env python3
"""
Diagnostic script to investigate ChromaDB collection issues.
Usage: python scripts/diagnose_collection.py <user_email> <collection_name>
"""

import sys
from pathlib import Path
import chromadb

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def diagnose_collection(user_email: str, collection_name: str):
    """Diagnose issues with a specific collection."""
    
    logger.info(f"Diagnosing collection '{collection_name}' for user '{user_email}'")
    
    # Construct database path
    db_path = PROJECT_ROOT / "src" / "database" / user_email
    
    if not db_path.exists():
        logger.error(f"Database path does not exist: {db_path}")
        return
    
    logger.info(f"Database path: {db_path}")
    
    try:
        # Connect to ChromaDB
        logger.info("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=str(db_path))
        
        # List all collections
        all_collections = client.list_collections()
        logger.info(f"Found {len(all_collections)} total collections")
        
        # Get the specific collection
        logger.info(f"Getting collection: {collection_name}")
        collection = client.get_collection(name=collection_name)
        
        # Get collection info
        logger.info("Getting collection count...")
        count = collection.count()
        logger.info(f"Collection has {count} documents")
        
        # Try to peek at data
        logger.info("Peeking at collection data...")
        try:
            peek_data = collection.peek(limit=5)
            logger.info(f"Successfully peeked at {len(peek_data.get('ids', []))} documents")
            
            # Check for metadata
            if peek_data.get('metadatas'):
                logger.info(f"Sample metadata keys: {list(peek_data['metadatas'][0].keys()) if peek_data['metadatas'] else []}")
        except Exception as peek_error:
            logger.error(f"Error peeking at collection: {peek_error}")
        
        # Try to get a small sample
        logger.info("Attempting to query collection with small limit...")
        try:
            # Create a simple embedding (all zeros for testing)
            test_embedding = [0.0] * 1024  # BGE-Large dimension
            
            logger.info("Querying with n_results=1...")
            result = collection.query(
                query_embeddings=[test_embedding],
                n_results=1,
                include=["metadatas", "documents", "distances"]
            )
            logger.info(f"✅ Small query successful! Got {len(result['ids'][0])} results")
            
            logger.info("Querying with n_results=5...")
            result = collection.query(
                query_embeddings=[test_embedding],
                n_results=5,
                include=["metadatas", "documents", "distances"]
            )
            logger.info(f"✅ Medium query successful! Got {len(result['ids'][0])} results")
            
        except Exception as query_error:
            logger.error(f"❌ Query failed: {query_error}", exc_info=True)
            logger.info("\n🔍 DIAGNOSIS: Collection query is failing - likely corruption or size issue")
            
        logger.info("\n✅ Diagnosis complete!")
        
    except Exception as e:
        logger.error(f"Error during diagnosis: {e}", exc_info=True)


def list_collections(user_email: str):
    """List all collections for a user."""
    
    logger.info(f"Listing collections for user '{user_email}'")
    
    db_path = PROJECT_ROOT / "src" / "database" / user_email
    
    if not db_path.exists():
        logger.error(f"Database path does not exist: {db_path}")
        return
    
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collections = client.list_collections()
        
        logger.info(f"\n📚 Found {len(collections)} collections:")
        for i, coll in enumerate(collections, 1):
            count = coll.count()
            logger.info(f"  {i}. {coll.name} ({count} documents)")
            
    except Exception as e:
        logger.error(f"Error listing collections: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  List all collections:  python scripts/diagnose_collection.py <user_email>")
        print("  Diagnose collection:   python scripts/diagnose_collection.py <user_email> <collection_name>")
        print("\nExample:")
        print("  python scripts/diagnose_collection.py vardhan@coophive.network")
        print("  python scripts/diagnose_collection.py vardhan@coophive.network marker_recursive_bgelarge")
        sys.exit(1)
    
    user_email = sys.argv[1]
    
    if len(sys.argv) == 2:
        # Just list collections
        list_collections(user_email)
    else:
        # Diagnose specific collection
        collection_name = sys.argv[2]
        diagnose_collection(user_email, collection_name)

