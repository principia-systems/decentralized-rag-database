# 🧠 CoopHive: Decentralized RAG Database

CoopHive is a decentralized RAG (Retrieval-Augmented Generation) platform for scientific literature with a modular backend processing system for document ingestion, processing, and querying.

---

## 🏗️ Architecture

### Decentralized RAG Database
- **Modular Processing Pipeline**: Configurable document conversion, chunking, and embedding
- **Multi-Server Architecture**: Light, Heavy, and Database servers for optimized performance
- **IPFS Integration**: Decentralized storage and content addressing
- **Token Rewards**: Blockchain-based incentivization system

## ✨ Key Features

- **🔄 Modular Architecture**: Pluggable converters, chunkers, embedders, and reward strategies
- **📊 Reproducibility**: Deterministic pipelines with version-controlled configurations
- **🔗 Transparency**: All processing traceable through Git commits and IPFS hashes
- **🚀 Scalable Servers**: Distributed architecture for optimal resource utilization
- **📈 Progress Tracking**: Real-time job status and completion monitoring

---

## 🚀 Quick Start

### Prerequisites

- **Backend**: Python 3.10+, Poetry, Docker (optional)
- **Services**: OpenAI API, Lighthouse/IPFS, Neo4j, PostgreSQL
- **Optional**: NVIDIA GPU for local embeddings

### Installation

```bash
git clone https://github.com/coophive/decentralized-rag-database.git
cd decentralized-rag-database
bash scripts/setup.sh
poetry lock --no-update
poetry install
cp .env.example .env
```

### 💡 Environment Variables

Configuration (`.env`):
```bash
# Core APIs
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
LIGHTHOUSE_TOKEN=your_lighthouse_token

# Database Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Blockchain (for token rewards)
OWNER_ADDRESS=your_wallet_address
PRIVATE_KEY=your_private_key

# Server URLs (for multi-server setup)
LIGHT_SERVER_URL=http://localhost:5001
HEAVY_SERVER_URL=http://localhost:5002
DATABASE_SERVER_URL=http://localhost:5003

# GPU Configuration (optional)
GPU_SPLIT=0.75  # Percentage of GPUs for local embeddings
```

### Running the Platform

#### Start Backend Servers
```bash
# Terminal 1 - Light Server (port 5001)
bash scripts/start_light_server.sh

# Terminal 2 - Heavy Server (port 5002) 
bash scripts/start_heavy_server.sh

# Terminal 3 - Database Server (port 5003)
bash scripts/start_database_server.sh
```

#### API Documentation
- Light Server: http://localhost:5001/docs
- Heavy Server: http://localhost:5002/docs  
- Database Server: http://localhost:5003/docs

---

## 🔌 Public API Documentation

CoopHive provides public v1 API endpoints for customer integration. All v1 endpoints are production-ready and customer-facing.

### 🚀 Light Server - Quick Operations (Port 5001)

**`GET /api/v1/user/status`**
- **Description**: Get real-time processing status for a user
- **Parameters**: `?user_email={email}` (query parameter)
- **Response**: 
  ```json
  {
    "total_jobs": 100,
    "completed_jobs": 75,
    "completion_percentage": 75.0
  }
  ```
- **Use Case**: Monitor document processing progress

### ⚡ Heavy Server - Document Processing (Port 5002)

**`POST /api/v1/users/ingestion`**
- **Description**: Ingest and process PDFs from Google Drive
- **Content-Type**: `application/json`
- **Request Body**: 
  ```json
  {
    "drive_url": "https://drive.google.com/drive/folders/1ABC123...",
    "processing_combinations": [
      ["markitdown", "recursive", "bge"],
      ["openai", "semantic_split", "openai"]
    ],
    "user_email": "user@example.com"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "message": "Processing started for 10 PDF files with 2 combinations",
    "downloaded_files": ["paper1.pdf", "paper2.pdf"],
    "total_files": 10,
    "processing_combinations": ["markitdown_recursive_bge", "openai_semantic_split_openai"],
    "processing_started": true
  }
  ```

#### Processing Components

| Component | Options | Description |
|-----------|---------|-------------|
| **Converters** | `marker`, `openai`, `markitdown` | PDF to Markdown conversion |
| **Chunkers** | `fixed_length`, `recursive`, `markdown_aware`, `semantic_split` | Text segmentation strategies |
| **Embedders** | `openai`, `nvidia`, `bge`, `bgelarge`, `e5large` | Vector embedding models |

### 🗄️ Database Server - Querying & Evaluation (Port 5003)

**`POST /api/v1/user/evaluate`**
- **Description**: Query user's processed document databases
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "query": "What are the latest developments in CRISPR gene editing?",
    "user_email": "user@example.com",
    "model_name": "openai/gpt-4o-mini",
    "collections": ["optional_collection_filter"],
    "k": 5
  }
  ```
- **Response**: 
  ```json
  {
    "query": "What are the latest developments in CRISPR?",
    "results": [
      {
        "collection": "markitdown_recursive_bge",
        "documents": [
          {
            "content": "CRISPR-Cas9 has revolutionized...",
            "metadata": {...},
            "score": 0.95
          }
        ]
      }
    ],
    "total_results": 15,
    "query_time": 1.23
  }
  ```

**`POST /api/v1/user/evaluate/aggregate`**
- **Description**: Query user's databases with intelligent result aggregation
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "query": "What are the latest developments in CRISPR gene editing?",
    "user_email": "user@example.com",
    "model_name": "openai/gpt-4o-mini",
    "k": 10,
    "aggregation_strategy": "hybrid",
    "top_k": 5,
    "similarity_weight": 0.7,
    "frequency_weight": 0.3,
    "min_similarity_threshold": 0.1
  }
  ```
- **Response**: 
  ```json
  {
    "query": "What are the latest developments in CRISPR?",
    "user_email": "user@example.com",
    "aggregation_strategy": "hybrid",
    "aggregated_results": [
      {
        "content": "CRISPR-Cas9 has revolutionized gene editing...",
        "similarity": 0.95,
        "frequency": 3,
        "final_score": 0.89,
        "rank": 1,
        "collection": "markitdown_recursive_bge",
        "metadata": {...}
      }
    ],
    "total_aggregated_items": 5
  }
  ```
- **Parameters**:
  - `aggregation_strategy`: `"frequency"`, `"similarity"`, or `"hybrid"` (default: `"hybrid"`)
  - `k`: Number of results per collection (default: `5`)
  - `top_k`: Final number of aggregated results (default: `5`)
  - `similarity_weight`: Weight for similarity in hybrid mode (default: `0.7`)
  - `frequency_weight`: Weight for frequency in hybrid mode (default: `0.3`)
  - `min_similarity_threshold`: Minimum similarity to consider (default: `0.1`)

**`POST /api/v1/user/reranker`**
- **Description**: Rerank a list of text items using a cross-encoder model for better relevance scoring
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "user_email": "user@example.com",
    "query": "What are the latest developments in CRISPR gene editing?",
    "items": [
      "CRISPR-Cas9 has revolutionized gene editing...",
      "Gene therapy approaches using viral vectors...",
      "Recent advances in base editing techniques..."
    ],
    "model_preset": "msmarco-MiniLM-L-6-v2",
    "batch_size": 16,
    "max_length": 256,
    "top_k": 10,
    "descending": true,
    "device": "cpu"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "results": [
      {"item": "Text item 1...", "score": 0.95},
      {"item": "Text item 3...", "score": 0.87}
    ]
  }
  ```
- **Parameters**:
  - `user_email`: User identifier for logging and tracking (required)
  - `query`: The search query to rank items against (required)
  - `items`: List of text items to rerank (required)
  - `model_preset`: Cross-encoder model to use (default: `"msmarco-MiniLM-L-6-v2"`)
  - `batch_size`: Processing batch size for efficiency (default: `16`)
  - `max_length`: Maximum token length for input sequences (default: `256`)
  - `top_k`: Number of top results to return (optional, returns all if not specified)
  - `descending`: Sort order - true for highest scores first (default: `true`)
  - `device`: Processing device - `"cpu"` or `"cuda"` (default: `"cpu"`)

#### Metadata Structure

The `metadata` field in responses contains comprehensive document information:

```json
{
  "content_cid": "QmABC123...",
  "root_cid": "QmXYZ789...",
  "embedding_cid": "QmDEF456...",
  "content": "Full text content of the chunk",
  "title": "Paper Title from PDF metadata",
  "authors": "['Author 1', 'Author 2']",
  "abstract": "Paper abstract text",
  "doi": "10.1000/journal.2023.001",
  "publication_date": "2023-01-01",
  "journal": "Journal Name",
  "keywords": "['keyword1', 'keyword2']",
  "url": "https://example.com/paper.pdf"
}
```

**Key Fields**:
- `content_cid`: IPFS hash of the text chunk
- `root_cid`: IPFS hash of the original PDF document
- `embedding_cid`: IPFS hash of the vector embedding
- `content`: The actual text content of the chunk
- PDF metadata fields (when available): `title`, `authors`, `abstract`, `doi`, etc.

---

## 📄 Project Structure

```bash
decentralized-rag-database/
├── config/        # YAML pipeline configs
├── src/           # Core libraries (processing, storage, rewards)
├── scripts/       # CLI scripts for pipelines
├── contracts/     # Blockchain contract ABIs
├── docker/        # Container specs
├── erc20-token/   # Token contract config
├── tests/         # Unit and integration tests
└── .github/       # CI/CD configurations
```

---

## 📅 License

This project is open-sourced under the MIT License.
