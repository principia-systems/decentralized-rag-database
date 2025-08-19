# 🧠 CoopHive: Decentralized RAG Database & Web Platform

CoopHive is a comprehensive decentralized RAG (Retrieval-Augmented Generation) platform for scientific literature. It features a modular backend processing system and a modern web interface for document ingestion, processing, querying, and collaborative research.

---

## 🏗️ Architecture

CoopHive consists of two main components:

### Backend: Decentralized RAG Database (`decentralized-rag-database/`)
- **Modular Processing Pipeline**: Configurable document conversion, chunking, and embedding
- **Multi-Server Architecture**: Light, Heavy, and Database servers for optimized performance
- **IPFS Integration**: Decentralized storage and content addressing
- **Token Rewards**: Blockchain-based incentivization system

### Frontend: Web Platform (`dvd-frontend/`)
- **Modern Next.js App**: Built with T3 stack (TypeScript, Tailwind, NextAuth)
- **Authenticated API Gateway**: Secure proxy to backend services
- **Real-time Chat Interface**: Interactive research assistant
- **User Management**: Whitelist-based access control

## ✨ Key Features

- **🔄 Modular Architecture**: Pluggable converters, chunkers, embedders, and reward strategies
- **📊 Reproducibility**: Deterministic pipelines with version-controlled configurations
- **🔗 Transparency**: All processing traceable through Git commits and IPFS hashes
- **🚀 Scalable Servers**: Distributed architecture for optimal resource utilization
- **🔐 Secure Access**: NextAuth-based authentication with role-based permissions
- **💬 Interactive Chat**: Real-time research assistance with context-aware responses
- **📈 Progress Tracking**: Real-time job status and completion monitoring

---

## 🚀 Quick Start

### Prerequisites

- **Backend**: Python 3.10+, Poetry, Docker (optional)
- **Frontend**: Node.js 18+, npm/yarn
- **Services**: OpenAI API, Lighthouse/IPFS, Neo4j, PostgreSQL
- **Optional**: NVIDIA GPU for local embeddings

### Installation

#### Backend Setup
```bash
git clone https://github.com/coophive/decentralized-rag-database.git
cd decentralized-rag-database
bash scripts/setup.sh
poetry lock --no-update
poetry install
cp .env.example .env
```

#### Frontend Setup
```bash
cd ../dvd-frontend
npm install
cp .env.example .env.local
```

### 💡 Environment Variables

#### Backend Configuration (`.env`)
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

#### Frontend Configuration (`.env.local`)
```bash
# NextAuth Configuration
NEXTAUTH_SECRET=your_nextauth_secret
NEXTAUTH_URL=http://localhost:3000

# Server URLs (backend services)
LIGHT_SERVER_URL=http://localhost:5001
HEAVY_SERVER_URL=http://localhost:5002
DATABASE_SERVER_URL=http://localhost:5003

# OpenRouter (server-side only)
OPENROUTER_API_KEY=your_openrouter_key
```

#### GPU Configuration

The `GPU_SPLIT` environment variable controls multi-GPU usage for local embedding models:

- **0.75** (default): Uses 75% of available GPUs
- **1.0**: Uses all available GPUs  
- **0.5**: Uses 50% of available GPUs
- **Local models**: `bge`, `bgelarge`, `e5large`
- **API models**: `openai`, `nvidia` (not affected)

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

#### Start Frontend
```bash
# Terminal 4 - Web Interface (port 3000)
cd dvd-frontend
npm run dev
```

#### Access the Platform
- **Web Interface**: http://localhost:3000
- **API Documentation**: 
  - Light Server: http://localhost:5001/docs
  - Heavy Server: http://localhost:5002/docs  
  - Database Server: http://localhost:5003/docs

#### CLI Processing (Alternative)
```bash
bash scripts/run_processor.sh         # Convert, chunk, embed documents
bash scripts/run_db_creator.sh        # Recreate DBs from IPFS
bash scripts/run_evaluation.sh        # Query and evaluate DBs
bash scripts/run_token_reward.sh      # Distribute token rewards
```

### Code Quality and Testing

```bash
bash scripts/lint.sh                   # Lint (black, isort, flake8, mypy)
bash scripts/test.sh --integration      # Run integration tests
```

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
    "collections": ["optional_collection_filter"]
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
            "metadata": {"source": "paper1.pdf", "page": 3},
            "score": 0.95
          }
        ]
      }
    ],
    "total_results": 15,
    "query_time": 1.23
  }
  ```

### 🌐 Web Interface - Authenticated Proxy (Port 3000)

The frontend provides authenticated proxy routes to all v1 endpoints:

- **Light Server**: `/api/light/v1/*` → `http://localhost:5001/api/v1/*`
- **Heavy Server**: `/api/heavy/v1/*` → `http://localhost:5002/api/v1/*`  
- **Database Server**: `/api/database/v1/*` → `http://localhost:5003/api/v1/*`

**Authentication Required**: All proxy routes require valid NextAuth session.

### 📋 Usage Examples

#### 1. Check Processing Status
```bash
curl "http://localhost:5001/api/v1/user/status?user_email=researcher@university.edu"
```

#### 2. Start Document Processing
```bash
curl -X POST http://localhost:5002/api/v1/users/ingestion \
  -H "Content-Type: application/json" \
  -d '{
    "drive_url": "https://drive.google.com/drive/folders/1ABC123",
    "processing_combinations": [["openai", "recursive", "openai"]],
    "user_email": "researcher@university.edu"
  }'
```

#### 3. Query Processed Documents
```bash
curl -X POST http://localhost:5003/api/v1/user/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning applications in biology",
    "user_email": "researcher@university.edu"
  }'
```

### 🔒 Rate Limits & Authentication

- **Rate Limits**: 100 requests/minute per user
- **Authentication**: Whitelist-based access control
- **API Keys**: Contact admin for production access
- **CORS**: Enabled for web applications

---

## 📚 System Components

### 🔄 Processing Pipeline

The processing pipeline converts documents through three configurable stages:

1. **Conversion**: PDF → Markdown
   - `marker`: High-quality academic PDF conversion
   - `openai`: GPT-4 powered conversion with OCR
   - `markitdown`: Microsoft's open-source converter

2. **Chunking**: Document → Text Chunks  
   - `fixed_length`: Fixed character/token chunks
   - `recursive`: Recursive text splitting
   - `markdown_aware`: Preserves markdown structure
   - `semantic_split`: AI-powered semantic boundaries

3. **Embedding**: Text → Vector Embeddings
   - `openai`: OpenAI text-embedding-ada-002
   - `nvidia`: NVIDIA NeMo embeddings
   - `bge`, `bgelarge`: BGE models (local GPU)
   - `e5large`: E5 model (local GPU)

### 🗄️ Database Architecture

- **ChromaDB**: Vector storage for embeddings
- **Neo4j**: Graph database for lineage tracking
- **PostgreSQL**: User data and job tracking
- **IPFS/Lighthouse**: Decentralized content storage

### 🔍 Query & Evaluation

- **Multi-DB Querying**: Searches across user's vector databases
- **Cross-Encoder Ranking**: Re-ranks results for relevance
- **LLM Integration**: OpenRouter/OpenAI for response generation
- **Evaluation Storage**: Tracks user feedback for improvement

### 🏆 Token Rewards

- **ERC20 Integration**: Blockchain-based contributor rewards
- **Contribution Tracking**: Git commits and processing jobs
- **Reward Models**: Job count, bonuses, time decay
- **Smart Contracts**: Automated distribution via Hardhat

### 🔐 Authentication & Security

- **NextAuth.js**: Session-based authentication
- **Whitelist System**: Admin-controlled user access
- **API Gateway**: Secure proxy to backend services
- **CORS Configuration**: Development and production modes

---

## 🛠️ Tech Stack

### Backend
- **Python 3.10+** - Core processing engine
- **FastAPI** - High-performance API servers
- **ChromaDB** - Vector database for embeddings
- **Neo4j** - Graph database for lineage tracking
- **PostgreSQL** - Relational data storage
- **IPFS/Lighthouse** - Decentralized content storage
- **Poetry** - Dependency management

### Frontend  
- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **NextAuth.js** - Authentication system
- **Tailwind CSS** - Utility-first styling
- **Radix UI** - Accessible component primitives
- **T3 Stack** - Type-safe full-stack development

### AI/ML
- **OpenAI API** - GPT models and embeddings
- **OpenRouter** - Multi-model API gateway
- **NVIDIA NeMo** - Enterprise embeddings
- **BGE/E5 Models** - Local embedding models
- **Sentence Transformers** - Cross-encoder ranking

### Blockchain
- **Hardhat** - Ethereum development framework
- **Solidity** - Smart contract development
- **ERC20** - Token standard implementation

### DevOps
- **Docker** - Containerization
- **Poetry** - Python package management
- **ESLint/Prettier** - Code formatting
- **GitHub Actions** - CI/CD (optional)

---

## 📄 Project Structure

```bash
coophive-markdown-converter/
├── config/        # YAML pipeline configs
├── src/       # Core libraries (processing, storage, rewards)
├── scripts/       # CLI scripts for pipelines
├── contracts/     # Blockchain contract ABIs
├── docker/        # Container specs
├── erc20-token/   # Token contract config
├── papers/        # Example documents
├── tests/         # Unit and integration tests
└── .github/       # CI/CD configurations
```

---

## 📅 License

This project is open-sourced under the MIT License.
