# 🧠 CoopHive: Modular RAG Pipelines for Scientific Papers

CoopHive builds modular, reproducible RAG (Retrieval-Augmented Generation) pipelines for scientific literature. It supports customizable document processing, embedding, querying, and incentivization mechanisms, enabling decentralized and transparent scientific collaboration.

---

## ✨ Features

- **Modular Architecture**: Easily swap converters, chunkers, embedders, and reward strategies via configuration files. Each component (conversion, chunking, embedding) is pluggable, allowing you to add your own custom methods or extend existing ones.
- **Reproducibility**: Deterministic pipelines with version-controlled configs ensure consistent, repeatable results.
- **Transparency**: All processing and contributions are traceable through Git commits and IPFS hashes.
- **Built-in Pipelines**:
  - Document conversion to markdown
  - Text chunking
  - Embedding of chunks
  - Storage in vector databases and decentralized storage
  - Database recreation from decentralized storage
  - Querying and evaluation of vector databases
  - Token rewarding for contributors

Examples for common methods (like Marker, OpenAI, Lighthouse, etc.) are provided but can be extended with any custom logic by implementing your own classes.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Poetry
- Node.js 18+
- Access to APIs (e.g., OpenAI, Lighthouse, OpenRouter)

### Installation

```bash
git clone https://github.com/your-repo/coophive-markdown-converter.git
cd coophive-markdown-converter
bash scripts/setup.sh
poetry lock --no-update
poetry install
cp .env.example .env
```

### 💡 Environment Variables

Create a `.env` file with the following keys:

```bash
OPENAI_API_KEY=
NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=
OWNER_ADDRESS=
PRIVATE_KEY=
LIGHTHOUSE_TOKEN=
OPENROUTER_API_KEY=
GPU_SPLIT=0.75  # Optional: Percentage of GPUs to use for embedding (0.75 = 75%)
```

#### GPU Configuration

The `GPU_SPLIT` environment variable controls multi-GPU usage for local embedding models (BGE, E5):

- **0.75** (default): Uses 75% of available GPUs, starting from the highest-numbered GPU
- **1.0**: Uses all available GPUs  
- **0.5**: Uses 50% of available GPUs
- **Examples**:
  - 4 GPUs with `GPU_SPLIT=0.75`: Uses GPUs 1, 2, 3 (skips GPU 0)
  - 4 GPUs with `GPU_SPLIT=1.0`: Uses GPUs 0, 1, 2, 3
  - 2 GPUs with `GPU_SPLIT=0.75`: Uses GPU 1
  
> **Note**: Multi-GPU acceleration applies only to local embedding models (`bge`, `bgelarge`, `e5large`). API-based models (`openai`, `nvidia`) are not affected.

### Running Modules

```bash
bash scripts/run_processor.sh         # Convert, chunk, embed, and store documents
bash scripts/run_db_creator.sh         # Recreate DBs from IPFS graph
bash scripts/run_evaluation.sh         # Query and evaluate across DBs
bash scripts/run_token_reward.sh       # Distribute ERC20 token rewards
```

Or launch an interactive session:

```bash
poetry shell
```

### Code Quality and Testing

```bash
bash scripts/lint.sh                   # Lint (black, isort, flake8, mypy)
bash scripts/test.sh --integration      # Run integration tests
```

---

## �\uddna Module Configuration

### ✨ Processor

- Converts PDF to markdown using a configurable converter module
- Chunks text into paragraphs, sentences, or fixed lengths
- Embeds text chunks using a configurable embedder
- Uploads outputs to IPFS and stores metadata in databases
- Logs all processing as Git commits

All core components are modular and can be customized by implementing new classes under `src/core/` and specifying them in the config file.

Configurable at [`config/processor.yml`](config/processor.yml):

```yaml
converter: marker # Options: marker, openai, custom
embedder: openai # Options: openai, nvidia, custom
```

### 🔁 DB Creator

- Traverses the IPFS graph in Neo4j
- Rebuilds documents, chunks, and embeddings
- Supports depth control and relationship mapping

Configurable at [`config/db_creator.yml`](config/db_creator.yml):

```yaml
components:
  converter:
    - marker
  chunker:
    - fixed_length
  embedder:
    - openai

cids_file_paths:
  - cids.txt
```

### 🔎 Evaluation Agent

- Runs queries across vector DBs
- Uses an LLM to rank and analyze retrieval performance

Example snippet:

```yaml
query: "impact of CRISPR on neuroscience"
model_name: "gpt-4"
```

Outputs are saved in `temp/evaluation/`.

### 🏅 Token Rewarding

- Analyzes contributions and distributes ERC20 tokens
- Supports job count, bonuses, and time decay reward models

Configuration example:

```yaml
databases:
  - converter: openai
    chunker: paragraph
    embedder: openai
```

Extend `src/rewards/token_rewarder.py` to customize reward mechanisms.

---

## 🛠️ Tech Stack

- **Python** (workflow orchestration)
- **ChromaDB** (vector storage)
- **Neo4j** (graph lineage and job tracking)
- **IPFS / Lighthouse** (document storage)
- **Embedding Models** (pluggable, default examples included)
- **Hardhat + Solidity** (ERC20 token management)
- **Docker + Nomad** (optional deployment)

---

## 📄 Directory Overview

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
