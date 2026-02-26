# noid-rag

Production RAG CLI and Python library. Parse documents, chunk them, generate embeddings, store in PostgreSQL with pgvector, and run semantic search — from the command line or your own code.

## Features

- **Document parsing** via [Docling](https://github.com/DS4SD/docling) with optional OCR
- **Hybrid chunking** — structure-aware or fixed-size, with configurable token limits
- **Embedding** via any OpenAI-compatible API (OpenRouter, OpenAI, local)
- **PostgreSQL + pgvector** storage with async connection pooling
- **LLM answer synthesis** — search results fed to an LLM for grounded answers
- **Hybrid search** — vector + keyword search with Reciprocal Rank Fusion (RRF)
- **Batch processing** with progress bars, retry support, and error resilience
- **Python API** (`NoidRag` class) with sync and async interfaces
- **Eval & tuning** — evaluate pipeline quality with RAGAS or [promptfoo](https://www.promptfoo.dev/) (external, requires `npx`), generate synthetic datasets
- **Batch history** — each batch run is saved to `~/.noid-rag/history/` for retry and audit
- **Fully configurable** — every quality-affecting parameter tunable via YAML, env vars, or both

## Quick Start

```bash
# Install
uv sync

# Set your API key and database DSN
cp .env.example .env
# Edit .env with your values

# Copy and edit the config file
cp config.example.yml ~/.noid-rag/config.yml

# Ingest a document
noid-rag ingest document.pdf

# Search
noid-rag search "your query"
```

## Installation

**Prerequisites:** Python 3.11+, PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension.

```bash
# With uv (recommended)
uv sync

# With pip
pip install .

# With local embedding models (sentence-transformers)
uv sync --extra local

```

### OCR Support

[Docling](https://github.com/DS4SD/docling) is installed automatically and handles document parsing. EasyOCR is included as a core dependency for OCR on scanned PDFs. To use Tesseract instead, install it as a **system dependency** (it is not a Python package):

```bash
# Tesseract (alternative OCR engine — system package, not pip-installable)
brew install tesseract          # macOS
sudo apt install tesseract-ocr  # Ubuntu/Debian
```

Set `ocr_engine` in your config to `easyocr` (default), `tesseract`, or `auto` (tries easyocr first, falls back to tesseract).

### Promptfoo (optional)

The `promptfoo` eval backend requires [promptfoo](https://www.promptfoo.dev/) to be available via `npx`. It is **not** bundled as a Python dependency:

```bash
npm install -g promptfoo   # or use npx (auto-installed on first run)
```

## Configuration

noid-rag loads settings from environment variables and an optional YAML config file.

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# Embedding API key
NOID_RAG_EMBEDDING__API_KEY=your-openrouter-api-key
# Or: OPENROUTER_API_KEY=your-key

# PostgreSQL connection
NOID_RAG_VECTORSTORE__DSN=postgresql+asyncpg://user:pass@localhost:5432/noid_rag

# LLM API key (for answer synthesis)
NOID_RAG_LLM__API_KEY=your-openrouter-api-key

# Optional
NOID_RAG_VERBOSE=false
```

### YAML Config

Copy `config.example.yml` to `~/.noid-rag/config.yml` or pass `--config path/to/config.yml`:

```yaml
parser:
  ocr_enabled: true
  ocr_engine: easyocr
  max_pages: 0

chunker:
  method: hybrid        # hybrid or fixed
  max_tokens: 512
  tokenizer: BAAI/bge-small-en-v1.5

embedding:
  provider: openrouter
  api_url: https://openrouter.ai/api/v1/embeddings
  model: openai/text-embedding-3-small
  batch_size: 64

vectorstore:
  dsn: postgresql+asyncpg://user:pass@localhost:5432/noid_rag
  table_name: documents
  embedding_dim: 1536
  pool_size: 20
  fts_language: english  # PostgreSQL FTS language (english, spanish, simple, etc.)

search:
  top_k: 5              # chunks to retrieve per query
  rrf_k: 60             # Reciprocal Rank Fusion constant

llm:
  api_url: https://openrouter.ai/api/v1/chat/completions
  model: openai/gpt-4o-mini
  max_tokens: 1024
  temperature: 0.0       # 0.0 = deterministic, best for RAG
  # system_prompt: "..."  # customize answer behavior

batch:
  max_retries: 3
  continue_on_error: true
```

See `config.example.yml` for the full list of options with detailed comments.

## CLI Usage

All commands support `--config/-c` for a custom config file and `--verbose/-v` for verbose output.

### `parse` — Parse a document to markdown

```bash
noid-rag parse document.pdf
noid-rag parse document.pdf --show           # Display parsed content
noid-rag parse document.pdf --output out.md  # Save markdown to file
```

### `chunk` — Parse and chunk a document

```bash
noid-rag chunk document.pdf
noid-rag chunk document.pdf --method fixed --max-tokens 256
noid-rag chunk document.pdf --show           # Display chunks
noid-rag chunk document.pdf --output chunks.json
```

### `ingest` — Full pipeline (parse, chunk, embed, store)

```bash
noid-rag ingest document.pdf
```

### `search` — Semantic search

```bash
noid-rag search "how does authentication work?"
noid-rag search "deployment steps" --top-k 10
noid-rag search "error handling" --no-answer   # Skip LLM, show raw results
noid-rag search "error handling" --output results.json
```

### `batch` — Batch process a directory

```bash
noid-rag batch ./docs/
noid-rag batch ./docs/ --pattern "*.pdf"
noid-rag batch ./docs/ --dry-run             # List files without processing
noid-rag batch ./docs/ --retry <run-id>      # Retry failed files from a previous run
```

### `generate` — Generate synthetic eval datasets

```bash
noid-rag generate -o dataset.yml                    # Generate from indexed docs
noid-rag generate -o dataset.yml --num-questions 50
noid-rag generate -o dataset.yml --strategy random  # random or diverse sampling
```

### `eval` — Evaluate RAG pipeline quality

```bash
noid-rag eval dataset.yml                        # Run with default metrics
noid-rag eval dataset.yml --backend promptfoo    # Use promptfoo instead of ragas
noid-rag eval dataset.yml --metrics faithfulness,context_precision
noid-rag eval dataset.yml --top-k 10 --verbose   # Per-question breakdown
```

### `tune` — Hyperparameter optimization

Automatically find optimal RAG parameters using Bayesian search (Optuna). Define a search space in your config YAML:

```yaml
tune:
  max_trials: 10
  search_space:
    chunker:
      max_tokens: [256, 512, 1024]       # categorical: pick one per trial
      method: [hybrid, fixed]
    search:
      top_k: [3, 5, 10, 15]
      rrf_k: [20, 40, 60, 80]
    llm:
      temperature: {low: 0.0, high: 0.3, step: 0.1}  # numeric range
    embedding:
      model: [openai/text-embedding-3-small]
```

Run tuning:

```bash
noid-rag tune dataset.yml --source doc1.pdf --source doc2.pdf
noid-rag tune dataset.yml -s doc.pdf --max-trials 20   # override trial count
noid-rag tune dataset.yml -s doc.pdf --output results.json --verbose
```

The optimizer uses the mean of all configured eval metrics as the objective. Ingestion is cached — trials that only change search/LLM parameters skip re-ingestion.

### `info` — Vector store statistics

```bash
noid-rag info
```

### `reset` — Reset the vector store

```bash
noid-rag reset                   # Drop table (with confirmation prompt)
noid-rag reset --yes             # Skip confirmation
```

## Python API

Use the `NoidRag` class for programmatic access:

```python
from noid_rag.api import NoidRag

rag = NoidRag()

# Parse a document
doc = rag.parse("document.pdf")

# Parse and chunk
chunks = rag.chunk("document.pdf")

# Full pipeline: parse, chunk, embed, store
result = rag.ingest("document.pdf")

# Semantic search (top_k defaults to config, or pass explicitly)
results = rag.search("your query")

# Search + LLM answer synthesis
answer = rag.answer("your query", top_k=10)

# Batch process a directory
result = rag.batch("./docs/", pattern="*.pdf")
```

Hyperparameter tuning:

```python
rag = NoidRag(config={
    "tune": {
        "max_trials": 10,
        "search_space": {
            "search": {"top_k": [3, 5, 10]},
            "llm": {"temperature": {"low": 0.0, "high": 0.3, "step": 0.1}},
        },
    }
})
result = rag.tune("dataset.yml", sources=["doc.pdf"])
print(result.best_params)  # {'search': {'top_k': 5}, 'llm': {'temperature': 0.1}}
print(result.best_score)   # 0.82
```

Async variants are also available:

```python
import asyncio
from noid_rag.api import NoidRag

rag = NoidRag()

async def main():
    result = await rag.aingest("document.pdf")
    results = await rag.asearch("your query")
    answer = await rag.aanswer("your query", top_k=10)
    batch_result = await rag.abatch("./docs/", pattern="*.pdf")

asyncio.run(main())
```

Pass custom configuration:

```python
rag = NoidRag({"embedding": {"model": "text-embedding-3-small"}})
```

## Architecture

```
document → parse (Docling) → chunk (hybrid/fixed) → embed (API) → store (pgvector) → search → answer (LLM)
```

1. **Parse** — Docling converts PDF/DOCX/HTML to structured markdown
2. **Chunk** — Hybrid (structure-aware) or fixed-size splitting with token limits
3. **Embed** — Vectors generated via any OpenAI-compatible embeddings API
4. **Store** — Chunks and vectors upserted into PostgreSQL with pgvector
5. **Search** — Hybrid search (vector + keyword) with Reciprocal Rank Fusion
6. **Answer** — LLM synthesizes a grounded answer from search results

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
pytest

# Run with coverage
pytest --cov=noid_rag

# Lint
ruff check src/ tests/
```

## License

MIT
