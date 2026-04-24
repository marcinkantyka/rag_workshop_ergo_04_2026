# Retrieval Augmented Generation
**A Comprehensive Engineering Guide - ERGO x Netlight**

---

## Welcome

### What you will build

A production-oriented RAG pipeline on real insurance documents - from raw PDFs to a measurable, improvable system:

1. Load and prepare multilingual insurance PDFs (FR/NL/EN)
2. Compare chunking strategies and measure their effect on retrieval quality
3. Build a hybrid retrieval pipeline (semantic + keyword) tuned for insurance terminology
4. Evaluate pipeline quality with RAGAS: Faithfulness, Answer Relevancy, Context Precision, Context Recall
5. Run structured experiments, log results, and build an improvement backlog
6. Own the code and continue experimenting independently after the workshop

### What tech stack we are using

| Component | Library | Details |
|-----------|---------|---------|
| LLM | `openai` (LiteLLM proxy) | `claude-haiku-4-5` (fast) / `claude-sonnet-4-5` (quality) |
| Embeddings | `sentence-transformers` | `paraphrase-multilingual-mpnet-base-v2` - covers FR/NL/EN, 768-dim |
| Vector DB | `chromadb` | Local, file-based - no cloud setup required |
| Evaluation | `ragas` + `promptfoo` | Faithfulness, relevancy, precision, recall |
| PDF loading | `pdfplumber` + `pypdf` | Structured extraction for insurance PDFs |
| Hybrid search | `rank-bm25` | BM25 for exact term matching (INAMI, RIZIV) |

---

## Workshop

### Agenda

*Subject to change.*

#### Day 1 - Build the prototype (11:00–18:00)

| Time | Module | Content |
|------|--------|---------|
| 11:00–11:45 | Module 0 | Architecture walkthrough - RAG concept, ERGO context |
| 11:45–12:00 | - | Environment setup check |
| 12:00–13:00 | Module 1 | Document ingestion - load & explore PDFs |
| 13:00–14:00 | Lunch | |
| 14:00–15:00 | Module 2 | Chunking & multilingual embedding |
| 15:00–15:15 | Break | |
| 15:15–16:15 | Module 3 | Retrieval + pipeline - first FR/NL/EN queries |
| 16:15–17:00 | - | Results discussion - what works, what to improve |
| 17:00–18:00 | - | Day 2 prep - write test questions + ground truth |


#### Day 2 - Measure, experiment, improve (09:00–17:00)

| Time | Module | Content |
|------|--------|---------|
| 09:00–09:30 | Recap | Review Day 1 + finalise test dataset |
| 09:30–11:00 | Module 4 | Evaluation harness - baseline metrics |
| 11:00–11:15 | Break | |
| 11:15–12:45 | Module 5 | Experiment round 1: chunking strategies |
| 12:45–13:45 | Lunch | |
| 13:45–15:15 | Module 5 | Experiment round 2: retrieval config (vector vs. hybrid) |
| 15:15–15:30 | Break | |
| 15:30–16:30 | Module 6 | Advanced RAG - query expansion, HyDE, re-ranking |
| 16:30–17:00 | - | Experiment log review + improvement backlog |



### Working mode

All exercises run in **JupyterLab** as a sequence of notebooks. Each notebook covers one topic, builds on the previous one, and contains:

- Brief concept intro (markdown cells)
- Guided coding exercises with `# TODO` stubs to fill in
- Verification cells to check your output before moving on

The `src/` folder contains the shared library code - you import it directly in the notebooks. You never need to edit `src/` during the workshop, but the code is there to read.

### Repository structure

```
notebooks/   - Workshop exercises, one notebook per module
src/         - Shared library code imported by the notebooks
data/        - Sample insurance docs (FR/NL/EN) and ground-truth Q&A pairs
bonus/       - Optional extras: data cleaning and a Streamlit chat UI
```

> **Note:** Day 2 notebooks (Modules 4–6) will be published at the end of Day 1.

---

## Get Started


### Prerequisites

| Requirement | Minimum |
|-------------|---------|
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 |
| RAM | 8 GB (16 GB recommended) |
| Disk | 6 GB free |
| Docker Desktop | 4.x+ |

### Option A - Docker (recommended)

#### Step 1 - Install Docker Desktop

[docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)

```bash
docker --version        # Docker version 24.x or higher
docker compose version  # Docker Compose version 2.x
```

#### Step 2 - Get the workshop code

```bash
git clone https://github.com/lukasstuetzel/rag_workshop_ergo_04_2026.git
cd rag_workshop_ergo_04_2026
```

#### Step 3 - Set your API key

```bash
cp .env.example .env
```

Open `.env` and set:
```
NETLIGHT_API_KEY=sk-...        # provided by the workshop organisers
NETLIGHT_BASE_URL=https://llm.netlight.ai/
```

#### Step 4 - Start the workshop environment

```bash
docker compose up
```

On first run this will:
1. Pull the workshop image (~5 GB, one-time) — no registry login needed
2. Start JupyterLab (ML models are pre-baked into the image)


Open your browser: **http://localhost:8888?token=workshop**

You should see JupyterLab with the workshop notebooks in the left sidebar.

#### Stopping and restarting

```bash
docker compose down   # stop

docker compose up     # restart - notebook edits and ChromaDB data are preserved
```

---

### Option B - Local Python venv (fallback)

Use this only if Docker is not available on your machine.

#### Step 1 - Python 3.11 or 3.12

```bash
python --version   # must be 3.11.x or 3.12.x - NOT 3.13
```

Install from [python.org/downloads](https://www.python.org/downloads/) if needed.
macOS: `brew install python@3.11`

#### Step 2 - Get the code and install dependencies

```bash
git clone https://github.com/lukasstuetzel/rag_workshop_ergo_04_2026.git
cd rag_workshop_ergo_04_2026
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> Installing CPU-only torch first avoids accidentally pulling the 2.5 GB CUDA variant and matches the Docker environment exactly.

#### Step 3 - Download the embedding model (~600 MB - do on a fast connection)

```bash
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading multilingual model...')
m = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
print(f'Done! Vector size: {len(m.encode(\"test\"))}')
"
```

#### Step 4 - Set your API key

```bash
cp .env.example .env
# add your NETLIGHT_API_KEY to .env
```

---

### Verify

**Docker**
- Open http://localhost:8888?token=workshop
- Navigate to `notebooks/00_setup_and_architecture.ipynb`
- Run all cells - all checkmarks should be green

**Local venv**
- Run `jupyter lab`
- Open `notebooks/00_setup_and_architecture.ipynb`
- Run all cells - all checkmarks should be green

---

### Troubleshooting

**Port conflict...**
Change the port in `docker-compose.yml`: `"8889:8888"`, then open http://localhost:8889?token=workshop

**Image pull is slow...**
Do it the evening before. On Day 1 morning, `docker compose up` starts in seconds.

**"NETLIGHT_API_KEY not found"...**
Make sure `.env` is in the project root (same folder as `docker-compose.yml`).

**"Kernel not found" in JupyterLab...**
Activate the venv first: `source .venv/bin/activate && jupyter lab`

**Windows: empty `/workspace`...**
Docker Desktop - Settings - Resources - File Sharing - add the project folder.
