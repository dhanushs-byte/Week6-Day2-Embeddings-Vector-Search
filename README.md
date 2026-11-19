# Week 6 Day 2 - Embeddings and Vector Search (No Database)

This project is a database-free Python adaptation of both Week 6 Day 2 PDFs.

## What it covers

### Part 1 - Exact vs Approximate Search
- Creates 50,000 random vectors with 128 dimensions.
- Runs exact NumPy nearest-neighbour search.
- Runs FAISS `IndexFlatL2` exact search.
- Runs FAISS `IndexHNSWFlat` approximate search.
- Measures search time and Recall@10.
- Changes HNSW `M` and `efSearch`.
- Saves `recall_vs_time.png`.
- Explains HNSW and the exact-vs-ANN trade-off.

### Part 2 - In-memory vector store
- Creates 30 sample employee-help documents.
- Generates 384-dimensional sentence embeddings with `all-MiniLM-L6-v2`.
- Stores vectors and payloads in Python dictionaries (no database).
- Demonstrates semantic search and payload filters:
  `category == "HR"` and `page >= 3`.
- Re-upserts the same IDs and verifies that count remains 30.
- Deletes points by category and verifies the new count.
- Maintains a Python category index.
- Compares cosine, L2 and inner-product rankings.
- Includes a conceptual pgvector-vs-Qdrant comparison.

## Important limitation

This is **not** a real Qdrant or PostgreSQL/pgvector implementation. It does not start a database, create a Qdrant dashboard, execute SQL, or produce PostgreSQL `EXPLAIN ANALYZE` screenshots. It demonstrates the underlying vector-search and payload-management concepts locally, as requested.

## Requirements

Recommended on Windows: Python 3.11 or 3.12. FAISS wheels may not be available for every Python version, especially very new versions.

## Setup in VS Code (Windows PowerShell)

Open this folder in VS Code, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
python main.py
```

The first run downloads the sentence-transformer model. Internet access is needed for this initial model download. The model is cached locally for later runs.

The program prints:
- NumPy and FAISS search times
- HNSW recall and timing for each parameter setting
- Filtered search results and payloads
- Point counts before and after re-upsert
- Point count after deletion
- Similarity metric comparison

It also creates `recall_vs_time.png` in the project folder.

## Project structure

```text
Week6_Day2_No_Database/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── recall_vs_time.png   # generated after running main.py
```

## Concepts

### Recall@10
Recall@10 is the fraction of the exact top-10 neighbours that also appear in the approximate top-10 results.

### HNSW parameters
- `M`: maximum graph connections per node (higher values generally use more memory and can improve connectivity).
- `efSearch`: search candidate-list size (higher values generally improve recall but take more time).

### HNSW in five lines
1. HNSW builds a graph with multiple layers.
2. Upper layers contain fewer nodes and help navigation over long distances.
3. Search starts at an entry point in the top layer.
4. It moves to closer neighbours and descends layer by layer.
5. The bottom layer performs a more detailed search to return the nearest vectors.

### Similarity metrics
- Cosine similarity: compares vector direction; larger is more similar.
- L2 distance: Euclidean distance; smaller is closer.
- Inner product: dot product; larger is preferred when vectors and model use that scoring convention.

## GitHub submission

Create a new GitHub repository named `Week6-Day2-Embeddings-Vector-Search`.

Then, from this project folder:

```powershell
git init
git add .
git commit -m "Complete Week 6 Day 2 vector search tasks"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Week6-Day2-Embeddings-Vector-Search.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

Do not commit `.venv/` or large downloaded model-cache folders. They are excluded by `.gitignore`.

## Suggested screenshots

1. Terminal output showing NumPy, FAISS Flat and HNSW timings/Recall@10.
2. `recall_vs_time.png`.
3. Terminal output showing filtered HR results.
4. Terminal output showing `Points before re-upsert: 30` and `Points after re-upsert: 30`.
5. Terminal output showing deletion and final point count.

Because this is the no-database version, do not submit a Qdrant dashboard or PostgreSQL EXPLAIN screenshot as if you had run those services.
