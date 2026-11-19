"""
Week 6 Day 2 - Database-free implementation
Covers:
1) Exact vs approximate nearest-neighbour search (NumPy + FAISS Flat + FAISS HNSW)
2) In-memory vector store: 384-d embeddings, payloads, filtering, upsert, delete,
   category index, and cosine/L2/inner-product comparisons.

No PostgreSQL, pgvector, Qdrant server, or Docker is required.
FAISS and Sentence Transformers are local Python libraries, not databases.
"""

from __future__ import annotations

import time
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# --------------------------- Configuration ---------------------------

RANDOM_SEED = 42
NUM_VECTORS = 50_000
VECTOR_DIM = 128
TOP_K = 10
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def require_faiss():
    if faiss is None:
        raise RuntimeError(
            "FAISS is not installed. Run: pip install faiss-cpu\n"
            "If installation fails on Windows/Python 3.14, use Python 3.11 or 3.12."
        )


def require_sentence_transformers():
    if SentenceTransformer is None:
        raise RuntimeError(
            "Sentence Transformers is not installed. Run: pip install sentence-transformers"
        )


# =====================================================================
# TASK 1 - EXACT VS APPROXIMATE SEARCH
# =====================================================================

def numpy_exact_search(vectors: np.ndarray, query: np.ndarray, k: int):
    """Exact L2 nearest-neighbour search using NumPy."""
    start = time.perf_counter()
    distances = np.sum((vectors - query) ** 2, axis=1)
    ids = np.argpartition(distances, k)[:k]
    ids = ids[np.argsort(distances[ids])]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return ids, distances[ids], elapsed_ms


def recall_at_k(reference_ids: np.ndarray, found_ids: np.ndarray) -> float:
    """Fraction of exact top-k IDs found by the approximate search."""
    return len(set(reference_ids.tolist()) & set(found_ids.tolist())) / len(reference_ids)


def run_ann_experiment():
    print("\n" + "=" * 68)
    print("TASK 1: EXACT VS APPROXIMATE NEAREST-NEIGHBOUR SEARCH")
    print("=" * 68)

    require_faiss()
    rng = np.random.default_rng(RANDOM_SEED)

    vectors = rng.random((NUM_VECTORS, VECTOR_DIM), dtype=np.float32)
    query = rng.random((VECTOR_DIM,), dtype=np.float32)

    # Exact NumPy baseline
    exact_ids, exact_distances, numpy_ms = numpy_exact_search(vectors, query, TOP_K)
    print(f"NumPy exact search: {numpy_ms:.3f} ms")

    # FAISS exact search
    flat_index = faiss.IndexFlatL2(VECTOR_DIM)
    flat_index.add(vectors)
    start = time.perf_counter()
    flat_distances, flat_ids = flat_index.search(query.reshape(1, -1), TOP_K)
    flat_ms = (time.perf_counter() - start) * 1000
    flat_ids = flat_ids[0]
    flat_distances = flat_distances[0]
    flat_recall = recall_at_k(exact_ids, flat_ids)
    print(f"FAISS IndexFlatL2: {flat_ms:.3f} ms | Recall@10: {flat_recall:.3f}")

    # HNSW parameter experiments
    experiment_rows = []
    settings = [
        {"M": 8, "efSearch": 16},
        {"M": 16, "efSearch": 32},
        {"M": 32, "efSearch": 64},
        {"M": 32, "efSearch": 128},
        {"M": 48, "efSearch": 256},
    ]

    print("\nHNSW parameter experiments:")
    print("M\t efSearch\t time (ms)\t Recall@10")

    for setting in settings:
        hnsw = faiss.IndexHNSWFlat(VECTOR_DIM, setting["M"])
        hnsw.hnsw.efConstruction = 100
        hnsw.hnsw.efSearch = setting["efSearch"]
        hnsw.add(vectors)

        start = time.perf_counter()
        distances, ids = hnsw.search(query.reshape(1, -1), TOP_K)
        elapsed_ms = (time.perf_counter() - start) * 1000

        ids = ids[0]
        recall = recall_at_k(exact_ids, ids)
        experiment_rows.append((setting["M"], setting["efSearch"], elapsed_ms, recall))
        print(f'{setting["M"]}\t {setting["efSearch"]}\t\t {elapsed_ms:.3f}\t\t {recall:.3f}')

    # Plot recall against query latency
    times = [row[2] for row in experiment_rows]
    recalls = [row[3] for row in experiment_rows]
    labels = [f"M={row[0]}, ef={row[1]}" for row in experiment_rows]

    plt.figure(figsize=(9, 5))
    plt.plot(times, recalls, marker="o")
    for x, y, label in zip(times, recalls, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 6))
    plt.xlabel("Search time (milliseconds)")
    plt.ylabel("Recall@10")
    plt.title("FAISS HNSW: Recall vs Search Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("recall_vs_time.png", dpi=150)
    plt.show()

    print("\nSaved plot: recall_vs_time.png")
    print("\nHNSW in five lines:")
    print("1. HNSW builds a graph with multiple layers.")
    print("2. Upper layers contain fewer nodes and help navigation over long distances.")
    print("3. Search starts at an entry point in the top layer.")
    print("4. It moves to closer neighbours and descends layer by layer.")
    print("5. The bottom layer performs a more detailed search to return the nearest vectors.")

    print("\nExact vs ANN trade-off:")
    print("| Method | Accuracy | Speed | Memory |")
    print("| NumPy exact / FAISS Flat | Exact | Can be slower at scale | Stores all vectors |")
    print("| FAISS HNSW | Approximate; measure Recall@10 | Usually faster for large indexes | Extra graph memory |")


# =====================================================================
# TASK 2 - IN-MEMORY VECTOR STORE (QDRANT / PGVECTOR CONCEPTS)
# =====================================================================

class InMemoryVectorStore:
    """
    Small local vector store using Python dictionaries and NumPy.
    This demonstrates vector-store concepts; it is not a Qdrant server.
    """

    def __init__(self):
        self.points: Dict[int, Dict[str, Any]] = {}
        self.category_index: Dict[str, set] = {}

    def _index_add(self, point_id: int, category: str):
        self.category_index.setdefault(category, set()).add(point_id)

    def _index_remove(self, point_id: int, category: str):
        ids = self.category_index.get(category)
        if ids is not None:
            ids.discard(point_id)
            if not ids:
                self.category_index.pop(category, None)

    def upsert(self, point_id: int, text: str, vector: np.ndarray, payload: dict):
        """Insert or replace a point without increasing count for an existing ID."""
        if point_id in self.points:
            old_category = self.points[point_id]["payload"]["category"]
            self._index_remove(point_id, old_category)

        self.points[point_id] = {
            "text": text,
            "vector": np.asarray(vector, dtype=np.float32),
            "payload": dict(payload),
        }
        self._index_add(point_id, payload["category"])

    def count(self) -> int:
        return len(self.points)

    def filter_ids(self, category=None, min_page=None) -> List[int]:
        """Apply category == value and/or page >= min_page."""
        if category is not None:
            candidate_ids = set(self.category_index.get(category, set()))
        else:
            candidate_ids = set(self.points.keys())

        result = []
        for point_id in candidate_ids:
            payload = self.points[point_id]["payload"]
            if min_page is not None and payload["page"] < min_page:
                continue
            result.append(point_id)
        return sorted(result)

    def search(self, query_vector: np.ndarray, limit=3, category=None, min_page=None):
        """Cosine similarity search, optionally filtered by payload."""
        candidate_ids = self.filter_ids(category=category, min_page=min_page)
        if not candidate_ids:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        results = []

        for point_id in candidate_ids:
            item = self.points[point_id]
            vector = item["vector"]
            denom = query_norm * np.linalg.norm(vector)
            score = float(np.dot(query, vector) / denom) if denom else 0.0
            results.append({
                "id": point_id,
                "score": score,
                "payload": item["payload"],
                "text": item["text"],
            })

        return sorted(results, key=lambda row: row["score"], reverse=True)[:limit]

    def delete_by_filter(self, category=None):
        """Delete all points matching a category."""
        if category is None:
            raise ValueError("Provide a category to avoid accidental deletion.")
        ids_to_delete = list(self.category_index.get(category, set()))
        for point_id in ids_to_delete:
            self.points.pop(point_id, None)
        self.category_index.pop(category, None)
        return len(ids_to_delete)


def run_distance_comparison(vectors: np.ndarray, query: np.ndarray, texts: list):
    """Compare cosine, L2 and inner-product rankings for the same query."""
    print("\nDistance metric comparison (top 3):")

    q_norm = query / (np.linalg.norm(query) + 1e-12)
    v_norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)

    cosine_scores = v_norms @ q_norm
    l2_distances = np.sum((vectors - query) ** 2, axis=1)
    inner_scores = vectors @ query

    rankings = [
        ("Cosine", np.argsort(-cosine_scores)),
        ("L2", np.argsort(l2_distances)),
        ("Inner product", np.argsort(-inner_scores)),
    ]

    for name, order in rankings:
        print(f"\n{name}:")
        for idx in order[:3]:
            metric = cosine_scores[idx] if name == "Cosine" else (
                l2_distances[idx] if name == "L2" else inner_scores[idx]
            )
            print(f"  score={metric:.4f} | {texts[idx]}")


def run_in_memory_vector_store():
    print("\n" + "=" * 68)
    print("TASK 2: IN-MEMORY VECTOR STORE - FILTERING AND MANAGEMENT")
    print("=" * 68)

    require_sentence_transformers()
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 30 sample documents; exactly four HR points for the expected HR result.
    documents = [
        ("How do I apply for annual leave?", "HR"),
        ("Where can I find the employee handbook?", "HR"),
        ("How can I update my personal details?", "HR"),
        ("What is the company's work from home policy?", "HR"),
        ("How do I reset my company laptop?", "IT"),
        ("The office Wi-Fi is not connecting.", "IT"),
        ("How can I request software access?", "IT"),
        ("My monitor is not working.", "IT"),
        ("How do I submit a travel expense?", "Finance"),
        ("When will my salary be credited?", "Finance"),
        ("How can I download my payslip?", "Finance"),
        ("What is the reimbursement process?", "Finance"),
        ("How do I contact the sales team?", "Sales"),
        ("Where can I find the product catalogue?", "Sales"),
        ("How do I request a customer demo?", "Sales"),
        ("What are the current product features?", "Sales"),
        ("How do I report a workplace safety issue?", "Operations"),
        ("Where is the office emergency plan?", "Operations"),
        ("How do I request office supplies?", "Operations"),
        ("What are the building access timings?", "Operations"),
        ("How do I register for employee training?", "Training"),
        ("Where can I find course materials?", "Training"),
        ("How do I get a training completion certificate?", "Training"),
        ("When is the next technical workshop?", "Training"),
        ("How do I report a website bug?", "IT"),
        ("How can I request a new keyboard?", "IT"),
        ("What is the client meeting schedule?", "Sales"),
        ("How do I submit a vendor invoice?", "Finance"),
        ("Where can I find the company holiday list?", "Operations"),
        ("How do I update my emergency contact?", "Operations"),
    ]

    texts = [item[0] for item in documents]
    categories = [item[1] for item in documents]
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=False,
        show_progress_bar=True
    ).astype(np.float32)

    store = InMemoryVectorStore()

    # Stable integer IDs and required payload fields: source, page, category.
    for i, (text, category) in enumerate(documents, start=1):
        payload = {
            "source": "employee_handbook.pdf",
            "page": i,
            "category": category,
        }
        store.upsert(i, text, embeddings[i - 1], payload)

    print(f"\nPoints before re-upsert: {store.count()}")

    query_text = "How do I request leave from work?"
    query_vector = model.encode([query_text], convert_to_numpy=True)[0].astype(np.float32)

    print("\nUnfiltered semantic search:")
    for result in store.search(query_vector, limit=5):
        print(
            f"ID: {result['id']} | Score: {result['score']:.4f} | "
            f"Payload: {result['payload']} | Text: {result['text']}"
        )

    # Payload filter 1: category == HR (the PDF's expected output is 4 results).
    hr_ids = store.filter_ids(category="HR")
    hr_results = store.search(query_vector, limit=10, category="HR")
    print(f"\\nFiltered search (category=HR): {len(hr_results)} results | IDs: {hr_ids}")
    for result in hr_results:
        print(
            f"ID: {result['id']} | Score: {result['score']:.4f} | "
            f"Payload: {result['payload']} | Text: {result['text']}"
        )

    # Payload filter 2: numeric range page >= 3.
    page_ids = store.filter_ids(min_page=3)
    print(f"\\nRange filter (page>=3): {len(page_ids)} results | IDs: {page_ids}")

    # Demonstrate combining both filters.
    combined_ids = store.filter_ids(category="HR", min_page=3)
    print(f"Combined filter (category=HR AND page>=3): {len(combined_ids)} results | IDs: {combined_ids}")

    # Re-upsert the same IDs with changed text: count should remain 30.
    for i, (old_text, category) in enumerate(documents, start=1):
        changed_text = old_text + " (updated version)"
        changed_vector = model.encode(
            [changed_text], convert_to_numpy=True
        )[0].astype(np.float32)
        payload = {
            "source": "employee_handbook_updated.pdf",
            "page": i,
            "category": category,
        }
        store.upsert(i, changed_text, changed_vector, payload)

    print(f"\nPoints after re-upsert: {store.count()}")
    print("Upsert is idempotent for an existing ID: the count remains 30.")

    # Category index is maintained by the store for faster category candidate lookup.
    print(f"\nCategory index keys: {sorted(store.category_index.keys())}")
    print("The category index avoids scanning every point just to find a category.")

    # Delete by filter and verify count.
    deleted = store.delete_by_filter(category="Training")
    print(f"\nDeleted Training points: {deleted}")
    print(f"Points after deletion: {store.count()}")

    # Nearest-three comparison using the 20 first embedded sentences.
    run_distance_comparison(
        embeddings[:20],
        query_vector,
        texts[:20],
    )

    print("\npgvector vs Qdrant (conceptual comparison):")
    print("| Factor | pgvector | Qdrant |")
    print("| Setup | PostgreSQL + extension | Dedicated vector database |")
    print("| Filtering | SQL WHERE clauses | Payload filters |")
    print("| Scale | Fits PostgreSQL workloads | Purpose-built vector search |")
    print("| Choose when | Your app already uses PostgreSQL | You want a dedicated vector-search service |")
    print("\nThis project uses neither product; it demonstrates the concepts in local memory.")


def main():
    print("WEEK 6 DAY 2 - DATABASE-FREE PYTHON PROJECT")
    print("No PostgreSQL, pgvector, Qdrant server, or Docker.")
    run_ann_experiment()
    run_in_memory_vector_store()
    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()
