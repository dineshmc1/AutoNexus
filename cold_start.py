"""
cold_start.py
=============
Phase 4.2 — Adaptive Cold-Start Strategy.

Given a new dataset embedding, decides whether to use **memory-based**
retrieval (similar datasets exist in the knowledge base) or fall back to
a **cold-start** strategy (broad AutoML search / LLM shortlisting).

The decision is driven by an adaptive threshold ε(D) computed from the
distribution of cosine similarities to the top-K nearest neighbours
stored in a FAISS index.

Public API
----------
- ``ColdStartConfig``        — dataclass with all tuneable hyperparameters
- ``ColdStartResult``        — typed dictionary returned by the router
- ``adaptive_cold_start()``  — main entry point
- ``build_faiss_index()``    — convenience helper to create a FAISS index
- ``ColdStartLogger``        — structured logging for paper-ready metrics

Embedding compatibility:
    Expects float32 vectors of dimension ``dataset_embedding.EMBEDDING_DIM``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import pickle
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

DEFAULT_INDEX_PATH = "memory_store.faiss"
DEFAULT_METADATA_PATH = "memory_store.pkl"

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import faiss — fall back to a stub so the module stays importable
# even if faiss is not installed (useful for testing the decision logic).
# ---------------------------------------------------------------------------
try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]
    logger.warning(
        "FAISS is not installed.  cold_start.build_faiss_index() will "
        "raise at runtime.  Install with: pip install faiss-cpu"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ColdStartConfig:
    """All tuneable hyperparameters for the cold-start router.

    Attributes
    ----------
    k_neighbors : int
        Number of nearest neighbours to retrieve from FAISS (default 10).
    lambda_sensitivity : float
        Controls how aggressively ε(D) separates memory from cold-start.
        Higher λ → lower threshold → more memory-based decisions.
        Default 0.5.
    memory_models_count : int
        Number of top models to select from memory (memory path).
        Typical range: 3–5.  Default 3.
    fallback_models_count : int
        Number of candidate models to evaluate in cold-start path.
        Typical range: 5–8.  Default 5.
    use_top_k_mean : bool
        If True, compute S(D, M) as the mean of the top-3 similarities
        (more robust).  If False, use max similarity.  Default True.
    top_k_for_score : int
        How many top similarities to average when ``use_top_k_mean=True``.
        Default 3.
    """

    k_neighbors: int = 10
    lambda_sensitivity: float = 0.5
    memory_models_count: int = 3
    fallback_models_count: int = 5
    use_top_k_mean: bool = True
    top_k_for_score: int = 3
    
    alpha: float = 0.6
    beta: float = 0.3
    gamma: float = 0.1
    recency_decay_days: float = 30.0

    def __post_init__(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if self.top_k_for_score < 1:
            raise ValueError("top_k_for_score must be >= 1")
        if self.memory_models_count < 1:
            raise ValueError("memory_models_count must be >= 1")
        if self.fallback_models_count < 1:
            raise ValueError("fallback_models_count must be >= 1")
        if abs(self.alpha + self.beta + self.gamma - 1.0) > 0.01:
            raise ValueError(f"alpha + beta + gamma must sum to 1.0, got {self.alpha + self.beta + self.gamma}")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ColdStartResult(dict):
    """Typed dictionary returned by :func:`adaptive_cold_start`.

    Keys
    ----
    decision : str
        ``"memory"`` or ``"cold_start"``.
    epsilon : float
        Adaptive threshold ε(D).
    similarity_score : float
        Overall similarity score S(D, M).
    mu_s : float
        Mean of top-K cosine similarities.
    sigma_s : float
        Standard deviation of top-K cosine similarities.
    similarities : list[float]
        All K cosine similarities (sorted descending).
    neighbor_indices : list[int]
        FAISS indices of the K nearest neighbours.
    models_selected : list[str]
        Names / IDs of the models selected for evaluation.
    num_models_evaluated : int
        Count of models that will be evaluated downstream.
    """


# ---------------------------------------------------------------------------
# Core maths helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    similarity = dot / (norm_a * norm_b)
    return float(np.clip(similarity, 0.0, 1.0))

def normalize_scores(values: list, higher_is_better: bool = True) -> list:
    """
    Min-max normalize a list of floats to [0, 1].
    If higher_is_better=False (e.g. regression MSE which is negative),
    flip the sign first so that less negative = higher normalized score.
    Returns list of floats in [0, 1].
    Edge case: if all values are identical, return list of 1.0s.
    """
    if not values:
        return []
    
    vals = np.array(values, dtype=float)
    if not higher_is_better:
        vals = -vals
        
    val_min = np.min(vals)
    val_max = np.max(vals)
    
    if val_max == val_min:
        return [1.0] * len(values)
        
    normalized = (vals - val_min) / (val_max - val_min)
    return normalized.tolist()

def compute_recency_score(record_time: float, decay_days: float = 30.0) -> float:
    """
    Computes recency score using exponential decay.
    
    Formula: R = exp(-age_in_days / decay_days)
    
    Where age_in_days = (current_time - record_time) / 86400
    
    Returns float in (0, 1]:
        - Record added today: R ≈ 1.0
        - Record added decay_days ago: R ≈ 0.37
        - Record added 3*decay_days ago: R ≈ 0.05
    
    If record_time is 0 or None: return 0.5 (neutral score)
    Use time.time() for current time.
    """
    if not record_time:
        return 0.5
    age_in_days = (time.time() - record_time) / 86400.0
    if age_in_days < 0:
        age_in_days = 0.0
    return float(np.exp(-age_in_days / decay_days))

def compute_performance_score(records: list, problem_type: str) -> list:
    """
    Extracts and normalizes performance scores from a list of records.
    
    For classification: score is accuracy-like (higher is better, 0 to 1).
        normalize with higher_is_better=True
    
    For regression: score is negative MSE (more negative = worse).
        normalize with higher_is_better=False
        (so that -0.1 gets higher score than -50000)
    
    Returns list of normalized floats in [0, 1], one per record.
    If a record has no score or score is None: assign 0.5 (neutral).
    """
    scores = []
    higher_is_better = (problem_type == "classification")
    for r in records:
        s = r.metadata.get("score")
        if s is None:
            scores.append(0.5)
        else:
            scores.append(float(s))
            
    # Normalize valid scores
    valid_indices = [i for i, s in enumerate(scores) if records[i].metadata.get("score") is not None]
    if valid_indices:
        valid_scores = [scores[i] for i in valid_indices]
        norm_valid = normalize_scores(valid_scores, higher_is_better=higher_is_better)
        for idx, norm_val in zip(valid_indices, norm_valid):
            scores[idx] = norm_val
    return scores

def compute_adaptive_weighted_score(
    query_vec: np.ndarray,
    records: list,
    similarities: list,
    problem_type: str,
    config: ColdStartConfig
) -> list:
    """
    Computes the weighted multi-factor score for each candidate record.
    
    Steps:
    1. Normalize similarity scores to [0, 1] using normalize_scores()
       (similarities are already cosine similarities, clip to [0,1] first)
    2. Compute performance scores using compute_performance_score()
    3. Compute recency scores using compute_recency_score() for each record
    4. Combine: Score = alpha * S + beta * P + gamma * R
    5. Return list of combined scores, one per record
    
    Log each record's breakdown at DEBUG level:
        [Retrieval] key=openml_X | sim=0.94 | perf=0.87 | rec=0.92 | 
        combined=0.917
    """
    sims_norm = normalize_scores([float(np.clip(s, 0.0, 1.0)) for s in similarities], higher_is_better=True)
    perfs_norm = compute_performance_score(records, problem_type)
    recs_norm = [compute_recency_score(r.metadata.get("time"), config.recency_decay_days) for r in records]
    
    combined = []
    for i, r in enumerate(records):
        s = sims_norm[i]
        p = perfs_norm[i]
        rec = recs_norm[i]
        score = config.alpha * s + config.beta * p + config.gamma * rec
        combined.append(score)
        logger.debug(
            f"[Retrieval] key={r.key} | sim={s:.4f} | perf={p:.4f} | rec={rec:.4f} | combined={score:.4f}"
        )
    return combined


def compute_similarity_scores(
    query: np.ndarray,
    memory_embeddings: np.ndarray,
    neighbor_indices: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between *query* and each neighbour.

    Parameters
    ----------
    query : np.ndarray
        1-D float32 embedding of the new dataset.
    memory_embeddings : np.ndarray
        2-D float32 matrix of all stored embeddings.
    neighbor_indices : np.ndarray
        1-D array of row indices into *memory_embeddings*.

    Returns
    -------
    np.ndarray
        1-D array of cosine similarities, one per neighbour (same order).
    """
    sims = np.array(
        [_cosine_similarity(query, memory_embeddings[idx]) for idx in neighbor_indices],
        dtype=np.float64,
    )
    return sims


def compute_adaptive_threshold(
    similarities: np.ndarray,
    lambda_sensitivity: float = 0.5,
) -> Tuple[float, float, float]:
    """Compute ε(D) = μ_S - λ · σ_S.

    Returns
    -------
    epsilon : float
    mu_s : float
    sigma_s : float
    """
    mu_s = float(np.mean(similarities))
    sigma_s = float(np.std(similarities))
    epsilon = mu_s - lambda_sensitivity * sigma_s
    return epsilon, mu_s, sigma_s


def compute_overall_similarity(
    similarities: np.ndarray,
    use_top_k_mean: bool = True,
    top_k: int = 3,
) -> float:
    """Compute S(D, M).

    Parameters
    ----------
    similarities : array-like
        Cosine similarities (descending order preferred).
    use_top_k_mean : bool
        If True, return mean of top-*k* similarities (more robust).
    top_k : int
        How many top similarities to average.

    Returns
    -------
    float
        The overall similarity score.
    """
    sorted_desc = np.sort(similarities)[::-1]
    if use_top_k_mean:
        k = min(top_k, len(sorted_desc))
        return float(np.mean(sorted_desc[:k]))
    return float(sorted_desc[0])


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

def build_faiss_index(
    embeddings: np.ndarray,
    use_ip: bool = False,
) -> Any:
    """Build a FAISS index from a 2-D float32 embedding matrix.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(N, D)`` with dtype ``float32``.
    use_ip : bool
        If True, use inner-product index (IndexFlatIP); otherwise use
        L2 (IndexFlatL2).  For cosine similarity, normalise embeddings
        before indexing and set ``use_ip=True``.

    Returns
    -------
    faiss.IndexFlat
    """
    if faiss is None:
        raise ImportError(
            "FAISS is not installed.  Run: pip install faiss-cpu"
        )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info(
        "FAISS index built: %d vectors, dim=%d, metric=%s",
        index.ntotal, dim, "IP" if use_ip else "L2",
    )
    return index


def search_faiss(
    index: Any,
    query: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search FAISS index for top-K nearest neighbours.

    Parameters
    ----------
    index : faiss.Index
    query : np.ndarray
        1-D or 2-D float32 query vector(s).
    k : int
        Number of neighbours to retrieve.

    Returns
    -------
    distances : np.ndarray   — shape ``(1, k)``
    indices   : np.ndarray   — shape ``(1, k)``
    """
    if faiss is None:
        raise ImportError("FAISS is not installed.")
    query = np.ascontiguousarray(
        query.reshape(1, -1) if query.ndim == 1 else query,
        dtype=np.float32,
    )
    # Clamp k to index size
    k = min(k, index.ntotal)
    distances, indices = index.search(query, k)
    return distances, indices


# ---------------------------------------------------------------------------
# Memory-based model registry  (lightweight in-memory store)
# ---------------------------------------------------------------------------

@dataclass
class DatasetRecord:
    """A single record in the memory store linking a dataset to its models."""
    dataset_id: str
    embedding: np.ndarray
    models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"openml_{self.dataset_id}"


class MemoryStore:
    """Lightweight in-memory store for dataset → model mappings.

    Wraps a FAISS index plus metadata so the cold-start router can
    retrieve the best models from similar datasets.
    """

    def __init__(self) -> None:
        self.records: List[DatasetRecord] = []
        self._index: Optional[Any] = None

    # -- mutators -----------------------------------------------------------

    def add(
        self,
        dataset_id: str,
        embedding: np.ndarray,
        models: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a dataset and its best-performing models."""
        embedding_arr = np.asarray(embedding, dtype=np.float32)
        self.records.append(
            DatasetRecord(
                dataset_id=dataset_id,
                embedding=embedding_arr,
                models=list(models),
                metadata=metadata or {},
            )
        )
        if self._index is not None:
            self._index.add(embedding_arr.reshape(1, -1))

    def build_index(self) -> None:
        """(Re)build the FAISS index from current records."""
        if not self.records:
            raise ValueError("MemoryStore is empty — nothing to index.")
        matrix = np.vstack([r.embedding for r in self.records]).astype(np.float32)
        self._index = build_faiss_index(matrix, use_ip=False)

    def rebuild_index(self, new_vectors: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Rebuild index, optionally updating embeddings first."""
        if new_vectors is not None:
            for r in self.records:
                if r.key in new_vectors:
                    r.embedding = np.asarray(new_vectors[r.key], dtype=np.float32)
        self.build_index()

    # -- persistence & management -------------------------------------------

    def save_index(self, index_path: str = DEFAULT_INDEX_PATH, metadata_path: str = DEFAULT_METADATA_PATH) -> None:
        if self._index is None:
            raise RuntimeError("Cannot save: FAISS index is not built.")
        faiss.write_index(self._index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.records, f)
        print(f"Saved {len(self.records)} records to {index_path} and {metadata_path}")

    def load_index(self, index_path: str = DEFAULT_INDEX_PATH, metadata_path: str = DEFAULT_METADATA_PATH) -> int:
        with open(metadata_path, 'rb') as f:
            loaded_records = pickle.load(f)
            
        existing_keys = set(self.get_keys())
        added_count = 0
        for r in loaded_records:
            if r.key not in existing_keys:
                self.records.append(r)
                added_count += 1
                
        if self.records:
            self.build_index()
            
        return added_count

    def remove_entry(self, dataset_key: str) -> bool:
        original_len = len(self.records)
        self.records = [r for r in self.records if r.key != dataset_key]
        if len(self.records) < original_len:
            if self.records:
                self.build_index()
            else:
                self._index = None
            return True
        return False

    def remove_entries(self, dataset_keys: list) -> int:
        keys_to_remove = set(dataset_keys)
        original_len = len(self.records)
        self.records = [r for r in self.records if r.key not in keys_to_remove]
        removed_count = original_len - len(self.records)
        if removed_count > 0:
            if self.records:
                self.build_index()
            else:
                self._index = None
        return removed_count

    def get_keys(self) -> list:
        return [r.key for r in self.records]

    def get_all_vectors(self) -> np.ndarray:
        if not self.records:
            return np.empty((0, 17), dtype=np.float32)
        return self.embeddings

    def get_all_metadata(self) -> list:
        if not self.records:
            return []
        
        meta_list = []
        for r in self.records:
            d = r.metadata.copy()
            d['dataset_id'] = r.dataset_id
            d['key'] = r.key
            meta_list.append(d)
        return meta_list

    # -- queries ------------------------------------------------------------

    @property
    def index(self) -> Any:
        if self._index is None:
            self.build_index()
        return self._index

    @property
    def embeddings(self) -> np.ndarray:
        return np.vstack([r.embedding for r in self.records]).astype(np.float32)

    def get_models_for_indices(
        self, indices: np.ndarray, top_n: int = 3
    ) -> List[str]:
        """Collect the best models from the top-N nearest datasets.

        Models that appear more frequently are ranked higher (voting).
        """
        model_votes: Dict[str, int] = {}
        for idx in indices[:top_n]:
            idx = int(idx)
            if 0 <= idx < len(self.records):
                for m in self.records[idx].models:
                    model_votes[m] = model_votes.get(m, 0) + 1

        # Sort by vote count (desc), then alphabetically for determinism
        ranked = sorted(model_votes.keys(), key=lambda m: (-model_votes[m], m))
        return ranked

    def __len__(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Structured logger for paper-ready metrics
# ---------------------------------------------------------------------------

class ColdStartLogger:
    """Collects cold-start decision logs for analysis / paper tables.

    Each call to :meth:`log` stores a row; use :meth:`to_json` or
    :meth:`to_dataframe` to export.
    """

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def log(self, result: ColdStartResult) -> None:
        """Append a decision record."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mu_s": result["mu_s"],
            "sigma_s": result["sigma_s"],
            "epsilon": result["epsilon"],
            "similarity_score": result["similarity_score"],
            "decision": result["decision"],
            "num_models_evaluated": result["num_models_evaluated"],
            "models_selected": result["models_selected"],
        }
        self.entries.append(entry)
        logger.info(
            "[ColdStart] decision=%s  S(D,M)=%.4f  ε=%.4f  μ=%.4f  σ=%.4f  "
            "models=%d",
            entry["decision"],
            entry["similarity_score"],
            entry["epsilon"],
            entry["mu_s"],
            entry["sigma_s"],
            entry["num_models_evaluated"],
        )

    def to_json(self, path: str) -> str:
        """Write all entries to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh, indent=2)
        return os.path.abspath(path)

    def to_dataframe(self):
        """Return entries as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.entries)


# ---------------------------------------------------------------------------
# Default fallback model lists
# ---------------------------------------------------------------------------

# These mirror the model keys used in model_trainer.py.
_DEFAULT_MEMORY_MODELS: List[str] = ["rf", "gb", "lightgbm"]
_DEFAULT_FALLBACK_MODELS_CLF: List[str] = [
    "logistic", "rf", "gb", "lgbm_clf", "xgb_clf",
]
_DEFAULT_FALLBACK_MODELS_REG: List[str] = [
    "ridge", "rf_reg", "gb_reg", "lgbm_reg", "xgb_reg",
]


def get_fallback_models(
    problem_type: str = "classification",
    count: int = 5,
) -> List[str]:
    """Return a shortlist of candidate models for the cold-start path.

    In production this could call an LLM; here we return the default
    catalogue ordered by general robustness.
    """
    pool = (
        _DEFAULT_FALLBACK_MODELS_CLF
        if problem_type == "classification"
        else _DEFAULT_FALLBACK_MODELS_REG
    )
    return pool[:count]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def adaptive_cold_start(
    query_embedding: np.ndarray,
    memory: MemoryStore,
    config: Optional[ColdStartConfig] = None,
    problem_type: str = "classification",
    cs_logger: Optional[ColdStartLogger] = None,
) -> ColdStartResult:
    """Route a new dataset to memory-based retrieval or cold-start fallback.

    Parameters
    ----------
    query_embedding : np.ndarray
        1-D float32 embedding of the new dataset (from
        ``compute_dataset_embedding``).
    memory : MemoryStore
        The populated memory store with FAISS index.
    config : ColdStartConfig, optional
        Tuneable hyperparameters.  Defaults are used if ``None``.
    problem_type : str
        ``"classification"`` or ``"regression"``.
    cs_logger : ColdStartLogger, optional
        If provided, the decision is automatically logged.

    Returns
    -------
    ColdStartResult
        Dict-like with keys: decision, epsilon, similarity_score,
        mu_s, sigma_s, similarities, neighbor_indices, models_selected,
        num_models_evaluated.
    """
    if config is None:
        config = ColdStartConfig()

    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    # --- Edge case: empty memory → always cold-start ---
    if len(memory) == 0:
        logger.info("[ColdStart] Memory is empty → cold-start.")
        models = get_fallback_models(problem_type, config.fallback_models_count)
        result = ColdStartResult(
            decision="cold_start",
            epsilon=0.0,
            similarity_score=0.0,
            combined_score=0.0,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            winning_key="",
            winning_perf=0.0,
            winning_recency=0.0,
            mu_s=0.0,
            sigma_s=0.0,
            similarities=[],
            neighbor_indices=[],
            models_selected=models,
            num_models_evaluated=len(models),
        )
        if cs_logger is not None:
            cs_logger.log(result)
        return result

    # STEP 1: Retrieve top-K neighbours from FAISS
    k = min(config.k_neighbors, len(memory))
    distances, indices = search_faiss(memory.index, query_embedding, k=k)
    neighbor_idx = indices[0]  # shape (k,)

    # STEP 1b: Compute cosine similarities
    sims = compute_similarity_scores(
        query_embedding, memory.embeddings, neighbor_idx,
    )

    # STEP 2: Retrieve records and compute combined scores
    candidates = [memory.records[i] for i in neighbor_idx]
    combined_scores = compute_adaptive_weighted_score(
        query_embedding, candidates, sims, problem_type, config
    )
    
    # STEP 3: Identify winner and threshold
    best_idx_in_k = int(np.argmax(combined_scores))
    best_combined = combined_scores[best_idx_in_k]
    best_sim = float(sims[best_idx_in_k])
    winner_record = candidates[best_idx_in_k]
    
    # Epsilon on combined scores
    mu_c = float(np.mean(combined_scores))
    sigma_c = float(np.std(combined_scores))
    epsilon = mu_c - config.lambda_sensitivity * sigma_c
    
    # Performance/recency extraction for winner
    perfs = compute_performance_score([winner_record], problem_type)
    winner_perf = perfs[0]
    winner_recency = compute_recency_score(winner_record.metadata.get("time"), config.recency_decay_days)

    # STEP 4: Decision logic
    SIMILARITY_FLOOR = 0.75
    # Use raw similarity of the winner to check the absolute floor
    if best_sim < SIMILARITY_FLOOR:
        decision = "cold_start"
        models = get_fallback_models(problem_type, config.fallback_models_count)
    elif best_combined >= epsilon:
        # HIGH CONFIDENCE — memory-based retrieval
        decision = "memory"
        sorted_pairs = sorted(zip(combined_scores, candidates), key=lambda x: x[0], reverse=True)
        models = []
        for _, rec in sorted_pairs:
            m = rec.metadata.get("best_model")
            if m:
                # Map model name to the current problem_type
                if problem_type == "regression":
                    if m in ["logistic", "linear"]: m = "ridge"
                    elif m == "rf": m = "rf_reg"
                    elif m == "gb": m = "gb_reg"
                    elif m in ["lgbm_clf", "lightgbm"]: m = "lgbm_reg"
                    elif m in ["xgb_clf", "xgboost"]: m = "xgb_reg"
                    elif m == "et_clf": m = "et_reg"
                    elif m == "knn_clf": m = "knn_reg"
                    elif m == "svc": m = "svr"
                    elif m == "dt_clf": m = "dt_reg"
                    elif m == "mlp_clf": m = "mlp_reg"
                    elif m == "ada_clf": m = "ada_reg"
                    elif m == "bag_clf": m = "bag_reg"
                else:
                    if m in ["ridge", "linear"]: m = "logistic"
                    elif m == "rf_reg": m = "rf"
                    elif m == "gb_reg": m = "gb"
                    elif m in ["lgbm_reg", "lightgbm"]: m = "lgbm_clf"
                    elif m in ["xgb_reg", "xgboost"]: m = "xgb_clf"
                    elif m == "et_reg": m = "et_clf"
                    elif m == "knn_reg": m = "knn_clf"
                    elif m == "svr": m = "svc"
                    elif m == "dt_reg": m = "dt_clf"
                    elif m == "mlp_reg": m = "mlp_clf"
                    elif m == "ada_reg": m = "ada_clf"
                    elif m == "bag_reg": m = "bag_clf"
                
                if m not in models:
                    models.append(m)
            if len(models) >= config.memory_models_count:
                break
        if not models:
            models = get_fallback_models(problem_type, config.memory_models_count)
    else:
        # COLD-START — broader search
        decision = "cold_start"
        models = get_fallback_models(problem_type, config.fallback_models_count)

    # Build result
    result = ColdStartResult(
        decision=decision,
        epsilon=round(epsilon, 6),
        similarity_score=round(best_sim, 6),
        combined_score=round(best_combined, 6),
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        winning_key=winner_record.key,
        winning_perf=round(winner_perf, 6),
        winning_recency=round(winner_recency, 6),
        mu_s=round(mu_c, 6),
        sigma_s=round(sigma_c, 6),
        similarities=[round(float(s), 6) for s in np.sort(sims)[::-1]],
        neighbor_indices=[int(i) for i in neighbor_idx],
        models_selected=models,
        num_models_evaluated=len(models),
    )

    # STEP 5: Logging
    if cs_logger is not None:
        cs_logger.log(result)

    logger.info(
        "[ColdStart] decision=%s | S(D,M)=%.4f | C(D,M)=%.4f | ε(D)=%.4f | "
        "μ_c=%.4f | σ_c=%.4f | models=%s",
        decision, best_sim, best_combined, epsilon, mu_c, sigma_c, models,
    )

    return result
