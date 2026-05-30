"""
utils/vector_store.py

Netflix-style semantic retrieval engine
- Uses precomputed embeddings (data/embeddings.npy)
- Uses metadata (data/metadata.parquet)
- No Chroma / no vector DB required
- Optimized for Streamlit Cloud
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")

EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
METADATA_FILE = DATA_DIR / "metadata.parquet"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MIN_SIMILARITY = 0.15


# ============================================================
# MODEL (cached)
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Missing {EMBEDDINGS_FILE}")

    emb = np.load(EMBEDDINGS_FILE)

    return emb.astype(np.float32)


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_resource
def load_metadata():

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing {METADATA_FILE}")

    df = pd.read_parquet(METADATA_FILE)

    return df.reset_index(drop=True)


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(query: str, model: SentenceTransformer):

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    return np.array(query_vector, dtype=np.float32)


# ============================================================
# SEMANTIC SEARCH (FAST + CORRECT)
# ============================================================

def semantic_search(
    query: str,
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    model: SentenceTransformer,
    k: int = 10
):

    query_vec = embed_query(query, model)

    # FAST cosine similarity (embeddings already normalized)
    scores = embeddings @ query_vec

    top_idx = np.argsort(scores)[::-1]

    results = []

    for idx in top_idx:

        score = float(scores[idx])

        if score < MIN_SIMILARITY:
            continue

        row = metadata_df.iloc[idx].to_dict()

        row["similarity"] = score
        row["embedding_idx"] = idx

        results.append(row)

        if len(results) >= k:
            break

    return results

def cached_semantic_search(query: str, k: int = 30):
    """
    Single source of truth for retrieval.
    Returns LIST[dict]
    """

    embeddings = load_embeddings()
    metadata_df = load_metadata()
    model = load_embedding_model()

    query_vec = model.encode(
        query,
        normalize_embeddings=True
    )

    sims = cosine_similarity([query_vec], embeddings)[0]

    top_idx = np.argsort(sims)[-k:][::-1]

    results = []

    for i in top_idx:
        row = metadata_df.iloc[i].to_dict()

        row["similarity"] = float(sims[i])

        results.append(row)

    return results
# ============================================================
# HYBRID SEARCH (FILTERING LAYER)
# ============================================================

def hybrid_search(
    query: str,
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    model: SentenceTransformer,
    genre: str | None = None,
    min_rating: float | None = None,
    k: int = 20
):

    candidates = semantic_search(
        query=query,
        embeddings=embeddings,
        metadata_df=metadata_df,
        model=model,
        k=100
    )

    filtered = []

    for item in candidates:

        if genre:
            g = str(item.get("genre", "")).lower()
            if genre.lower() not in g:
                continue

        if min_rating is not None:
            try:
                rating = float(item.get("imdb_rating", 0))
                if rating < min_rating:
                    continue
            except Exception:
                continue

        filtered.append(item)

    return filtered[:k]


# ============================================================
# RECOMMENDATION SCORE (NORMALIZED)
# ============================================================

def _norm(x, min_v, max_v):
    try:
        x = float(x)
        return (x - min_v) / (max_v - min_v)
    except:
        return 0.0


def recommendation_score(movie: dict):

    similarity = float(movie.get("similarity", 0))

    imdb = _norm(movie.get("imdb_rating", 0), 0, 10)
    watchtime = _norm(movie.get("watchtime_zscore", 0), -3, 3)
    evergreen = _norm(movie.get("evergreen_score", 0), 0, 100)

    return round(
        similarity * 0.60 +
        imdb * 0.15 +
        watchtime * 0.15 +
        evergreen * 0.10,
        4
    )


# ============================================================
# RANK RESULTS
# ============================================================

def rank_results(results):
    return sorted(
        results,
        key=recommendation_score,
        reverse=True
    )


# ============================================================
# SIMILAR MOVIES (FIXED INDEX ALIGNMENT)
# ============================================================

def similar_movies(
    title: str,
    metadata_df: pd.DataFrame,
    embeddings: np.ndarray,
    model: SentenceTransformer,
    k: int = 10
):

    matches = metadata_df[
        metadata_df["title"].astype(str).str.lower()
        == title.lower()
    ]

    if matches.empty:
        return []

    # IMPORTANT: reset index safety
    row_idx = int(matches.index[0])

    base_vec = embeddings[row_idx]

    scores = embeddings @ base_vec

    top_idx = np.argsort(scores)[::-1]

    results = []

    for idx in top_idx:

        if idx == row_idx:
            continue

        score = float(scores[idx])

        if score < MIN_SIMILARITY:
            continue

        row = metadata_df.iloc[idx].to_dict()
        row["similarity"] = score

        results.append(row)

        if len(results) >= k:
            break

    return results


# ============================================================
# RAG CONTEXT RETRIEVAL
# ============================================================

def retrieve_context(
    query: str,
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    model: SentenceTransformer,
    k: int = 5
):

    results = semantic_search(
        query=query,
        embeddings=embeddings,
        metadata_df=metadata_df,
        model=model,
        k=k
    )

    context_blocks = []

    for r in results:

        if "document" in r:
            context_blocks.append(r["document"])

        elif "plot" in r:
            context_blocks.append(r["plot"])

        elif "title" in r:
            context_blocks.append(r["title"])

    return "\n\n".join(context_blocks)