# utils/vector_store.py

from pathlib import Path

import streamlit as st

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ============================================================
# CONFIG
# ============================================================

VECTOR_DB_DIR = "vector_store"

COLLECTION_NAME = "netflix"

EMBEDDING_MODEL = "text-embedding-3-small"

# ============================================================
# EMBEDDINGS
# ============================================================


@st.cache_resource(show_spinner=False)
def get_embeddings():

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )


# ============================================================
# VECTOR STORE
# ============================================================


@st.cache_resource(show_spinner=False)
def load_vector_store():

    db_path = Path(VECTOR_DB_DIR)

    if not db_path.exists():

        raise FileNotFoundError(
            f"""
Vector store not found.

Expected location:
{VECTOR_DB_DIR}

Run:

python build_vector_store.py
"""
        )

    embeddings = get_embeddings()

    db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    return db


# ============================================================
# VALIDATION
# ============================================================


def validate_vector_store():

    try:

        db = load_vector_store()

        count = db._collection.count()

        return {
            "valid": True,
            "count": count
        }

    except Exception as e:

        return {
            "valid": False,
            "error": str(e)
        }


# ============================================================
# RETRIEVAL
# ============================================================


def semantic_search(
    query: str,
    k: int = 5
):

    db = load_vector_store()

    results = db.similarity_search_with_score(
        query,
        k=k
    )

    matches = []

    for doc, score in results:

        matches.append(
            {
                "title": doc.metadata.get(
                    "title", ""
                ),
                "genre": doc.metadata.get(
                    "genre", ""
                ),
                "director": doc.metadata.get(
                    "director", ""
                ),
                "actors": doc.metadata.get(
                    "actors", ""
                ),
                "country": doc.metadata.get(
                    "country", ""
                ),
                "language": doc.metadata.get(
                    "language", ""
                ),
                "imdb_rating": doc.metadata.get(
                    "imdb_rating", ""
                ),
                "score": float(score),
                "document": doc.page_content
            }
        )

    return matches


# ============================================================
# CONFIDENCE
# ============================================================


def retrieval_confidence(results):

    if not results:
        return 0

    scores = []

    for item in results:

        score = item.get("score", 0)

        confidence = max(
            0,
            min(
                100,
                int(
                    100 /
                    (1 + score)
                )
            )
        )

        scores.append(confidence)

    return int(sum(scores) / len(scores))


# ============================================================
# SIMPLE RETRIEVER
# ============================================================


def get_retriever(k: int = 5):

    db = load_vector_store()

    return db.as_retriever(
        search_kwargs={
            "k": k
        }
    )


@st.cache_data(show_spinner=False)
def cached_search(query: str):
    return semantic_search(query)