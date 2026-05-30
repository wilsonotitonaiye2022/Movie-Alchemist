# build_embeddings.py

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "data/netflix_sample.xlsx"
SHEET_NAME = "Additional IMDb Data"

OUTPUT_DIR = Path("data")

EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "metadata.parquet"
BUILD_METADATA_FILE = OUTPUT_DIR / "embedding_build.json"

# Recommended for Streamlit Cloud
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Optional fallback
OPENAI_MODEL = "text-embedding-3-small"

load_dotenv()

# ============================================================
# HASHING
# ============================================================


def generate_dataset_hash(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(
        df.fillna(""),
        index=False
    ).values

    return hashlib.md5(hashed).hexdigest()


# ============================================================
# LOAD DATASET
# ============================================================


def load_dataset():

    print("Loading dataset...")

    df = pd.read_excel(
        DATA_FILE,
        sheet_name=SHEET_NAME
    )
    df['title'] = df['title'].astype('string')
    df['plot'] = df['plot'].astype('string')
    df['imdb_title'] = df['imdb_title'].astype('string')

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    if "title" not in df.columns:
        raise ValueError("title column missing")

    if "plot" not in df.columns:
        raise ValueError("plot column missing")

    return df


# ============================================================
# HELPERS
# ============================================================


def safe_value(row, column):

    if column not in row:
        return ""

    value = row[column]

    if pd.isna(value):
        return ""

    return str(value)


# ============================================================
# DOCUMENT BUILDER
# ============================================================


def build_text_documents(df: pd.DataFrame):

    texts = []

    print(
        f"Building documents for {len(df):,} titles..."
    )

    for _, row in df.iterrows():

        text = f"""
TITLE
{safe_value(row,'title')}

TYPE
{safe_value(row,'type')}
{safe_value(row,'imdb_type')}

GENRE
{safe_value(row,'genre')}
{safe_value(row,'imdb_genre')}

CREATIVE TEAM
Director: {safe_value(row,'director')}
Writer: {safe_value(row,'writer')}
Actors: {safe_value(row,'actors')}

PLOT
{safe_value(row,'plot')}

AUDIENCE
Rated: {safe_value(row,'rated')}

LANGUAGE
{safe_value(row,'language')}

COUNTRY
{safe_value(row,'country')}

QUALITY SIGNALS
IMDb Rating: {safe_value(row,'imdb_rating')}
IMDb Votes: {safe_value(row,'imdb_votes')}
IMDb Rating Z Score: {safe_value(row,'imdb_rating_zscore')}

POPULARITY SIGNALS
Total Watchtime: {safe_value(row,'total_watchtime')}
Watchtime Z Score: {safe_value(row,'watchtime_zscore')}
Recency Weighted Watchtime:
{safe_value(row,'recency_weighted_watchtime')}

LONGEVITY
Evergreen Score:
{safe_value(row,'evergreen_score')}

Years Since Release:
{safe_value(row,'years_since_release')}

STRUCTURE
Runtime Minutes:
{safe_value(row,'runtime_minutes')}

Total Seasons:
{safe_value(row,'total_seasons')}

AWARDS
{safe_value(row,'awards')}

CINEMATIC PROFILE

This is a {safe_value(row,'genre')}
title with IMDb rating
{safe_value(row,'imdb_rating')}.

Known for
{safe_value(row,'evergreen_score')}
evergreen engagement.
"""

        texts.append(text.strip())

    print(
        f"Created {len(texts):,} documents."
    )

    return texts


# ============================================================
# LOCAL EMBEDDINGS
# ============================================================


def build_embeddings_local(texts):

    print(
        f"Using local embedding model: {HF_MODEL}"
    )

    from sentence_transformers import (
        SentenceTransformer
    )

    model = SentenceTransformer(
        HF_MODEL
    )

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.astype(
        np.float32
    )


# ============================================================
# OPENAI FALLBACK
# ============================================================


def build_embeddings_openai(texts):

    print(
        "Using OpenAI fallback embeddings..."
    )

    from langchain_openai import (
        OpenAIEmbeddings
    )

    model = OpenAIEmbeddings(
        model=OPENAI_MODEL
    )

    vectors = np.array(
        model.embed_documents(texts),
        dtype=np.float32
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    vectors = vectors / np.maximum(
        norms,
        1e-12
    )

    return vectors


# ============================================================
# EMBEDDING ROUTER
# ============================================================


def generate_embeddings(texts):

    try:

        return build_embeddings_local(
            texts
        )

    except Exception as e:

        print(
            f"Local model failed: {e}"
        )

        return build_embeddings_openai(
            texts
        )


# ============================================================
# BUILD METADATA
# ============================================================


def load_build_metadata():

    if not BUILD_METADATA_FILE.exists():
        return None

    with open(
        BUILD_METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_build_metadata(
    dataset_hash
):

    with open(
        BUILD_METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "dataset_hash": dataset_hash,
                "embedding_model": HF_MODEL,
                "fallback_model": OPENAI_MODEL
            },
            f,
            indent=2
        )


# ============================================================
# REBUILD CHECK
# ============================================================


def should_rebuild(dataset_hash):

    if not EMBEDDINGS_FILE.exists():
        return True

    if not METADATA_FILE.exists():
        return True

    metadata = load_build_metadata()

    if metadata is None:
        return True

    if metadata.get(
        "dataset_hash"
    ) != dataset_hash:

        print(
            "Dataset changed."
        )

        return True

    if metadata.get(
        "embedding_model"
    ) != HF_MODEL:

        print(
            "Embedding model changed."
        )

        return True

    return False


# ============================================================
# SAVE OUTPUTS
# ============================================================


def save_outputs(
    df,
    embeddings
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        EMBEDDINGS_FILE,
        embeddings.astype(
            np.float32
        )
    )

    df.to_parquet(
        METADATA_FILE,
        index=False
    )


# ============================================================
# VALIDATION
# ============================================================


def validate():

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    metadata = pd.read_parquet(
        METADATA_FILE
    )

    print(
        f"Embeddings shape: {embeddings.shape}"
    )

    print(
        f"Metadata rows: {len(metadata):,}"
    )

    assert (
        embeddings.shape[0]
        == len(metadata)
    ), (
        "Embedding count mismatch"
    )

    print(
        "Validation passed."
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 60)
    print("NETFLIX EMBEDDING BUILDER")
    print("=" * 60)

    df = load_dataset()

    dataset_hash = generate_dataset_hash(
        df
    )

    if not should_rebuild(
        dataset_hash
    ):

        print(
            "Embeddings already current."
        )

        return

    texts = build_text_documents(
        df
    )

    df["document"] = texts

    embeddings = generate_embeddings(
        texts
    )

    save_outputs(
        df,
        embeddings
    )

    save_build_metadata(
        dataset_hash
    )

    validate()

    print()
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)

    print(
        f"Rows embedded: {len(df):,}"
    )

    print(
        f"Embedding dimensions: {embeddings.shape[1]}"
    )


if __name__ == "__main__":
    main()