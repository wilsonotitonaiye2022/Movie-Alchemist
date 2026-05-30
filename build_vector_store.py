import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "data/netflix_sample.xlsx"

SHEET_NAME = 'Additional IMDb Data'

VECTOR_DB_DIR = "vector_store"

COLLECTION_NAME = "netflix"

EMBEDDING_MODEL = "text-embedding-3-small"


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

METADATA_FILE = os.path.join(
    VECTOR_DB_DIR,
    "metadata.json"
)

# ============================================================
# DATASET HASHING
# ============================================================


def generate_dataset_hash(df: pd.DataFrame) -> str:
    """
    Generate deterministic hash for dataset.
    Used to determine whether embeddings
    need rebuilding.
    """

    hashed = pd.util.hash_pandas_object(
        df.fillna(""),
        index=False
    ).values

    return hashlib.md5(hashed).hexdigest()


# ============================================================
# LOAD DATASET
# ============================================================


def load_dataset() -> pd.DataFrame:

    print("Loading dataset...")

    df = pd.read_excel(DATA_FILE, sheet_name = SHEET_NAME)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    if "title" not in df.columns:
        raise ValueError(
            "Dataset must contain title column"
        )

    if "plot" not in df.columns:
        raise ValueError(
            "Dataset must contain plot column"
        )

    return df


# ============================================================
# DOCUMENT BUILDER
# ============================================================


def safe_value(row, column):

    if column not in row:
        return ""

    value = row[column]

    if pd.isna(value):
        return ""

    return str(value)

def build_documents(df: pd.DataFrame):

    docs = []

    total_rows = len(df)

    print(f"Creating enriched embeddings ({total_rows:,} rows)...")

    for idx, row in df.iterrows():

        content = f"""
TITLE
{safe_value(row, 'title')}

TYPE
{safe_value(row, 'type')} | {safe_value(row, 'imdb_type')}

GENRE
{safe_value(row, 'genre')} | {safe_value(row, 'imdb_genre')}

CREATIVE TEAM
Director: {safe_value(row, 'director')}
Writer: {safe_value(row, 'writer')}
Actors: {safe_value(row, 'actors')}

CONTENT SUMMARY
Plot: {safe_value(row, 'plot')}

AUDIENCE & TONE
Rated: {safe_value(row, 'rated')}
Language: {safe_value(row, 'language')}
Country: {safe_value(row, 'country')}

QUALITY SIGNALS
IMDb Rating: {safe_value(row, 'imdb_rating')}
IMDb Votes: {safe_value(row, 'imdb_votes')}
Rating Z-Score: {safe_value(row, 'imdb_rating_zscore')}

POPULARITY SIGNALS
Total Watchtime: {safe_value(row, 'total_watchtime')}
Recency Weighted Watchtime: {safe_value(row, 'recency_weighted_watchtime')}
Watchtime Z-Score: {safe_value(row, 'watchtime_zscore')}

LONGEVITY SIGNALS
Evergreen Score: {safe_value(row, 'evergreen_score')}
Years Since Release: {safe_value(row, 'years_since_release')}

STRUCTURE
Runtime Minutes: {safe_value(row, 'runtime_minutes')}
Seasons: {safe_value(row, 'total_seasons')}

CINEMATIC PROFILE
This title is a {safe_value(row, "genre")} story with {safe_value(row, "imdb_rating")} IMDb rating,
known for {safe_value(row, "evergreen_score")} rewatchability and strong audience retention.
"""

        metadata = {
            "title": safe_value(row, "title"),
            "type": safe_value(row, "type"),
            "genre": safe_value(row, "genre"),
            "director": safe_value(row, "director"),
            "actors": safe_value(row, "actors"),
            "country": safe_value(row, "country"),
            "language": safe_value(row, "language"),
            "imdb_rating": safe_value(row, "imdb_rating"),
            "imdb_votes": safe_value(row, "imdb_votes"),
            "evergreen_score": safe_value(row, "evergreen_score"),
            "watchtime_zscore": safe_value(row, "watchtime_zscore"),
            "row_id": str(idx)
        }

        docs.append(
            Document(
                page_content=content.strip(),
                metadata=metadata
            )
        )

    return docs

# def build_documents(df: pd.DataFrame):

#     docs = []

#     total_rows = len(df)

#     print(
#         f"Creating LangChain documents "
#         f"({total_rows:,} rows)..."
#     )

#     for idx, row in df.iterrows():

#         content = f"""
# Title: {safe_value(row, 'title')}

# Genre: {safe_value(row, 'genre')}

# Director: {safe_value(row, 'director')}

# Actors: {safe_value(row, 'actors')}

# Country: {safe_value(row, 'country')}

# Language: {safe_value(row, 'language')}

# IMDb Rating: {safe_value(row, 'imdb_rating')}

# Plot:
# {safe_value(row, 'plot')}
# """

#         metadata = {
#             "title": safe_value(row, "title"),
#             "genre": safe_value(row, "genre"),
#             "director": safe_value(row, "director"),
#             "actors": safe_value(row, "actors"),
#             "country": safe_value(row, "country"),
#             "language": safe_value(row, "language"),
#             "imdb_rating": safe_value(row, "imdb_rating"),
#             "row_id": str(idx)
#         }

#         docs.append(
#             Document(
#                 page_content=content,
#                 metadata=metadata
#             )
#         )

#     return docs


# ============================================================
# METADATA
# ============================================================


def load_existing_metadata():

    if not os.path.exists(METADATA_FILE):
        return None

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_metadata(dataset_hash):

    os.makedirs(
        VECTOR_DB_DIR,
        exist_ok=True
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "dataset_hash": dataset_hash,
                "embedding_model": EMBEDDING_MODEL,
                "collection_name": COLLECTION_NAME
                
            },
            f,
            indent=2
        )


# ============================================================
# REBUILD CHECK
# ============================================================


def should_rebuild(current_hash):

    if not os.path.exists(VECTOR_DB_DIR):
        print(
            "Vector store not found."
        )
        return True

    metadata = load_existing_metadata()

    if metadata is None:
        print(
            "Metadata file missing."
        )
        return True

    stored_hash = metadata.get(
        "dataset_hash"
    )

    if stored_hash != current_hash:

        print(
            "Dataset changed."
        )

        return True

    print(
        "Dataset unchanged."
    )

    return False


# ============================================================
# DELETE OLD STORE
# ============================================================


def remove_existing_store():

    if os.path.exists(VECTOR_DB_DIR):

        print(
            "Removing existing vector store..."
        )

        shutil.rmtree(
            VECTOR_DB_DIR,
            ignore_errors=True
        )


# ============================================================
# BUILD VECTOR STORE
# ============================================================


def build_vector_store(documents):

    print(
        f"Creating embeddings "
        f"using {EMBEDDING_MODEL}"
    )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    print(
        "Building Chroma database..."
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME
    )

    return vectorstore


# ============================================================
# VALIDATION
# ============================================================


def validate_vector_store():

    print(
        "Validating vector store..."
    )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    count = db._collection.count()

    print(
        f"Stored documents: {count:,}"
    )

    return count


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 60)
    print("NETFLIX VECTOR STORE BUILDER")
    print("=" * 60)

    df = load_dataset()

    current_hash = generate_dataset_hash(df)

    if not should_rebuild(current_hash):

        print(
            "Using existing vector store."
        )

        return

    remove_existing_store()

    documents = build_documents(df)

    build_vector_store(documents)

    save_metadata(current_hash)

    count = validate_vector_store()

    print()
    print("=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(
        f"Documents embedded: {count:,}"
    )
    print(
        f"Vector store: {VECTOR_DB_DIR}"
    )


if __name__ == "__main__":
    main()