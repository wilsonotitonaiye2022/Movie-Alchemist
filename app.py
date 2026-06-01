import random
from typing import TypedDict, List, Dict, Any

import streamlit as st
import pandas as pd
import streamlit as st
from langgraph.graph import StateGraph

# ============================================================
# LOCAL MODULES
# ============================================================

#from utils.vector_store import semantic_search
from utils.llm_provider import get_llm, safe_invoke
from utils.pdf_utils import create_watch_party_pdf, create_pitch_pdf
from utils.poster_generator import generate_movie_poster
from utils.data_quality import compute_column_coverage, compute_confidence

from utils.recommender import score_title

from utils.vector_store import (
    load_embedding_model,
    load_embeddings,
    load_metadata,
    semantic_search,
    cached_semantic_search
)
# from utils.vector_store import (
#     load_embedding_model,
#     load_embeddings,
#     load_metadata,
#     semantic_search,
#     hybrid_search,
#     similar_movies,
#     rank_results
# )

@st.cache_resource
def initialize_search():

    model = load_embedding_model()

    embeddings = load_embeddings()

    metadata_df = load_metadata()

    return (
        model,
        embeddings,
        metadata_df
    )

model, embeddings, metadata_df = initialize_search()

# @st.cache_data(show_spinner=False)
# def cached_semantic_search(
#     query: str,
#     k: int = 30
# ):

#     return semantic_search(
#         query=query,
#         embeddings=embeddings,
#         metadata_df=metadata_df,
#         model=model,
#         k=k
#     )

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="🎬 Netflix AI Engine",
    layout="wide"
)

llm = get_llm()

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_excel("data/netflix_sample.xlsx", sheet_name = 'Additional IMDb Data')
    df.columns = df.columns.str.strip().str.lower()
    return df


df = load_data()

@st.cache_resource
def build_title_index(df):

    return df.set_index(
        "title",
        drop=False
    )

title_index = build_title_index(df)

# ============================================================
# SESSION STATE
# ============================================================

if "memory" not in st.session_state:
    st.session_state.memory = {
        "vibes": [],
        "pitches": [],
        "watch_plans": []
    }

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "genres": [],
        "actors": [],
        "vibes": []
    }


def update_profile(vibe=None, genre=None):

    if vibe:
        st.session_state.user_profile["vibes"].append(vibe)

    if genre:
        st.session_state.user_profile["genres"].append(genre)


# ============================================================
# STATE
# ============================================================
state = {
    "vibe": str,
    "results": list,
    "analysis": str,
    "status": "success | error"
}


class AppState(TypedDict, total=False):
    vibe: str
    titles: List[Dict[str, Any]]
    a: str
    b: str
    pitch: Dict[str, Any]
    watch_plan: Dict[str, Any]
    analysis: str 

# ============================================================
# LLM SAFE PARSER
# ============================================================

# def safe_json_parse(text: str):

#     import json

#     try:
#         return json.loads(text)

#     except Exception:

#         return {"raw": text}


# ============================================================
# RANKED RECOMMENDER
# ============================================================

def ranked_recommendations(query: str, user_profile: dict):
    """
    Netflix-style ranking layer.
    """

    results = cached_semantic_search(query, k=30)

    ranked = []

    for movie in results:

        score = score_title(
            row=movie,
            user_profile=user_profile,
            query_vector_score=movie.get("similarity", 0.0)
        )

        movie["score"] = float(score)

        ranked.append(movie)

    ranked.sort(key=lambda x: x["score"], reverse=True)

    return ranked[:10]

# ============================================================
# VIBE NODE
# ============================================================


def vibe_node(state):

    results = ranked_recommendations(
        state["vibe"],
        st.session_state.user_profile
    )

    title_names = [m.get("title", "") for m in results]

    prompt = f"""
You are a Netflix programming strategist.

Explain why these titles match the vibe:

Vibe: {state['vibe']}

Titles:
{title_names}
"""

    analysis = safe_invoke(llm, prompt)

    return {
        **state,
        "titles": results,
        "analysis": str(analysis) if analysis else ""
    }


# ============================================================
# WATCH NODE
# ============================================================
def watch_node(state: AppState):

    titles = [
        t.get("title")
        for t in state.get("titles", [])
        if t.get("title")
    ]

    if not titles:

        return {
            **state,
            "watch_plan": {
                "theme": state["vibe"],
                "selections": [],
                "narrative": "No matching titles found."
            }
        }

    if len(titles) < 3:
        picks = titles
    else:
        picks = random.sample(
            titles,
            3
        )

    prompt = f"""
Design a cinematic watch party flow with emotional escalation.

Titles:
{picks}
"""

    plan_text = safe_invoke(
        llm,
        prompt
    )

    plan = {
        "theme": state["vibe"],
        "selections": picks,
        "narrative": plan_text
    }

    return {
        **state,
        "watch_plan": plan
    }


# ============================================================
# PITCH NODE
# ============================================================
def pitch_node(state):

    a, b = state["a"], state["b"]

    concept = safe_invoke(
        llm,
        f"""
            Create a streaming series concept from:

            {a} + {b}
            """
    )

    return {
        **state,
        "pitch": {
            "title": f"{a} x {b}",
            "logline": "Hybrid cinematic streaming concept",
            "concept": concept,
            "data_report": "Auto-generated concept"
        }
    }

# ============================================================
# MARKET NODE
# ============================================================

def market_node(state):

    prompt = f"""
Estimate success probability for:

{state['pitch']['concept']}
"""

    analysis = safe_invoke(llm, prompt)

    return {
        **state,
        "market": analysis
    }


# ============================================================
# GRAPH BUILDERS
# ============================================================

def build_vibe_graph():

    g = StateGraph(AppState)

    g.add_node("vibe", vibe_node)

    g.set_entry_point("vibe")

    return g.compile()


def build_watch_graph():

    g = StateGraph(AppState)

    g.add_node("vibe", vibe_node)

    g.add_node("watch", watch_node)

    g.add_edge("vibe", "watch")

    g.set_entry_point("vibe")

    return g.compile()


# ============================================================
# 🎬 STREAMLIT UI
# ============================================================

st.markdown(
    """
    <div style="padding:10px 0;">
        <h1 style="color:#E50914;">🎥 The Cinematic Alchemist Pro</h1>
        <h4 style="color:gray;">Discover. Remix. Program. Watch smarter.</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Data Health")

coverage = compute_column_coverage(df)

for _, row in coverage.iterrows():
    st.sidebar.write(f"{row['column']}: {int(row['coverage']*100)}%")


# ============================================================
# UI TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎭 Vibe Search",
    "🚀 Greenlight Studio",
    "🍿 Watch Planner",
    "💬 Dataset Chat"
])

# ============================================================
# TAB 1 — VIBE SEARCH
# ============================================================

with tab1:

    vibe = st.text_input("Describe a vibe")

    if st.button("Analyze") and vibe:

        update_profile(vibe=vibe)

        graph = build_vibe_graph()

        result = graph.invoke({"vibe": vibe})

        st.subheader("Analysis")
        #st.write(result["analysis"])
        st.write(result.get("analysis", "Generating insights..."))

        st.subheader("Top Picks")

        for t in result["titles"]:

            st.write(f"- {t['title']}")


# ============================================================
# TAB 2 — PITCH
# ============================================================

with tab2:

    st.subheader("🚀 Greenlight Studio")

    a = st.selectbox(
        "Show A",
        df["title"],
        key="show_a"
    )

    b = st.selectbox(
        "Show B",
        df["title"],
        key="show_b"
    )

    # --------------------------------------------------------
    # GET SELECTED MOVIES
    # --------------------------------------------------------

    movie_a = df[df["title"] == a].iloc[0]
    movie_b = df[df["title"] == b].iloc[0]

    # --------------------------------------------------------
    # DISPLAY MOVIE CARDS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🎬 Show A")

        poster_url = movie_a.get("poster", None)

        if pd.notna(poster_url) and str(poster_url).strip():
            st.image(
                poster_url,
                width=300,
                use_container_width=True
            )
        else:
            st.info("No poster available")

        st.markdown(f"#### {movie_a['title']}")

        if "imdb_rating" in movie_a.index:
            st.write(f"⭐ IMDb Rating: {movie_a['imdb_rating']}")

        if "genre" in movie_a.index:
            st.write(f"🎭 Genre: {movie_a['genre']}")

        if "release_year" in movie_a.index:
            st.write(f"📅 Year: {movie_a['release_year']}")

    with col2:

        st.markdown("### 🎬 Show B")

        poster_url = movie_b.get("poster", None)

        if pd.notna(poster_url) and str(poster_url).strip():
            st.image(
                poster_url,
                width=300,
                use_container_width=True
            )
        else:
            st.info("No poster available")

        st.markdown(f"#### {movie_b['title']}")

        if "imdb_rating" in movie_b.index:
            st.write(f"⭐ IMDb Rating: {movie_b['imdb_rating']}")

        if "genre" in movie_b.index:
            st.write(f"🎭 Genre: {movie_b['genre']}")

        if "release_year" in movie_b.index:
            st.write(f"📅 Year: {movie_b['release_year']}")

    st.markdown("---")

    # --------------------------------------------------------
    # GENERATE PITCH
    # --------------------------------------------------------

    if st.button("🚀 Generate Pitch"):

        state = {
            "a": a,
            "b": b
        }

        state = pitch_node(state)
        state = market_node(state)

        used = df[df["title"].isin([a, b])]

        confidence = compute_confidence(used)

        st.subheader("🎬 Generated Pitch")

        st.write(
            state["pitch"]["concept"]
        )

        st.subheader("📊 Market Analysis")

        st.write(
            state["market"]
        )

        st.metric(
            "Confidence",
            f"{confidence}%"
        )

        # ----------------------------------------------------
        # PDF EXPORT
        # ----------------------------------------------------

        pdf = create_pitch_pdf(
            pitch=state["pitch"],
            market_summary=state["market"],
            success_probability=80,
            confidence=confidence
        )

        st.download_button(
            label="📄 Download Pitch PDF",
            data=pdf,
            file_name="pitch.pdf",
            mime="application/pdf"
        )

# with tab2:

#     a = st.selectbox("Show A", df["title"])
#     b = st.selectbox("Show B", df["title"])

#     if st.button("Generate Pitch"):

#         state = {"a": a, "b": b}

#         state = pitch_node(state)
#         state = market_node(state)

#         used = df[df["title"].isin([a, b])]

#         confidence = compute_confidence(used)

#         st.subheader("🎬 Pitch")
#         st.write(state["pitch"]["concept"])

#         st.subheader("📊 Market")
#         st.write(state["market"])

#         st.metric("Confidence", f"{confidence}%")

#         pdf = create_pitch_pdf(
#             pitch=state["pitch"],
#             market_summary=state["market"],
#             success_probability=80,
#             confidence=confidence
#         )

#         st.download_button(
#             "📄 Download Pitch PDF",
#             pdf,
#             file_name="pitch.pdf",
#             mime="application/pdf"
#         )

# ============================================================
# TAB 3 — WATCH PLANNER
# ============================================================

with tab3:

    theme = st.text_input("Watch party theme")

    if st.button("Create Plan") and theme:

        graph = build_watch_graph()

        result = graph.invoke({"vibe": theme})

        plan = result["watch_plan"]

        st.subheader("Theme")
        st.write(plan["theme"])

        st.subheader("Selections")

        for t in plan["selections"]:
            st.write(f"- {t}")

        st.subheader("Narrative")
        st.write(plan["narrative"])

        pdf = create_watch_party_pdf(plan)

        st.download_button(
            "📄 Download Plan PDF",
            pdf,
            file_name="watch_plan.pdf",
            mime="application/pdf"
        )


# ============================================================
# TAB 4 — DATASET CHAT (RAG)
# ============================================================

with tab4:

    st.subheader("💬 Ask the AI Assistant")

    # =========================================================
    # SESSION CHAT MEMORY
    # =========================================================

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "selected_query" not in st.session_state:
        st.session_state.selected_query = ""

    # =========================================================
    # QUICK SUGGESTION BUTTONS
    # =========================================================

    col1, col2, col3 = st.columns(3)

    if col1.button("Best thriller movies"):
        st.session_state.selected_query = "Best thriller movies"

    if col2.button("Hidden gems on Netflix"):
        st.session_state.selected_query = "Hidden gems on Netflix"

    if col3.button("What should I watch tonight?"):
        st.session_state.selected_query = "What should I watch tonight?"

    
    q = st.text_input("Ask about movies",value=st.session_state.selected_query)

    ask = st.button("🔎 Ask AI Assistant")

    # =========================================================
    # RUN CHAT QUERY
    # =========================================================

    if ask and q and q.strip():

        st.session_state.selected_query = ""

        results = semantic_search(
            query=q,
            embeddings=embeddings,
            metadata_df=metadata_df,
            model=model,
            k=5
        )

        context = "\n\n".join(
            [
                f"""
Title: {r.get('title', '')}

Genre: {r.get('genre', '')}

IMDb Rating: {r.get('imdb_rating', '')}

Plot:
{r.get('plot', '')}
"""
                for r in results
            ]
        )

        prompt = f"""
You are a Netflix assistant.

Use the provided catalog context to answer clearly and concisely.

Context:
{context}

User Question:
{q}
"""

        response = safe_invoke(llm, prompt)

        # Store in memory
        st.session_state.chat_history.append(
            {"role": "user", "content": q}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )

    # =========================================================
    # DISPLAY CHAT HISTORY (NETFLIX STYLE)
    # =========================================================

    st.markdown("---")
    st.subheader("🧠 Conversation")

    for msg in st.session_state.chat_history:

        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style="
                    background:#E50914;
                    color:white;
                    padding:10px;
                    border-radius:10px;
                    margin:5px 0;
                    max-width:80%;
                ">
                <b>You:</b> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"""
                <div style="
                    background:#222;
                    color:white;
                    padding:10px;
                    border-radius:10px;
                    margin:5px 0;
                    max-width:80%;
                ">
                <b>AI Assistant:</b> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )

    # =========================================================
    # CLEAR CHAT
    # =========================================================

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

