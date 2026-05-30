import random
from typing import TypedDict, List, Dict, Any

import streamlit as st
import pandas as pd

from langgraph.graph import StateGraph

# ============================================================
# LOCAL MODULES
# ============================================================

from utils.vector_store import semantic_search
from utils.llm_provider import get_llm, safe_invoke
from utils.pdf_utils import create_watch_party_pdf, create_pitch_pdf
from utils.poster_generator import generate_movie_poster
from utils.data_quality import compute_column_coverage, compute_confidence

from utils.recommender import score_title

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

class AppState(TypedDict, total=False):
    vibe: str
    titles: List[Dict[str, Any]]
    a: str
    b: str
    pitch: Dict[str, Any]
    watch_plan: Dict[str, Any]


# ============================================================
# LLM SAFE PARSER
# ============================================================

def safe_json_parse(text: str):

    import json

    try:
        return json.loads(text)

    except Exception:

        return {"raw": text}


# ============================================================
# RANKED RECOMMENDER
# ============================================================

def ranked_recommendations(query, df, user_profile):

    results = semantic_search(query, k=30)

    ranked = []

    for r in results:

        row = df[df["title"] == r["title"]]

        if row.empty:
            continue

        row = row.iloc[0]

        score = score_title(
            row=row,
            user_profile=user_profile,
            query_vector_score=1.0
        )

        ranked.append((r, score))

    ranked.sort(key=lambda x: x[1], reverse=True)

    return ranked[:10]


# ============================================================
# VIBE NODE
# ============================================================

def vibe_node(state):

    results = ranked_recommendations(
        state["vibe"],
        df,
        st.session_state.user_profile
    )

    titles = [r[0] for r in results]

    prompt = f"""
You are a Netflix programming strategist.

Explain why these titles match the vibe:
{state['vibe']}

Titles:
{titles}
"""

    analysis = safe_invoke(llm, prompt)

    return {
        **state,
        "titles": titles,
        "analysis": analysis
    }


# ============================================================
# WATCH NODE
# ============================================================

# def watch_node(state):

#     titles = [t["title"] for t in state["titles"]]

#     picks = titles[:3] if len(titles) < 3 else random.sample(titles, 3)

#     prompt = f"""
# Design a cinematic watch party.

# Return JSON:
# theme, selections, narrative

# Titles:
# {picks}
# """

#     raw = safe_invoke(llm, prompt)

#     parsed = safe_json_parse(raw)

#     plan = {
#         "theme": state["vibe"],
#         "selections": parsed.get("selections", picks),
#         "narrative": parsed.get("narrative", raw)
#     }

#     return {
#         **state,
#         "watch_plan": plan
#     }

def watch_node(state: AppState):

    titles = [
        t["title"]
        for t in state["titles"]
    ]

    if len(titles) < 3:
        picks = titles
    else:
        picks = random.sample(titles, 3)

    prompt = f"""
                Design a cinematic watch party flow with emotional escalation.

                Titles:
                {picks}
                """

    plan_text = safe_invoke(llm, prompt)

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

# def pitch_node(state):

#     a, b = state["a"], state["b"]

#     prompt = f"""
# Create a Netflix original concept.

# Return JSON:
# title, logline, concept, data_report

# Combine:
# {a} + {b}
# """

#     raw = safe_invoke(llm, prompt)

#     parsed = safe_json_parse(raw)

#     pitch = {
#         "title": parsed.get("title", f"{a} x {b}"),
#         "logline": parsed.get("logline", ""),
#         "concept": parsed.get("concept", raw),
#         "data_report": parsed.get("data_report", "")
#     }

#     return {
#         **state,
#         "pitch": pitch
#     }

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
        st.write(result["analysis"])

        st.subheader("Top Picks")

        for t in result["titles"]:

            st.write(f"- {t['title']}")


# ============================================================
# TAB 2 — PITCH
# ============================================================

with tab2:

    a = st.selectbox("Show A", df["title"])
    b = st.selectbox("Show B", df["title"])

    if st.button("Generate Pitch"):

        state = {"a": a, "b": b}

        state = pitch_node(state)
        state = market_node(state)

        used = df[df["title"].isin([a, b])]

        confidence = compute_confidence(used)

        st.subheader("🎬 Pitch")
        st.write(state["pitch"]["concept"])

        st.subheader("📊 Market")
        st.write(state["market"])

        st.metric("Confidence", f"{confidence}%")

        pdf = create_pitch_pdf(
            pitch=state["pitch"],
            market_summary=state["market"],
            success_probability=80,
            confidence=confidence
        )

        st.download_button(
            "📄 Download Pitch PDF",
            pdf,
            file_name="pitch.pdf",
            mime="application/pdf"
        )

        # poster = generate_movie_poster(f"{a} x {b} cinematic poster")

        # if poster:
        #     st.image(poster)


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

    q = st.text_input("Ask about movies")

    if q:

        results = semantic_search(q, k=5)

        context = "\n".join([r["document"] for r in results])

        prompt = f"""
You are a Netflix assistant.

Context:
{context}

Question:
{q}
"""

        response = safe_invoke(llm, prompt)

        st.write(response)



# import os
# import random
# from typing import TypedDict, List, Dict, Any

# import streamlit as st
# import pandas as pd

# from langgraph.graph import StateGraph

# from utils.models import WatchPlan
# from utils.llm_provider import safe_json_parse
# from utils.models import Pitch

# # ============================================================
# # LOCAL MODULES
# # ============================================================

# from utils.vector_store import (
#     semantic_search,
#     retrieval_confidence,
# )

# from utils.llm_provider import (
#     get_llm,
#     safe_invoke,
# )

# from utils.pdf_utils import (
#     create_watch_party_pdf,
#     create_pitch_pdf,
# )

# from utils.poster_generator import (
#     generate_movie_poster,
# )

# from utils.data_quality import (
#     compute_column_coverage,
#     compute_confidence,
#     generate_integrity_report,
# )

# from utils.recommender import score_title
# from utils.vector_store import semantic_search

# def ranked_recommendations(query, df, user_profile):

#     results = semantic_search(query, k=30)

#     ranked = []

#     for r in results:

#         row = df[df["title"] == r["title"]]

#         if row.empty:
#             continue

#         row = row.iloc[0]

#         score = score_title(
#             row=row,
#             user_profile=user_profile,
#             query_vector_score=1 / (r.get("score", 1) + 1)
#         )

#         ranked.append((r, score))

#     ranked.sort(key=lambda x: x[1], reverse=True)

#     return ranked[:10]
# # ============================================================
# # PAGE CONFIG
# # ============================================================

# st.set_page_config(
#     page_title="🎬 Cinematic Alchemist Pro",
#     layout="wide"
# )

# # ============================================================
# # DATA LOAD
# # ============================================================

# @st.cache_data
# def load_data():

#     df = pd.read_excel(
#         "data/netflix_sample.xlsx"
#     )

#     df.columns = (
#         df.columns
#         .str.strip()
#         .str.lower()
#     )

#     return df


# df = load_data()
# llm = get_llm()

# # ============================================================
# # SESSION MEMORY
# # ============================================================

# if "memory" not in st.session_state:

#     st.session_state.memory = {
#         "vibes": [],
#         "pitches": [],
#         "watch_plans": []
#     }

# if "user_profile" not in st.session_state:

#     st.session_state.user_profile = {
#         "genres": [],
#         "actors": [],
#         "vibes": []
#     }


# def update_memory(key, value):

#     st.session_state.memory[key].append(value)


# # ============================================================
# # STATE
# # ============================================================

# class AppState(TypedDict, total=False):
#     vibe: str
#     titles: List[Dict[str, Any]]
#     analysis: str
#     a: str
#     b: str
#     pitch: Dict[str, Any]
#     watch_plan: Dict[str, Any]
#     market: str

# # class AppState(TypedDict, total=False):
# #     vibe: str
# #     titles: List[Dict[str, Any]]
# #     analysis: str
# #     a: str
# #     b: str
# #     pitch: dict
# #     watch_plan: dict
# #     market: str

# # ============================================================
# # LLM PROMPTS
# # ============================================================

# SYSTEM_CONTEXT = """
# You are a senior streaming strategist and creative executive.
# Be precise, cinematic, and analytically grounded.
# Avoid fluff.
# """

# # ============================================================
# # VIBE NODE
# # ============================================================

# def vibe_node(state: AppState):

#     results = semantic_search(
#         state["vibe"],
#         k=5
#     )

#     prompt = f"""
# {SYSTEM_CONTEXT}

# Analyze why these titles match the vibe: {state['vibe']}

# Results:
# {results}
# """

#     analysis = safe_invoke(llm, prompt)

#     return {
#         **state,
#         "titles": results,
#         "analysis": analysis
#     }


# # ============================================================
# # WATCH PARTY NODE
# # ============================================================
# # def watch_node(state):

# #     titles = [t["title"] for t in state["titles"]]

# #     picks = titles[:3] if len(titles) < 3 else random.sample(titles, 3)

# #     raw = safe_invoke(
# #         llm,
# #         f"""
# # Design a watch party.

# # Return JSON:
# # theme, selections, narrative

# # Titles:
# # {picks}
# # """
# #     )

# #     parsed = safe_json_parse(raw)

# #     plan = WatchPlan(
# #         theme=parsed.get("theme", state["vibe"]),
# #         selections=parsed.get("selections", picks),
# #         narrative=parsed.get("narrative", raw)
# #     )

# #     return {
# #         **state,
# #         "watch_plan": plan.model_dump()
# #     }

# def watch_node(state: AppState):

#     titles = [
#         t["title"]
#         for t in state["titles"]
#     ]

#     if len(titles) < 3:
#         picks = titles
#     else:
#         picks = random.sample(titles, 3)

#     prompt = f"""
# Design a cinematic watch party flow with emotional escalation.

# Titles:
# {picks}
# """

#     plan_text = safe_invoke(llm, prompt)

#     plan = {
#         "theme": state["vibe"],
#         "selections": picks,
#         "narrative": plan_text
#     }

#     return {
#         **state,
#         "watch_plan": plan
#     }


# # ============================================================
# # PITCH NODE
# # ============================================================

# # def pitch_node(state):

# #     a, b = state["a"], state["b"]

# #     raw = safe_invoke(
# #         llm,
# #         f"""
# # Create a streaming series concept.

# # Return JSON with:
# # title, logline, concept, data_report

# # Movies:
# # {a} + {b}
# # """
# #     )

# #     parsed = safe_json_parse(raw)

# #     pitch = Pitch(
# #         title=parsed.get("title", f"{a} x {b}"),
# #         logline=parsed.get("logline", ""),
# #         concept=parsed.get("concept", raw),
# #         data_report=parsed.get("data_report", "auto-generated")
# #     )

# #     return {
# #         **state,
# #         "pitch": pitch.model_dump()
# #     }

# def pitch_node(state):

#     a, b = state["a"], state["b"]

#     concept = safe_invoke(
#         llm,
#         f"""
# Create a streaming series concept from:

# {a} + {b}
# """
#     )

#     return {
#         **state,
#         "pitch": {
#             "title": f"{a} x {b}",
#             "logline": "Hybrid cinematic streaming concept",
#             "concept": concept,
#             "data_report": "Auto-generated concept"
#         }
#     }

# # ============================================================
# # MARKET NODE
# # ============================================================

# def market_node(state: AppState):

#     prompt = f"""
# Estimate audience demand and streaming success probability
# for this concept:

# {state['pitch']['concept']}

# Return a short analytical summary.
# """

#     summary = safe_invoke(llm, prompt)

#     return {
#         **state,
#         "market": summary
#     }


# # ============================================================
# # LANGGRAPH BUILDERS
# # ============================================================

# def build_vibe_graph():

#     g = StateGraph(AppState)

#     g.add_node("vibe", vibe_node)

#     g.set_entry_point("vibe")

#     return g.compile()


# def build_watch_graph():

#     g = StateGraph(AppState)

#     g.add_node("vibe", vibe_node)

#     g.add_node("watch", watch_node)

#     g.add_edge("vibe", "watch")

#     g.set_entry_point("vibe")

#     return g.compile()

# def update_profile(vibe=None, genre=None):

#     if vibe:
#         st.session_state.user_profile["vibes"].append(vibe)

#     if genre:
#         st.session_state.user_profile["genres"].append(genre)




# # ============================================================
# # UI — SIDEBAR
# # ============================================================

# st.sidebar.title("📊 Data Health")

# coverage = compute_column_coverage(df)

# for _, row in coverage.iterrows():

#     st.sidebar.write(
#         f"{row['column']}: {int(row['coverage']*100)}%"
#     )


# # ============================================================
# # TABS
# # ============================================================

# tab1, tab2, tab3 = st.tabs(
#     [
#         "🎭 Vibe Search",
#         "🚀 Greenlight Studio",
#         "🍿 Watch Planner"
#     ]
# )

# # ============================================================
# # TAB 1 — VIBE SEARCH
# # ============================================================

# with tab1:

#     vibe = st.text_input("Describe a vibe")

#     if st.button("Analyze") and vibe:

#         update_memory("vibes", vibe)

#         graph = build_vibe_graph()

#         result = graph.invoke({"vibe": vibe})

#         st.subheader("Analysis")
#         st.write(result["analysis"])

#         st.subheader("Top Matches")

#         for t in result["titles"]:
#             st.write(f"- {t['title']}")

# # ============================================================
# # TAB 2 — GREENLIGHT STUDIO
# # ============================================================

# with tab2:

#     a = st.selectbox("Show A", df["title"])
#     b = st.selectbox("Show B", df["title"])

#     if st.button("Generate Pitch"):

#         state = {
#             "a": a,
#             "b": b
#         }

#         state = pitch_node(state)
#         state = market_node(state)

#         used = df[df["title"].isin([a, b])]

#         confidence = compute_confidence(used)

#         st.subheader("🎬 Pitch")
#         st.write(state["pitch"]["concept"])

#         st.subheader("📊 Market Analysis")
#         st.write(state["market"])

#         # st.metric(
#         #     "Confidence",
#         #     f"{confidence}%"
#         # )

#         # PDF
#         pdf = create_pitch_pdf(
#             pitch=type("obj", (), state["pitch"])(),
#             market_summary=state["market"],
#             success_probability=75,
#             confidence=confidence
#         )

#         st.download_button(
#             "📄 Download Pitch PDF",
#             pdf,
#             file_name="pitch.pdf",
#             mime="application/pdf"
#         )

#         # # Poster
#         # poster = generate_movie_poster(
#         #     f"{a} and {b} cinematic fusion poster"
#         # )

#         # if poster:
#         #     st.image(poster)

# # ============================================================
# # TAB 3 — WATCH PLANNER
# # ============================================================

# with tab3:

#     theme = st.text_input("Watch party theme")

#     if st.button("Create Plan") and theme:

#         update_memory("watch_plans", theme)

#         graph = build_watch_graph()

#         result = graph.invoke({"vibe": theme})

#         plan = result["watch_plan"]

#         st.subheader("Theme")
#         st.write(plan["theme"])

#         st.subheader("Selections")

#         for t in plan["selections"]:
#             st.write(f"- {t}")

#         st.subheader("Narrative")
#         st.write(plan["narrative"])

#         # PDF
#         pdf = create_watch_party_pdf(plan)

#         st.download_button(
#             "📄 Download Plan PDF",
#             pdf,
#             file_name="watch_plan.pdf",
#             mime="application/pdf"
#         )