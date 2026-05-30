# utils/llm_provider.py

import streamlit as st

import json

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential
)

from langchain_openai import ChatOpenAI

from utils.config import (
    GROQ_MODEL,
    OPENROUTER_MODEL,
    DEFAULT_TEMPERATURE
)


import json

def safe_json_parse(text: str):
    """
    Safely parse JSON from LLM output.
    Falls back to raw text if parsing fails.
    """
    try:
        return json.loads(text)
    except Exception:
        return {
            "raw": text
        }

    
# ============================================================
# PROVIDERS
# ============================================================


def build_groq_llm():

    key = st.secrets.get(
        "GROQ_API_KEY"
    )

    if not key:
        return None

    return ChatOpenAI(
        model=GROQ_MODEL,
        openai_api_key=key,
        openai_api_base=(
            "https://api.groq.com/openai/v1"
        ),
        temperature=DEFAULT_TEMPERATURE
    )


def build_openrouter_llm():

    key = st.secrets.get(
        "OPENROUTER_API_KEY"
    )

    if not key:
        return None

    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=key,
        openai_api_base=(
            "https://openrouter.ai/api/v1"
        ),
        temperature=DEFAULT_TEMPERATURE
    )


# ============================================================
# LLM FACTORY
# ============================================================


@st.cache_resource(show_spinner=False)
def get_llm():

    groq = build_groq_llm()

    if groq:
        return groq

    openrouter = build_openrouter_llm()

    if openrouter:
        return openrouter

    raise ValueError(
        """
No LLM provider configured.

Provide either:

GROQ_API_KEY

or

OPENROUTER_API_KEY
"""
    )


# ============================================================
# SAFE INVOKE
# ============================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1,
        min=2,
        max=10
    )
)
def safe_invoke(llm, prompt):

    response = llm.invoke(prompt)

    return response.content