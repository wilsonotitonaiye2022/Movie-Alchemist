# utils/poster_generator.py

import requests
import streamlit as st

from openai import OpenAI

from utils.config import (
    HF_IMAGE_MODEL,
    POSTER_IMAGE_SIZE
)

# ============================================================
# HUGGING FACE
# ============================================================


def generate_hf_poster(prompt):

    token = st.secrets.get(
        "HF_TOKEN"
    )

    if not token:
        return None

    url = (
        "https://api-inference.huggingface.co/models/"
        f"{HF_IMAGE_MODEL}"
    )

    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json={
                "inputs": prompt
            },
            timeout=120
        )

        if response.status_code == 200:
            return response.content

        return None

    except Exception:

        return None


# ============================================================
# OPENAI FALLBACK
# ============================================================


def generate_openai_poster(prompt):

    key = st.secrets.get(
        "OPENAI_API_KEY"
    )

    if not key:
        return None

    try:

        client = OpenAI(
            api_key=key
        )

        image = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=POSTER_IMAGE_SIZE
        )

        image_url = image.data[0].url

        image_response = requests.get(
            image_url,
            timeout=60
        )

        return image_response.content

    except Exception:

        return None


# ============================================================
# PUBLIC FUNCTION
# ============================================================


def generate_movie_poster(prompt):

    poster = generate_hf_poster(
        prompt
    )

    if poster:
        return poster

    return generate_openai_poster(
        prompt
    )