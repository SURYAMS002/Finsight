import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Fallback to Streamlit Cloud Secrets if os.getenv is empty
if not GEMINI_API_KEY:
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_KEY_HERE":
    raise ValueError(
        "GEMINI_API_KEY is missing! "
        "Please configure GEMINI_API_KEY in .env for local run or in Streamlit Cloud Secrets for deployment."
    )

# Model configuration
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
