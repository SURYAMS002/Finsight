import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_gemini_api_key() -> str:
    """
    Retrieves the Gemini API Key from:
    1. os.getenv("GEMINI_API_KEY")
    2. st.secrets["GEMINI_API_KEY"] (Streamlit Cloud)
    3. st.session_state["user_gemini_api_key"] (Sidebar user input)
    """
    key = os.getenv("GEMINI_API_KEY")
    if key and key != "YOUR_KEY_HERE":
        return key

    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        if st.session_state.get("user_gemini_api_key"):
            return st.session_state["user_gemini_api_key"]
    except Exception:
        pass

    return ""

GEMINI_API_KEY = get_gemini_api_key()

# Model configuration
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
