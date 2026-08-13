import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_KEY_HERE":
    raise ValueError(
        "GEMINI_API_KEY is missing or invalid in .env! "
        "Please configure a valid GEMINI_API_KEY in the .env file."
    )

# Model configuration
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
