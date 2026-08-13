import os
import sys
import re
import tempfile
import hashlib
import streamlit as st

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(__file__))

from config import get_gemini_api_key
from rag.pdf_processor import extract_pages_from_pdf
from rag.chunker import chunk_pages
from rag.vector_store import FAISSVectorStore
from tools.search_tool import set_vector_store
from agent import FinSightAgent

# Streamlit Page Config
st.set_page_config(
    page_title="FinSight — Autonomous Financial Verification Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Glassmorphism FinTech Aesthetics)
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        max-width: 1250px;
        margin: 0 auto;
    }
    .header-box {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .header-title {
        color: #38BDF8;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    .activity-card {
        background-color: #1E293B;
        border-left: 4px solid #38BDF8;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-family: monospace;
        color: #E2E8F0;
    }
    .verified-badge {
        background-color: #064E3B;
        color: #34D399;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
        border: 1px solid #059669;
    }
    .abstain-badge {
        background-color: #451A03;
        color: #FBBF24;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
        border: 1px solid #D97706;
    }
    .source-tag {
        background-color: #1E293B;
        color: #38BDF8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        border: 1px solid #334155;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "pdf_filename" not in st.session_state:
    st.session_state["pdf_filename"] = None
if "is_indexed" not in st.session_state:
    st.session_state["is_indexed"] = False

# Header Section
st.markdown("""
<div class="header-box">
    <h1 class="header-title">📊 FinSight</h1>
    <p class="header-subtitle">Autonomous Annual Report Analysis & Verification Agent with Page-Level Citations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Document Processing & Optional Credentials
with st.sidebar:
    active_api_key = get_gemini_api_key()

    # Only display credentials box if no key is configured in Secrets or .env
    if not active_api_key:
        st.header("🔑 Credentials")
        api_key_input = st.text_input("Enter Gemini API Key", type="password", help="Enter your key to unlock the agent if Secrets is not configured")
        if api_key_input:
            st.session_state["user_gemini_api_key"] = api_key_input
            active_api_key = api_key_input
            st.success("✓ Gemini API Key set!")
        else:
            st.warning("⚠️ GEMINI_API_KEY required. Enter key above or add to Streamlit Cloud Secrets.")
        st.markdown("---")

    st.header("📄 Document Ingestion")
    
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type=["pdf"])
    
    default_pdf_path = os.path.join("data", "annual_report.pdf")
    has_default_pdf = os.path.exists(default_pdf_path)
    
    use_sample_pdf = False
    if has_default_pdf:
        st.markdown("---")
        st.markdown("**Pre-indexed Sample Document**")
        use_sample_pdf = st.button("Use Sample Annual Report (TCS 2025-26)")

    # Processing Logic
    target_pdf_path = None
    target_filename = None

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            target_pdf_path = tmp_file.name
            target_filename = uploaded_file.name
    elif use_sample_pdf:
        target_pdf_path = default_pdf_path
        target_filename = "annual_report.pdf"

    def chunk_priority_score(chunk: dict) -> int:
        """
        Calculates priority score for document chunks based on financial statement keywords.
        Guarantees statement tables across ANY annual report PDF (TCS, Infosys, HDFC, etc.) are indexed.
        """
        text_lower = chunk["text"].lower()
        score = 0
        statement_terms = [
            "statement of profit and loss", "balance sheet", "results of operations",
            "revenue from operations", "total income", "net profit", "cash flows",
            "consolidated financial statements", "standalone financial statements"
        ]
        for term in statement_terms:
            if term in text_lower:
                score += 15
        
        # Numbers formatting presence
        number_matches = len(re.findall(r"\d{1,3}(?:,\d{3})+", chunk["text"]))
        score += min(number_matches, 10)
        return score

    # Index Auto-load or Process
    if target_pdf_path:
        # Check if already processed
        if st.session_state["pdf_filename"] != target_filename or not st.session_state["is_indexed"]:
            if not active_api_key:
                st.error("Cannot index PDF: GEMINI_API_KEY is missing. Please enter your key in the Credentials box above or add to Streamlit Secrets.")
            else:
                with st.spinner(f"Extracting & Indexing '{target_filename}'..."):
                    progress_bar = st.progress(0.0, text="Extracting text from PDF pages...")
                    pages = extract_pages_from_pdf(target_pdf_path)
                    
                    progress_bar.progress(0.3, text=f"Extracted {len(pages)} pages. Chunking text...")
                    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)
                    
                    # Sort chunks by priority score so financial statements are prioritized
                    chunks_with_scores = [(chunk_priority_score(c), c) for c in chunks]
                    chunks_with_scores.sort(key=lambda x: x[0], reverse=True)
                    
                    # Embed top 140 highest-signal chunks
                    top_chunks = [c for _, c in chunks_with_scores[:140]]
                    
                    progress_bar.progress(0.5, text=f"Generating embeddings for top {len(top_chunks)} financial chunks...")
                    
                    store = FAISSVectorStore()
                    
                    def update_progress(current, total):
                        pct = 0.5 + (0.4 * (current / max(total, 1)))
                        progress_bar.progress(pct, text=f"Embedding chunks: {current}/{total}")

                    store.build_from_chunks(top_chunks, progress_callback=update_progress)
                    
                    st.session_state["vector_store"] = store
                    st.session_state["pdf_filename"] = target_filename
                    st.session_state["is_indexed"] = True
                    set_vector_store(store)
                    
                    progress_bar.progress(1.0, text="Indexing complete!")
                    st.success(f"Indexed '{target_filename}' ({len(top_chunks)} chunks).")

    # Status Display
    if st.session_state["is_indexed"]:
        st.markdown("---")
        st.success(f"🟢 **Active Report:** {st.session_state['pdf_filename']}")
    else:
        st.info("👈 Upload a PDF annual report or click 'Use Sample' to get started.")

# Main Q&A Interface
st.subheader("🤖 Financial Query Assistant")

user_query = st.text_input(
    "Ask a financial question about the uploaded report:",
    placeholder="e.g., What was the revenue from operations in FY2026 and what was the YoY growth?"
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_button = st.button("Analyze & Verify", type="primary")

if run_button and user_query:
    active_api_key = get_gemini_api_key()
    if not active_api_key:
        st.error("⚠️ GEMINI_API_KEY is missing! Please enter your Gemini API Key in the sidebar or set it in Streamlit Cloud Secrets.")
    elif not st.session_state["is_indexed"] or st.session_state["vector_store"] is None:
        st.error("Please upload an annual report PDF or click 'Use Sample' in the sidebar first.")
    else:
        # Create expandable activity log container
        with st.expander("🤖 Agent Activity & Execution Graph Log", expanded=True):
            activity_container = st.container()
            
            with st.spinner("FinSight Agent analyzing report evidence..."):
                agent = FinSightAgent()
                result = agent.run(user_query)

            # Render Activity Feed
            with activity_container:
                for step in result["status_log"]:
                    st.markdown(f'<div class="activity-card">⚙️ {step}</div>', unsafe_allow_html=True)

        st.markdown("### 📋 Final Verification Report")
        
        # Verification Status Badge
        if result.get("abstained"):
            st.markdown('<div class="abstain-badge">⚠ ABSTAINED — INSUFFICIENT EVIDENCE</div>', unsafe_allow_html=True)
        elif result.get("is_supported"):
            st.markdown('<div class="verified-badge">✓ FULLY VERIFIED EVIDENCE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="abstain-badge">⚠ PARTIAL / UNVERIFIED EVIDENCE</div>', unsafe_allow_html=True)

        # Main Answer Markdown
        st.markdown(result["final_answer"])
        
        # Source Pages Pills
        if result["source_pages"]:
            st.markdown("**Cited Source Pages:**")
            pages_html = " ".join([f'<span class="source-tag">Page {p}</span>' for p in result["source_pages"]])
            st.markdown(pages_html, unsafe_allow_html=True)
