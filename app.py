import os
import sys
import re
import tempfile
import hashlib
import streamlit as st

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(__file__))

from config import GEMINI_API_KEY
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

# Sidebar - Document Processing
with st.sidebar:
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

    # Index Auto-load or Process
    if target_pdf_path:
        if st.session_state.get("pdf_filename") != target_filename or not st.session_state.get("is_indexed"):
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', target_filename)
            cache_dir = f"faiss_store_{safe_name}"
            
            store = FAISSVectorStore()
            
            # Check if index already exists in cache or default store
            if target_filename == "annual_report.pdf" and store.load("faiss_store"):
                st.session_state["vector_store"] = store
                set_vector_store(store)
                st.session_state["pdf_filename"] = target_filename
                st.session_state["is_indexed"] = True
                st.success("Loaded pre-indexed FAISS Vector Store!")
            elif store.load(cache_dir) and len(store.chunks_metadata) >= 20:
                st.session_state["vector_store"] = store
                set_vector_store(store)
                st.session_state["pdf_filename"] = target_filename
                st.session_state["is_indexed"] = True
                st.success(f"Loaded cached index for {target_filename} ({len(store.chunks_metadata)} chunks)!")
            else:
                st.markdown("#### ⏳ Financial Document Indexing")
                status_placeholder = st.empty()
                progress_bar = st.progress(0.0)

                status_placeholder.write("1/3 Extracting document pages...")
                all_pages = extract_pages_from_pdf(target_pdf_path)
                
                status_placeholder.write(f"2/3 Scanning {len(all_pages)} pages for financial statements...")
                
                # Priority Financial Table Indicators
                table_keywords = [
                    "revenue from operations", "statement of profit and loss", "profit for the year", 
                    "results of our operations", "consolidated statement of profit", "balance sheet", 
                    "standalone statement of profit", "total income", "income statement", "board's report"
                ]
                
                high_priority_pages = []
                general_financial_pages = []
                
                for p in all_pages:
                    txt_lower = p["text"].lower()
                    if any(kw in txt_lower for kw in table_keywords):
                        high_priority_pages.append(p)
                    elif p["page"] <= 15 or "revenue" in txt_lower or "profit" in txt_lower:
                        general_financial_pages.append(p)

                # Prioritize financial statement pages across document
                selected_pages = high_priority_pages + general_financial_pages
                if not selected_pages:
                    selected_pages = all_pages[:60]

                all_chunks = chunk_pages(selected_pages, chunk_size=1000, chunk_overlap=150)
                
                # Smart chunk sorting: prioritize chunks containing financial statement numbers
                def chunk_priority_score(c):
                    t = c["text"].lower()
                    score = 0
                    if any(k in t for k in ["revenue from operations", "results of our operations", "profit for the year", "statement of profit"]):
                        score += 10
                    if any(k in t for k in ["standalone", "consolidated", "balance sheet"]):
                        score += 5
                    if re.search(r'\d{1,3}(,\d{3})+', c["text"]):  # formatted financial numbers e.g. 1,62,990
                        score += 5
                    return score

                sorted_chunks = sorted(all_chunks, key=chunk_priority_score, reverse=True)
                
                # Take top 140 highest-signal financial chunks
                chunks = sorted_chunks[:140]

                status_placeholder.write(f"3/3 Generating Gemini vector embeddings ({len(chunks)} chunks)...")

                def update_progress(current, total):
                    pct = min(current / total, 1.0)
                    progress_bar.progress(pct)
                    status_placeholder.write(f"Embedding chunk {current}/{total}...")

                store.build_from_chunks(chunks, batch_size=30, progress_callback=update_progress)
                store.save(cache_dir)

                st.session_state["vector_store"] = store
                set_vector_store(store)
                st.session_state["pdf_filename"] = target_filename
                st.session_state["is_indexed"] = True
                status_placeholder.empty()
                progress_bar.empty()
                st.success(f"Successfully indexed {len(chunks)} high-priority financial chunks from {target_filename}!")
    
    # Status Card in Sidebar
    if st.session_state.get("is_indexed"):
        st.markdown("---")
        st.markdown("### 🟢 Vector Index Active")
        st.write(f"**Document:** `{st.session_state['pdf_filename']}`")
        st.write(f"**Indexed Chunks:** `{len(st.session_state['vector_store'].chunks_metadata)}`")
    else:
        st.info("Upload a PDF or select the sample report to begin.")

# Main Application Layout
col_main, col_info = st.columns([7, 3])

with col_main:
    st.subheader("💡 Financial Question")
    
    example_prompts = [
        "What was the revenue in FY2025?",
        "Compare revenue and net profit growth between the two latest fiscal years. Which grew faster?",
        "Compare Standalone vs Consolidated performance."
    ]
    
    selected_example = st.selectbox("Or choose a sample question:", ["Custom Query..."] + example_prompts)
    
    default_text = "" if selected_example == "Custom Query..." else selected_example
    user_query = st.text_area("Enter your question:", value=default_text, height=90, placeholder="e.g. Compare revenue and profit growth rate...")

    analyze_btn = st.button("🔍 Analyze & Verify Evidence", type="primary", use_container_width=True)

    if analyze_btn:
        if not st.session_state.get("is_indexed"):
            st.error("Please upload or load an annual report PDF first!")
        elif not user_query.strip():
            st.warning("Please enter a question to analyze.")
        else:
            set_vector_store(st.session_state["vector_store"])
            
            st.markdown("### 🤖 Agent Execution & Live Activity")
            activity_container = st.container()
            
            with st.spinner("FinSight Agent analyzing report evidence..."):
                agent = FinSightAgent()
                result = agent.run(user_query)
            
            # Render Activity Feed
            with activity_container:
                for log_item in result["status_log"]:
                    clean_log = log_item.encode("ascii", errors="replace").decode("ascii")
                    st.markdown(f'<div class="activity-card">{clean_log}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📋 Final Verification Report")
            
            if result.get("abstained"):
                st.markdown('<div class="abstain-badge">⚠ ABSTAINED — Evidence Insufficient</div>', unsafe_allow_html=True)
            elif result.get("is_supported"):
                st.markdown('<div class="verified-badge">✓ EVIDENCE VERIFIED ACCURATE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="abstain-badge">⚠ PARTIAL / UNVERIFIED EVIDENCE</div>', unsafe_allow_html=True)

            st.markdown(result["final_answer"])
            
            # Show Source Pages Tags
            if result.get("source_pages"):
                st.markdown("#### 📚 Page Citations")
                pages_html = " ".join([f'<span class="source-tag">Page {p}</span>' for p in result["source_pages"]])
                st.markdown(pages_html, unsafe_allow_html=True)

with col_info:
    st.subheader("🛡️ Agent Architecture")
    st.markdown("""
    **FinSight Agent Loop:**
    1. **Question Analysis:** Formulates targeted search queries.
    2. **Vector Retrieval:** Queries FAISS index (`gemini-embedding-2`).
    3. **Arithmetic Tool:** Calculates percentage growth/ratios.
    4. **Verification Engine:** Checks values, fiscal years & units.
    5. **Abstention Policy:** Abstains from hallucinating if evidence is missing.
    """)
    st.markdown("---")
    st.markdown("### ⚙️ System Config")
    st.write(f"**LLM Model:** `gemini-3.6-flash`")
    st.write(f"**Embeddings:** `gemini-embedding-2`")
    st.write(f"**Framework:** `LangGraph` + `PyMuPDF` + `FAISS`")
