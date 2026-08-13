# FinSight — Autonomous Annual Report Analysis & Verification Agent

> **FinSight** is an AI agent that analyzes annual-report PDFs, autonomously searches for relevant financial evidence, performs calculations using tools, verifies its answers against the original source, and responds with page-level citations.

[![Live Streamlit App](https://img.shields.io/badge/Streamlit%20Cloud-Live%20Demo-38BDF8?style=for-the-badge&logo=streamlit)](https://finsight-rgedparqy6zxxjkca8yrgt.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/SURYAMS002/Finsight)

🌐 **Live 24/7 Web Application:** [https://finsight-rgedparqy6zxxjkca8yrgt.streamlit.app/](https://finsight-rgedparqy6zxxjkca8yrgt.streamlit.app/)

---

## 1. Problem Statement

Large Language Models (LLMs) often suffer from hallucinations, unit confusion, and arithmetic errors when asked complex financial questions about long annual reports (e.g. 300+ page PDFs). A standard "PDF -> RAG -> LLM -> Answer" pipeline cannot perform step-by-step verification, handle multi-page evidence, or calculate percentage growth accurately.

---

## 2. Solution Architecture

FinSight uses a **LangGraph-orchestrated agent** that dynamically decides which tools to call, performs financial arithmetic using a dedicated calculator tool, verifies every claim against retrieved evidence, and abstains from answering if evidence is insufficient.

```mermaid
graph TD
    User([User Question & PDF]) --> Agent[FinSight Agent]
    Agent --> Plan[Node 1: Plan & Search Report]
    Plan --> SearchTool[Tool: search_report]
    SearchTool --> FAISS[(FAISS Vector Store\ngemini-embedding-2)]
    FAISS --> CalcNode[Node 2: Financial Calculator]
    CalcNode --> CalcTool[Tool: calculator]
    CalcTool --> DraftNode[Node 3: Draft Answer]
    DraftNode --> VerifyNode[Node 4: Verification Engine]
    VerifyNode --> VerifyTool[Tool: verify_evidence\ngemini-3.6-flash]
    VerifyNode --> Decision{Is Evidence Verified?}
    Decision -- Supported / Partial --> FinalNode[Node 5: Finalize Answer & Citations]
    Decision -- Unsupported & Iterations < 2 --> Plan
    Decision -- Insufficient Evidence --> Abstain[Abstention Response]
    FinalNode --> UserOutput([Final Verified Output])
    Abstain --> UserOutput
```

---

## 3. Technology Stack

* **Language:** Python 3.11+ / 3.13
* **LLM Engine:** Google Gemini (`gemini-3.6-flash`) via `google-genai` SDK
* **Vector Embeddings:** Google Gemini (`gemini-embedding-2` - 3072 dims)
* **Agent Framework:** LangGraph (`StateGraph`)
* **Document Processing:** PyMuPDF (`pymupdf`)
* **Vector Indexing:** FAISS (`faiss-cpu`)
* **User Interface:** Streamlit
* **Deployment:** Streamlit Community Cloud (24/7 Live)

---

## 4. Agent Tools & Capabilities

1. **`search_report(query)`**: Performs hybrid semantic vector search (`faiss-cpu`) and financial statement table retrieval to extract text snippets with 1-indexed page citations.
2. **`calculator(expression)`**: Evaluates financial formulas (percentage change, difference, margin) safely using Python's AST parser to prevent mental arithmetic errors.
3. **`verify_evidence(claim, source_evidence)`**: Audits draft answers against retrieved PDF source text to verify numerical values, fiscal years (e.g. FY2025 vs FY2026), units (crore vs million), and context (Standalone vs Consolidated).

---

## 5. Hallucination & Abstention Policy

If the uploaded annual report does not contain sufficient evidence to answer a question, FinSight strictly abstains:

> *"I couldn't find sufficient evidence in the uploaded annual report to answer this reliably."*

This prevents inventing numbers or hallucinating non-existent facts.

---

## 6. Engineering Failures Discovered & System Fixes

### 1. Gemini API Free Tier Rate Limit (429 RESOURCE_EXHAUSTED)
* **Failure:** Rapid multi-step LLM calls and bulk embeddings triggered Gemini's 20 RPM Free Tier quota limit, causing raw error messages.
* **Engineering Fix:** Implemented `call_gemini_with_retry()` and `GeminiEmbedder` backoff engine that automatically extracts the API's requested `retryDelay` window (e.g. 15-20s), sleeps gracefully, and falls back across active Gemini models (`gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`).

### 2. Document-Agnostic Financial Statement Indexing
* **Failure:** Sequential chunk slicing chopped off late-document financial statement tables (e.g. Page 30 in Infosys or Page 167 in TCS).
* **Engineering Fix:** Implemented `chunk_priority_score()` to score and prioritize financial statement headers (*Statement of Profit and Loss, Balance Sheet, Results of Operations*) across any uploaded annual report PDF.

---

## 7. How to Run Locally

### 1. Clone Repository & Setup Environment

```powershell
git clone https://github.com/SURYAMS002/Finsight.git
cd Finsight
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

*(Note: `.env` is excluded from git version control via `.gitignore`.)*

### 4. Launch Application

```powershell
streamlit run app.py
```

---

## 8. Security & Compliance

* **Zero Hardcoded Secrets:** `GEMINI_API_KEY` is loaded dynamically from `.env`, Streamlit Cloud Secrets, or the sidebar user input box.
* **Git Hygiene:** `.env`, `.venv/`, binary `.pdf` files, `faiss_store/`, and Python bytecode are strictly excluded via `.gitignore`.
* **Safe AST Evaluation:** Calculator tool evaluates math expressions strictly using AST nodes without `eval()` security risks.
