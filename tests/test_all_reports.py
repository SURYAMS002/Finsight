import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.pdf_processor import extract_pages_from_pdf
from rag.chunker import chunk_pages
from rag.vector_store import FAISSVectorStore
from tools.search_tool import set_vector_store
from agent import FinSightAgent

def test_report(pdf_path: str, document_name: str, query: str):
    print(f"\n=======================================================")
    print(f" TESTING REPORT: {document_name}")
    print(f" PDF File: {pdf_path}")
    print(f" Query: {query}")
    print(f"=======================================================")
    
    pages = extract_pages_from_pdf(pdf_path)
    print(f"[OK] Extracted {len(pages)} total pages.")
    
    statement_keywords = [
        "statement of profit and loss", "profit and loss", "consolidated statement", 
        "standalone statement", "balance sheet", "revenue from operations", 
        "financial results", "income statement", "board's report", "segment revenue",
        "financial performance", "total income", "profit for the year", "key financial"
    ]
    
    high_priority_pages = []
    general_pages = []
    
    for p in pages:
        txt_lower = p["text"].lower()
        if any(kw in txt_lower for kw in statement_keywords):
            high_priority_pages.append(p)
        elif p["page"] <= 15 or "revenue" in txt_lower or "profit" in txt_lower:
            general_pages.append(p)

    selected_pages = high_priority_pages + general_pages[:20]
    chunks = chunk_pages(selected_pages, chunk_size=1000, chunk_overlap=150)
    
    # Priority sorting for table chunks
    import re
    def priority_score(c):
        t = c["text"].lower()
        score = 0
        if any(k in t for k in ["revenue from operations", "results of our operations", "profit for the year", "statement of profit"]):
            score += 10
        if any(k in t for k in ["standalone", "consolidated", "balance sheet"]):
            score += 5
        if re.search(r'\d{1,3}(,\d{3})+', c["text"]):
            score += 5
        return score

    sorted_chunks = sorted(chunks, key=priority_score, reverse=True)[:140]
    
    store = FAISSVectorStore()
    store.build_from_chunks(sorted_chunks, batch_size=30)
    set_vector_store(store)
    
    agent = FinSightAgent()
    result = agent.run(query)
    
    print("\n--- AGENT FINAL ANSWER ---")
    clean_final = result['final_answer'].encode("ascii", errors="replace").decode("ascii")
    print(clean_final)
    print(f"\n--- SOURCE PAGES: {result['source_pages']} ---")

if __name__ == "__main__":
    # Test 1: TCS Annual Report
    tcs_pdf = os.path.join("data", "annual_report.pdf")
    if os.path.exists(tcs_pdf):
        test_report(tcs_pdf, "TCS Annual Report 2025-26", "Compare revenue and net profit growth between FY2025 and FY2026.")
        
    # Test 2: Infosys Annual Report
    infy_pdf = os.path.join("data", "infosys_ar_25.pdf")
    if os.path.exists(infy_pdf):
        test_report(infy_pdf, "Infosys Annual Report 2024-25", "What was the revenue from operations in FY2025?")
