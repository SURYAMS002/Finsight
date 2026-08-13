import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.calculator_tool import calculator, safe_eval_expr
from tools.verify_tool import verify_evidence
from tools.search_tool import search_report, set_vector_store
from rag.vector_store import FAISSVectorStore

def test_calculator_tool():
    print("--- Testing Calculator Tool ---")
    
    # 1. Growth calculation: (44960 - 40000) / 40000 * 100
    res1 = calculator("(44960 - 40000) / 40000 * 100")
    print(f"[OK] Growth test output: {res1.splitlines()[0]}")
    assert "12.4" in res1
    
    # 2. Currency/formatting cleanup test: 44,960 - 40,000
    res2 = calculator("44,960 - 40,000")
    print(f"[OK] Formatted currency test output: {res2.splitlines()[0]}")
    assert "4960" in res2

    # 3. Zero division protection
    res3 = calculator("100 / 0")
    print(f"[OK] Zero division output: {res3}")
    assert "Division by zero" in res3
    
    print("[SUCCESS] Calculator tool tests passed!\n")

def test_verify_tool():
    print("--- Testing Verify Tool ---")
    
    sample_evidence = """
    Annual Report FY2025:
    Consolidated Revenue from Operations for FY2025 was Rs 44960 crore, compared to Rs 40000 crore in FY2024.
    Net profit for FY2025 stood at Rs 12500 crore, up 8.7% from FY2024.
    """
    
    # Claim 1: Supported claim
    claim1 = "Revenue increased from Rs 40000 crore in FY2024 to Rs 44960 crore in FY2025."
    res1 = verify_evidence(claim1, sample_evidence)
    print(f"[OK] Supported claim verification status snippet: {res1.splitlines()[0]}")
    assert "SUPPORTED" in res1
    
    # Claim 2: False/Unsupported claim
    claim2 = "Revenue in FY2025 was Rs 90000 crore."
    res2 = verify_evidence(claim2, sample_evidence)
    print(f"[OK] Unsupported claim verification status snippet: {res2.splitlines()[0]}")
    assert "NOT_SUPPORTED" in res2 or "PARTIALLY_SUPPORTED" in res2
    
    print("[SUCCESS] Verify tool tests passed!\n")

def test_search_tool():
    print("--- Testing Search Tool ---")
    
    # Create vector store and mock metadata
    store = FAISSVectorStore()
    store.chunks_metadata = [
        {"text": "Consolidated Revenue for FY2025 reached Rs 44960 crore.", "page": 86, "source": "annual_report.pdf"},
        {"text": "Consolidated Net Profit for FY2025 reached Rs 12500 crore.", "page": 87, "source": "annual_report.pdf"}
    ]
    # Build fast dummy index
    import numpy as np
    import faiss
    vecs = np.random.randn(2, 3072).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    store.index = faiss.IndexFlatIP(3072)
    store.index.add(vecs)
    
    set_vector_store(store)
    
    output = search_report("FY2025 revenue", top_k=2)
    print(f"[OK] Search tool snippet: {output.splitlines()[0]}")
    assert "Page 86" in output or "Page 87" in output
    
    print("[SUCCESS] Search tool tests passed!\n")

if __name__ == "__main__":
    test_calculator_tool()
    test_verify_tool()
    test_search_tool()
