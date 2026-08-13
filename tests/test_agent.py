import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from build_index import build_or_load_index
from agent import FinSightAgent

def test_agent_workflow():
    print("=== FinSight Agent Workflow Integration Test ===")
    
    # 1. Load existing FAISS index
    store = build_or_load_index(index_dir="faiss_store", force_rebuild=False)
    print(f"[OK] FAISS Vector Store active with {len(store.chunks_metadata)} chunks.")
    
    agent = FinSightAgent()
    
    # Test 1: Financial comparison between FY2025 and FY2026
    q1 = "Compare revenue and net profit between FY2025 and FY2026 from consolidated results. Which grew faster?"
    print(f"\n--- TEST 1: Financial Comparison & Calculation (FY2025 vs FY2026) ---\nQuery: {q1}")
    
    result1 = agent.run(q1)
    
    print("\n[OK] Agent Status Log:")
    for log_item in result1["status_log"]:
        clean_log = log_item.encode("ascii", errors="replace").decode("ascii")
        print(f"  * {clean_log}")
        
    clean_final_1 = result1['final_answer'].encode("ascii", errors="replace").decode("ascii")
    print(f"\n[OK] Agent Final Answer:\n{clean_final_1}")
    
    assert "Sources:" in result1["final_answer"] or "Verification Status:" in result1["final_answer"]
    
    # Test 2: Non-existent evidence (Abstention Policy test)
    q2 = "What was the CEO's favorite personal vacation destination in 2025?"
    print(f"\n--- TEST 2: Non-Existent Information (Abstention Policy) ---\nQuery: {q2}")
    
    result2 = agent.run(q2)
    clean_final_2 = result2['final_answer'].encode("ascii", errors="replace").decode("ascii")
    print(f"\n[OK] Agent Final Answer:\n{clean_final_2}")
    
    assert "couldn't find sufficient evidence" in result2["final_answer"].lower() or result2["abstained"]
    
    print("\n[SUCCESS] All FinSight Agent workflow integration tests passed!")

if __name__ == "__main__":
    test_agent_workflow()
