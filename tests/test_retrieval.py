import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.pdf_processor import extract_pages_from_pdf
from rag.chunker import chunk_pages
from rag.embeddings import GeminiEmbedder
from rag.vector_store import FAISSVectorStore

def test_retrieval_pipeline():
    pdf_path = os.path.join("data", "annual_report.pdf")
    print(f"Testing RAG retrieval pipeline on: {pdf_path}")
    
    # 1. Process PDF
    pages = extract_pages_from_pdf(pdf_path)
    print(f"[OK] Extracted {len(pages)} pages")
    
    # 2. Chunk pages
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=150)
    print(f"[OK] Generated {len(chunks)} chunks")
    
    # Use a subset of pages for fast testing, e.g. first 50 chunks or load existing index
    vector_store = FAISSVectorStore()
    
    save_dir = "faiss_test_store"
    if not vector_store.load(save_dir):
        print("Building vector index for sample chunks...")
        sample_chunks = chunks[:100]  # First 100 chunks for quick test
        vector_store.build_from_chunks(sample_chunks, batch_size=50)
        vector_store.save(save_dir)
        print(f"[OK] Built and saved FAISS index with {len(sample_chunks)} vectors")
    else:
        print(f"[OK] Loaded existing FAISS index with {len(vector_store.chunks_metadata)} vectors")
        
    test_queries = [
        "FY2025 revenue",
        "FY2024 net profit",
        "total assets",
        "operating cash flow"
    ]
    
    for q in test_queries:
        print(f"\nSearching query: '{q}'")
        results = vector_store.search(q, top_k=2)
        assert len(results) > 0, f"No results returned for query: {q}"
        
        for idx, res in enumerate(results):
            print(f"  Result #{idx+1}: Score={res['score']:.4f} | Page={res['page']} | Source={res['source']}")
            print(f"  Snippet: {res['text'][:120]!r}...")
            
            assert "text" in res
            assert "page" in res
            assert "source" in res
            assert "score" in res
            assert isinstance(res["page"], int)

    print("\n[SUCCESS] RAG retrieval pipeline test passed successfully!")

if __name__ == "__main__":
    test_retrieval_pipeline()
