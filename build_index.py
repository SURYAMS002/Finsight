import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag.pdf_processor import extract_pages_from_pdf
from rag.chunker import chunk_pages
from rag.vector_store import FAISSVectorStore
from tools.search_tool import set_vector_store

def build_or_load_index(
    pdf_path: str = os.path.join("data", "annual_report.pdf"), 
    index_dir: str = "faiss_store",
    force_rebuild: bool = False
) -> FAISSVectorStore:
    """
    Builds or loads the FAISS vector index for an annual report PDF.
    """
    store = FAISSVectorStore()
    
    if not force_rebuild and store.load(index_dir):
        print(f"[OK] Loaded existing FAISS index from '{index_dir}' ({len(store.chunks_metadata)} chunks).")
        set_vector_store(store)
        return store

    print(f"Extracting pages from '{pdf_path}'...")
    all_pages = extract_pages_from_pdf(pdf_path)
    print(f"[OK] Extracted {len(all_pages)} total pages.")

    # Select financial pages & strategic overview pages for comprehensive retrieval
    keywords = [
        "revenue", "profit", "consolidated", "standalone", "balance sheet", 
        "financial statements", "operating cash flow", "total assets", 
        "earnings per share", "growth", "margin", "income"
    ]
    
    selected_pages = []
    for p in all_pages:
        txt = p["text"].lower()
        # Include first 15 pages + any page matching financial keywords
        if p["page"] <= 15 or any(kw in txt for kw in keywords):
            selected_pages.append(p)

    print(f"[OK] Selected {len(selected_pages)} relevant financial pages for indexing.")

    print("Chunking selected pages...")
    chunks = chunk_pages(selected_pages, chunk_size=1000, chunk_overlap=150)
    print(f"[OK] Generated {len(chunks)} relevant chunks.")

    print("Generating vector embeddings and building FAISS index...")
    # Batch size 20 with rate limit retries
    store.build_from_chunks(chunks, batch_size=20)
    store.save(index_dir)
    print(f"[OK] FAISS index saved to '{index_dir}'.")

    set_vector_store(store)
    return store

if __name__ == "__main__":
    build_or_load_index(force_rebuild=True)
