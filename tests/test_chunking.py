import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.pdf_processor import extract_pages_from_pdf
from rag.chunker import chunk_pages

def test_chunking_pipeline():
    pdf_path = os.path.join("data", "annual_report.pdf")
    print(f"Testing PDF -> Pages -> Chunks pipeline on: {pdf_path}")
    
    pages = extract_pages_from_pdf(pdf_path)
    print(f"[OK] Total extracted pages: {len(pages)}")
    
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=150)
    print(f"[OK] Total chunks generated: {len(chunks)}")
    
    assert len(chunks) > 0, "No chunks generated!"
    
    # Inspect first chunk
    c0 = chunks[0]
    print(f"[OK] Chunk 0 Metadata Check:")
    print(f"  - Chunk ID: {c0['chunk_id']}")
    print(f"  - Page Number: {c0['page']}")
    print(f"  - Source: {c0['source']}")
    print(f"  - Text Length: {len(c0['text'])} chars")
    
    # Verify metadata preservation
    for c in chunks[:50]:
        assert "chunk_id" in c
        assert "page" in c
        assert "source" in c
        assert "text" in c
        assert isinstance(c["page"], int)
        assert c["page"] >= 1
    
    print("\n[SUCCESS] Chunking pipeline test passed!")

if __name__ == "__main__":
    test_chunking_pipeline()
