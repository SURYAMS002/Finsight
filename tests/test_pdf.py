import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.pdf_processor import extract_pages_from_pdf

def test_pdf_extraction():
    pdf_path = os.path.join("data", "annual_report.pdf")
    print(f"Testing PDF extraction on: {pdf_path}")
    
    pages = extract_pages_from_pdf(pdf_path)
    
    print(f"[OK] Total pages extracted: {len(pages)}")
    assert len(pages) > 0, "No pages extracted from PDF!"
    
    # Check structure of first page
    first_page = pages[0]
    print(f"[OK] Page 1 Metadata check:")
    print(f"  - Page Number: {first_page['page']}")
    print(f"  - Source: {first_page['source']}")
    print(f"  - Text preview (first 150 chars): {first_page['text'][:150]!r}")
    
    assert "page" in first_page
    assert "text" in first_page
    assert "source" in first_page
    assert first_page["page"] == 1
    
    # Find a non-empty page preview
    sample_page = next((p for p in pages if len(p['text']) > 100), pages[0])
    print(f"\n[OK] Sample Page {sample_page['page']} text length: {len(sample_page['text'])} chars")
    
    print("\n[SUCCESS] PDF extraction test passed!")

if __name__ == "__main__":
    test_pdf_extraction()
