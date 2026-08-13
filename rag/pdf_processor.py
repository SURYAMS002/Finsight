import os
import pymupdf
from typing import List, Dict, Any

def extract_pages_from_pdf(pdf_path: str) -> List[Dict[Any, Any]]:
    """
    Extracts text and page metadata from a PDF file using PyMuPDF.
    
    Each page is extracted as a dictionary:
    {
        "page": page_number (1-indexed),
        "text": cleaned_page_text,
        "source": pdf_filename
    }
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    doc = pymupdf.open(pdf_path)
    extracted_pages = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text").strip()
        
        # 1-indexed page number for standard financial citations
        page_num = page_index + 1
        
        extracted_pages.append({
            "page": page_num,
            "text": text,
            "source": filename
        })

    doc.close()
    return extracted_pages
