import re
from typing import List, Dict, Any

def chunk_pages(
    pages: List[Dict[str, Any]], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Chunks extracted PDF pages while preserving page-level metadata.
    
    Each output chunk contains:
    {
        "chunk_id": "annual_report.pdf_p87_c0",
        "page": 87,
        "source": "annual_report.pdf",
        "text": "...chunk text..."
    }
    """
    chunks = []

    for page_data in pages:
        page_num = page_data["page"]
        source = page_data["source"]
        text = page_data["text"]

        if not text or not text.strip():
            continue

        # Split page text into chunks with overlap
        page_chunks = split_text_with_overlap(text, chunk_size, chunk_overlap)
        
        for idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{source}_p{page_num}_c{idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "source": source,
                "text": chunk_text
            })

    return chunks


def split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits text into chunks of maximum `chunk_size` characters with `chunk_overlap`.
    Attempts to break cleanly on newlines or sentence boundaries where possible.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to find a line break or period nearby to split cleanly
        split_at = -1
        # Look for double newline or newline
        for idx in range(end, max(start + chunk_size // 2, end - 200), -1):
            if text[idx] == '\n':
                split_at = idx
                break
        
        if split_at == -1:
            # Fall back to space
            for idx in range(end, max(start + chunk_size // 2, end - 200), -1):
                if text[idx] == ' ':
                    split_at = idx
                    break

        if split_at == -1:
            split_at = end

        chunk_content = text[start:split_at].strip()
        if chunk_content:
            chunks.append(chunk_content)

        # Advance start position considering overlap
        start = max(split_at - chunk_overlap, start + 1)

    return chunks
