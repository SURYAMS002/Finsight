import re
from typing import List, Dict, Any, Optional
from rag.vector_store import FAISSVectorStore

_GLOBAL_VECTOR_STORE: Optional[FAISSVectorStore] = None

def set_vector_store(store: FAISSVectorStore):
    """Sets the global FAISS vector store reference for the tool."""
    global _GLOBAL_VECTOR_STORE
    _GLOBAL_VECTOR_STORE = store

def get_vector_store() -> Optional[FAISSVectorStore]:
    """Gets the global FAISS vector store reference."""
    return _GLOBAL_VECTOR_STORE

def search_report(query: str, top_k: int = 5) -> str:
    """
    Search the uploaded annual report for relevant text chunks and financial evidence.
    Works dynamically across any financial document (TCS, Infosys, Wipro, etc.).
    
    Args:
        query: Natural language query string (e.g. 'What was the total revenue in FY2025?')
        top_k: Number of relevant snippets to retrieve (default: 5)
        
    Returns:
        Formatted evidence string containing snippet text and page citations.
    """
    store = get_vector_store()
    if store is None:
        return "Error: No annual report has been processed or vector store is not loaded."

    # 1. Semantic Vector Search
    results = store.search(query, top_k=top_k)
    retrieved_pages = {r["page"] for r in results}

    # 2. Dynamic Hybrid Boost for Financial Statement Tables across ANY document
    financial_keywords = ["revenue", "profit", "consolidated", "standalone", "income", "expense", "growth", "margin"]
    is_financial_query = any(kw in query.lower() for kw in financial_keywords)

    if is_financial_query and store.chunks_metadata:
        bonus_chunks = []
        table_indicators = [
            "revenue from operations", "statement of profit and loss", 
            "consolidated statement of profit", "profit for the year", 
            "total income", "standalone statement of profit"
        ]
        
        for chunk in store.chunks_metadata:
            page_num = chunk["page"]
            text_lower = chunk["text"].lower()
            
            # Boost chunks containing actual financial statement tables
            if any(ind in text_lower for ind in table_indicators) and page_num not in retrieved_pages:
                item = dict(chunk)
                item["score"] = 0.99  # High score for core statement tables
                bonus_chunks.append(item)
                retrieved_pages.add(page_num)
                if len(bonus_chunks) >= 3:
                    break
        
        # Merge vector results with dynamic statement table chunks
        results = bonus_chunks + results
        results = results[:top_k + 3]

    if not results:
        return f"No relevant evidence found in the annual report for query: '{query}'"

    formatted_outputs = []
    for idx, item in enumerate(results, start=1):
        formatted_outputs.append(
            f"--- Evidence #{idx} (Page {item['page']} | Source: {item['source']}) ---\n"
            f"{item['text']}\n"
        )

    return "\n".join(formatted_outputs)
