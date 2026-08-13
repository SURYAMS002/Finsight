import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from rag.embeddings import GeminiEmbedder

class FAISSVectorStore:
    """
    FAISS vector index for semantic retrieval over chunked PDF documents.
    Stores and maps embeddings -> chunk text & page metadata.
    """
    def __init__(self, embedder: GeminiEmbedder = None):
        self.embedder = embedder or GeminiEmbedder()
        self.index = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.dimension = None

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalizes vectors for Cosine Similarity search with IndexFlatIP."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    def build_from_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        batch_size: int = 30,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Generates embeddings for chunk texts and populates FAISS index.
        """
        if not chunks:
            raise ValueError("Cannot build vector store from empty chunks list.")

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(
            texts, 
            batch_size=batch_size, 
            progress_callback=progress_callback
        )
        
        self.dimension = embeddings.shape[1]
        normalized_embeddings = self._normalize(embeddings)

        # Inner product on normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(normalized_embeddings)
        self.chunks_metadata = list(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search for a query.
        Returns top_k matching chunk objects with 'score' (similarity metric).
        """
        if self.index is None or not self.chunks_metadata:
            raise ValueError("FAISS vector store is empty or not initialized.")

        query_vec = self.embedder.embed_query(query).reshape(1, -1)
        normalized_query_vec = self._normalize(query_vec)

        scores, indices = self.index.search(normalized_query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue
            item = dict(self.chunks_metadata[idx])
            item["score"] = float(score)
            results.append(item)

        return results

    def save(self, dir_path: str = "faiss_store") -> None:
        """Saves FAISS index and metadata to disk."""
        os.makedirs(dir_path, exist_ok=True)
        index_path = os.path.join(dir_path, "index.faiss")
        meta_path = os.path.join(dir_path, "metadata.json")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, ensure_ascii=False, indent=2)

    def load(self, dir_path: str = "faiss_store") -> bool:
        """Loads FAISS index and metadata from disk if available."""
        index_path = os.path.join(dir_path, "index.faiss")
        meta_path = os.path.join(dir_path, "metadata.json")

        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return False

        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.chunks_metadata = json.load(f)

        self.dimension = self.index.d
        return True
