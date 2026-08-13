import time
import httpx
import numpy as np
from typing import List, Union, Callable, Optional
from google import genai
from google.genai.errors import APIError, ClientError
from config import GEMINI_API_KEY, DEFAULT_EMBEDDING_MODEL

class GeminiEmbedder:
    """
    Generates vector embeddings using the Google Gemini API (google-genai SDK).
    Supports single text and batch embedding generation with exponential backoff for rate limits & network retries.
    """
    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    def embed_query(self, text: str, max_retries: int = 5) -> np.ndarray:
        """
        Embeds a single text with exponential backoff on rate limits (429) and network drops.
        """
        backoff = 2.0
        for attempt in range(max_retries):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                if hasattr(response, 'embedding') and hasattr(response.embedding, 'values'):
                    vector = response.embedding.values
                elif hasattr(response, 'embeddings') and len(response.embeddings) > 0:
                    vector = response.embeddings[0].values
                else:
                    raise ValueError(f"Unexpected embedding response structure: {response}")

                return np.array(vector, dtype=np.float32)
            except (APIError, ClientError, httpx.HTTPError, Exception) as e:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                raise e

    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: int = 30, 
        sleep_sec: float = 0.3, 
        max_retries: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Embeds a list of text strings in batches with rate-limit & network error protection.
        """
        all_embeddings = []
        total_texts = len(texts)

        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_success = False
            backoff = 2.0

            for attempt in range(max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch_texts
                    )
                    if hasattr(response, 'embeddings'):
                        batch_vectors = [e.values for e in response.embeddings]
                    else:
                        raise ValueError(f"Unexpected batch embedding response: {response}")
                    
                    all_embeddings.extend(batch_vectors)
                    batch_success = True
                    break
                except (APIError, ClientError, httpx.HTTPError, Exception) as e:
                    if attempt < max_retries - 1:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    break

            if not batch_success:
                # Fall back item-by-item with rate limit protection
                for single_text in batch_texts:
                    vec = self.embed_query(single_text, max_retries=max_retries)
                    all_embeddings.append(vec.tolist())
                    time.sleep(0.05)

            if progress_callback:
                progress_callback(min(i + batch_size, total_texts), total_texts)

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        return np.array(all_embeddings, dtype=np.float32)
