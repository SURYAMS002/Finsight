import time
import re
import httpx
import numpy as np
from typing import List, Union, Callable, Optional
from google import genai
from google.genai.errors import APIError, ClientError
from config import get_gemini_api_key, DEFAULT_EMBEDDING_MODEL

class GeminiEmbedder:
    """
    Generates vector embeddings using the Google Gemini API (google-genai SDK).
    Supports single text and batch embedding generation with exponential backoff for rate limits & network retries.
    """
    def __init__(self, api_key: str = None, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.api_key = api_key or get_gemini_api_key()
        self.model_name = model_name

    def _get_client(self):
        current_key = self.api_key or get_gemini_api_key()
        return genai.Client(api_key=current_key)

    def embed_query(self, text: str, max_retries: int = 6) -> np.ndarray:
        """
        Embeds a single text with exponential backoff on rate limits (429) and network drops.
        """
        backoff = 4.0
        client = self._get_client()
        
        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
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
                err_str = str(e)
                if attempt < max_retries - 1:
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        delay_match = re.search(r"retryDelay': '(\d+)", err_str)
                        sleep_time = int(delay_match.group(1)) + 3 if delay_match else int(backoff)
                        print(f"Embedding Rate Limit 429. Sleeping {sleep_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                        backoff *= 1.5
                        continue
                    else:
                        time.sleep(backoff)
                        backoff *= 1.5
                        continue
                raise e

    def embed_texts(
        self, 
        texts: List[str], 
        batch_size: int = 30, 
        sleep_sec: float = 0.5, 
        max_retries: int = 6,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Embeds a list of text strings in batches with rate-limit & network error protection.
        """
        all_embeddings = []
        total_texts = len(texts)
        client = self._get_client()

        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_success = False
            backoff = 4.0

            for attempt in range(max_retries):
                try:
                    response = client.models.embed_content(
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
                    err_str = str(e)
                    if attempt < max_retries - 1:
                        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                            delay_match = re.search(r"retryDelay': '(\d+)", err_str)
                            sleep_time = int(delay_match.group(1)) + 3 if delay_match else int(backoff)
                            print(f"Batch Embedding 429 at index {i}. Sleeping {sleep_time}s...")
                            time.sleep(sleep_time)
                            backoff *= 1.5
                            continue
                        else:
                            time.sleep(backoff)
                            backoff *= 1.5
                            continue
                    break

            if not batch_success:
                # Fall back item-by-item with rate limit protection
                for single_text in batch_texts:
                    vec = self.embed_query(single_text, max_retries=max_retries)
                    all_embeddings.append(vec.tolist())
                    time.sleep(0.1)

            if progress_callback:
                progress_callback(min(i + batch_size, total_texts), total_texts)

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        return np.array(all_embeddings, dtype=np.float32)
