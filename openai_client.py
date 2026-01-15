import time
from typing import List, Optional, Tuple

from openai import OpenAI


class OpenAIHelper:
    def __init__(self, api_key: str, embedding_model: str, llm_model: str):
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    def embed_documents(self, texts: List[str], batch_size: int = 128, max_retries: int = 6) -> List[List[float]]:
        # Embed a list of texts in batches with retries.
        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs = self._embed_batch(batch, max_retries=max_retries)
            all_vecs.extend(vecs)
        return all_vecs

    def embed_query(self, text: str, max_retries: int = 6) -> List[float]:
        # Embed a single text with retries.
        vecs = self._embed_batch([text], max_retries=max_retries)
        return vecs[0]

    def _embed_batch(self, batch: List[str], max_retries: int = 6) -> List[List[float]]:
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                )
                return [item.embedding for item in resp.data]
            except Exception as e:
                last_err = e
                # Exponential backoff
                sleep_s = min(2 ** attempt, 30)
                time.sleep(sleep_s)
        raise RuntimeError(f"Embedding failed after retries: {last_err}")

    def chat_completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.5,
        system: Optional[str] = None,
        max_retries: int = 6,
    ) -> Tuple[str, int]:
        # Create a chat completion and return (text, total_tokens_used).
        # Uses retries with exponential backoff.
        last_err = None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content or ""
                total_tokens = 0
                if getattr(resp, "usage", None) is not None and resp.usage is not None:
                    total_tokens = int(resp.usage.total_tokens or 0)
                return text, total_tokens
            except Exception as e:
                last_err = e
                sleep_s = min(2 ** attempt, 30)
                time.sleep(sleep_s)
        raise RuntimeError(f"ChatCompletion failed after retries: {last_err}")
