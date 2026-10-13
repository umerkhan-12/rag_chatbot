"""Embedding providers. All return L2-normalised float32 vectors so inner product == cosine similarity."""
from __future__ import annotations

import logging
import time

import numpy as np

from . import config

log = logging.getLogger("veridesk.embed")


class EmbeddingUnavailable(Exception):
    pass


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class LocalEmbedder:
    name = "local"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # noqa: BLE001
            raise EmbeddingUnavailable(f"sentence-transformers could not be imported ({e}); low memory can cause this") from e
        last = None
        for _attempt in range(2):  # a second try helps when a load fails transiently under memory pressure
            try:
                self.model = SentenceTransformer(config.LOCAL_EMBED_MODEL, device="cpu")
                break
            except Exception as e:  # noqa: BLE001 - offline and not cached, etc.
                last = e
                import gc

                gc.collect()
        else:
            raise EmbeddingUnavailable(f"could not load {config.LOCAL_EMBED_MODEL} ({last})") from last
        self.model_id = f"local:{config.LOCAL_EMBED_MODEL}"
        self.query_prefix = config.LOCAL_EMBED_QUERY_PREFIX if "bge" in config.LOCAL_EMBED_MODEL.lower() else ""

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self.model.encode([self.query_prefix + text], normalize_embeddings=True, show_progress_bar=False))


class OpenAICompatEmbedder:
    """OpenAI text-embedding-3-small, or Gemini embeddings through Google's OpenAI-compatible endpoint."""

    def __init__(self, provider: str) -> None:
        from openai import OpenAI

        if provider == "gemini":
            if not config.GEMINI_API_KEY:
                raise EmbeddingUnavailable("GEMINI_API_KEY not set")
            self.client = OpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL, timeout=60, max_retries=2)
            self.model = config.GEMINI_EMBED_MODEL
        else:
            if not config.OPENAI_API_KEY:
                raise EmbeddingUnavailable("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=60, max_retries=2)
            self.model = config.OPENAI_EMBED_MODEL
        self.name = provider
        self.model_id = f"{provider}:{self.model}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            for attempt in range(4):
                try:
                    r = self.client.embeddings.create(model=self.model, input=batch)
                    out += [d.embedding for d in r.data]
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        raise EmbeddingUnavailable(f"{self.name} embedding failed: {type(e).__name__}") from e
                    time.sleep(2 * (attempt + 1))
        return _normalize(np.array(out))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])


_cache: dict[str, object] = {}


def get_embedder(preferred: str | None = None):
    """Return the first embedder that works, starting from the preferred provider. None => BM25-only."""
    preferred = (preferred or config.EMBED_PROVIDER).lower()
    order = [preferred] + [p for p in ("local", "gemini", "openai") if p != preferred]
    if preferred == "none":
        return None
    for p in order:
        if p in _cache:
            return _cache[p]
        try:
            emb = LocalEmbedder() if p == "local" else OpenAICompatEmbedder(p)
            _cache[p] = emb
            if p != preferred:
                log.warning("Embedding provider '%s' unavailable; using '%s' instead", preferred, p)
            return emb
        except EmbeddingUnavailable as e:
            log.warning("Embedding provider '%s' unavailable: %s", p, e)
    log.error("No embedding provider available - falling back to BM25 keyword retrieval only")
    return None


def get_embedder_by_id(model_id: str):
    """Load the exact embedder an index was built with (so query and document vectors match)."""
    provider = model_id.split(":", 1)[0]
    emb = get_embedder(provider)
    if emb is None or emb.model_id != model_id:
        return None
    return emb
