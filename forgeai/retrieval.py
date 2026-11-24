"""In-memory semantic retrieval with a dependency-free keyword baseline."""

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Protocol

from forgeai.documents import Passage


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
STOP_WORDS = frozenset(
    "a an and are can compare for from has i in is it me need of on or please "
    "should that the their this to what when where which who will with would you".split()
)


class Encoder(Protocol):
    def encode(self, texts: list[str], *, batch_size: int = 16): ...


class SentenceEncoder:
    """Load the pinned public model. Only the model files are fetched on first use."""

    def __init__(self, device: str = "cuda:0", *, local_only: bool = True):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=device,
            local_files_only=local_only, model_kwargs={"use_safetensors": True},
        )

    def encode(self, texts: list[str], *, batch_size: int = 16):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )


def terms(text: str) -> Counter:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return Counter(token for token in tokens if token not in STOP_WORDS)


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


class Index:
    def __init__(self, passages: list[Passage], encoder: Encoder | None = None):
        self.passages = tuple(passages)
        self.encoder = encoder
        self.lexical = [terms(passage.text) for passage in passages]
        self.vectors = self._vectors([item.text for item in passages]) if encoder and passages else None

    @property
    def mode(self) -> str:
        return "semantic" if self.encoder else "keyword"

    def _vectors(self, texts: list[str]):
        import numpy as np

        vectors = np.asarray(self.encoder.encode(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("The embedding model returned an invalid matrix.")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError("The embedding model returned a nonfinite or zero vector.")
        return vectors / norms[:, None]

    def _keyword_scores(self, query: str) -> list[float]:
        query_terms = terms(query)
        query_norm = math.sqrt(sum(value * value for value in query_terms.values()))
        if not query_norm:
            return [0.0] * len(self.passages)
        scores = []
        for item in self.lexical:
            dot = sum(count * item.get(word, 0) for word, count in query_terms.items())
            norm = math.sqrt(sum(value * value for value in item.values()))
            scores.append(dot / (query_norm * norm) if norm else 0.0)
        return scores

    def search(self, query: str, limit: int = 5) -> list[Hit]:
        if limit < 1 or not terms(query) or not self.passages:
            return []
        if self.vectors is None:
            scores = self._keyword_scores(query)
            floor = 0.0
        else:
            scores = self.vectors @ self._vectors([query])[0]
            floor = 0.28
        hits = [
            Hit(passage, float(score))
            for passage, score in zip(self.passages, scores, strict=True)
            if float(score) > floor
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
