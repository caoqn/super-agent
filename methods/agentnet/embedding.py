"""Pluggable text encoder for AgentNet's experience pools (added for benchmark runs).

EMBEDDING_BACKEND:
  bge   (default) original AgentNet setting: FlagEmbedding BAAI/bge-large-en-v1.5, local.
        Downloads from huggingface.co by default; or set BGE_MODEL_PATH=/local/path/to/bge-large-en-v1.5.
  api   OpenAI-compatible /embeddings endpoint (EMB_BASE_URL / EMB_API_KEY / EMB_MODEL).
  hash  dependency-free hashed bag-of-ngrams (1024-d). Offline fallback / smoke tests only;
        retrieval quality is lower than bge, so do not use it for reported numbers.
Every encode call is timed (and, for the api backend, costed) in gaia.tracker.
"""
import hashlib
import os
import re
import time

import numpy as np


class _Timed:
    def __init__(self, inner, name):
        self.inner = inner
        self.name = name

    def encode(self, texts, batch_size=256, **kw):
        from gaia.tracker import TRACKER
        t0 = time.time()
        out = self.inner.encode(texts, batch_size=batch_size, **kw)
        if self.name != "api":  # api backend records itself (with tokens)
            n = 1 if isinstance(texts, str) else len(texts)
            TRACKER.record("embedding", time.time() - t0, 0, 0, extra={"backend": self.name, "n": n})
        return out


class HashEncoder:
    dim = 1024

    def _one(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        toks = re.findall(r"\w+", text.lower())
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        for t in feats:
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        n = np.linalg.norm(v)
        if n == 0:
            v[0] = 1.0
            n = 1.0
        return v / n

    def encode(self, texts, batch_size=256, **kw):
        if isinstance(texts, str):
            return self._one(texts)
        return np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class ApiEncoder:
    def encode(self, texts, batch_size=256, **kw):
        from gaia.llm import embed
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        items = [t if t.strip() else " " for t in items]
        vecs = []
        for i in range(0, len(items), min(batch_size, 128)):
            vecs.extend(embed(items[i:i + min(batch_size, 128)]))
        arr = np.asarray(vecs, dtype=np.float32)
        return arr[0] if single else arr


def build_encoder():
    backend = os.environ.get("EMBEDDING_BACKEND", "bge").lower()
    if backend == "hash":
        return _Timed(HashEncoder(), "hash")
    if backend == "api":
        return _Timed(ApiEncoder(), "api")
    from FlagEmbedding import FlagModel
    path = os.environ.get("BGE_MODEL_PATH", "BAAI/bge-large-en-v1.5")
    model = FlagModel(path,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                      use_fp16=os.environ.get("BGE_FP16", "0") == "1")
    return _Timed(model, "bge")


encode_model = build_encoder()
