"""Document index: chunks + FAISS dense vectors + BM25, persisted per (document hash, settings).

Re-ingesting the same PDF with the same settings reuses the saved index (no duplicate chunks).
A different PDF or different chunk/embedding settings produces a new index directory.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .chunking import Chunk, chunk_blocks
from .embeddings import get_embedder, get_embedder_by_id
from .ingest import IngestError, extract, file_sha256

log = logging.getLogger("veridesk.store")

STOPWORDS = set(
    """a an the and or but if then else of to in on at by for with about as into from up down out over under
    is are was were be been being am do does did done have has had having can could should would will shall may
    might must i me my we our you your he she it its they them their this that these those there here what which
    who whom whose when where why how all any both each few more most other some such no nor not only own same so
    than too very s t just don also please tell explain describe give know want need does say says said
    manual document guide handbook pdf according now today current currently right exactly exact
    many much often lot lots really actually okay ok hi hello thank thanks kindly sir maam bro anyone someone
    something anything everything way get got use using let possible sure mean meaning since because while
    although though whether yes still already ever even like said one per""".split()
)


def tokenize(text: str) -> list[str]:
    # "BE" / "B.E." (Bachelor of Engineering) would otherwise be lowercased into the stopword "be" and lost.
    text = re.sub(r"\bB\.?\s?E\b\.?", " bachelorengineering ", text)
    toks = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", text.lower())
    out = []
    for t in toks:
        t = t.replace(",", "")
        if t in STOPWORDS:  # keep stopwords unstemmed so later stopword filters recognise them ("does" -> not "doe")
            out.append(t)
            continue
        if len(t) > 6 and t.endswith("ers"):  # foreigners -> foreign (applied to query and document alike)
            t = t[:-3]
        elif len(t) > 6 and t.endswith("er"):  # foreigner -> foreign
            t = t[:-2]
        elif len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 5 and t.endswith("ing"):
            t = t[:-3]
        elif len(t) > 4 and t.endswith("ed") and not t.endswith("eed"):
            t = t[:-2]
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and (len(t) > 2 or t.isdigit())]


def bm25_terms(text: str) -> list[str]:
    """Unigrams plus adjacent-word bigrams, so exact phrases ("studio fee", "security deposit") rank higher than
    chunks that merely contain one of the words many times."""
    toks = [t for t in tokenize(text) if t not in STOPWORDS]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


@dataclass
class Hit:
    chunk: Chunk
    dense: float = 0.0
    bm25: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    fused: float = 0.0
    rerank: float | None = None
    neighbor: bool = False  # added as context next to a strong hit, not retrieved on its own

    def to_dict(self) -> dict:
        d = self.chunk.to_dict()
        d.pop("embed_text", None)
        d.update(
            dense=round(self.dense, 4),
            bm25=round(self.bm25, 3),
            dense_rank=self.dense_rank,
            bm25_rank=self.bm25_rank,
            fused=round(self.fused, 5),
            rerank=None if self.rerank is None else round(self.rerank, 3),
            pages_label=self.chunk.pages_label,
            neighbor=self.neighbor,
        )
        return d


@dataclass
class SearchResult:
    hits: list[Hit]
    top_dense: float
    lexical_coverage: float
    query_terms: list[str]
    mode: str
    timings: dict = field(default_factory=dict)
    missing_terms: list[str] = field(default_factory=list)  # query terms that appear nowhere in the document


def _settings_key(embed_id: str) -> str:
    s = json.dumps({**config.chunk_settings(), "embed": embed_id}, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:8]


class DocumentIndex:
    def __init__(self, directory: Path, meta: dict, chunks: list[Chunk], vectors: np.ndarray | None, embedder) -> None:
        self.dir = directory
        self.meta = meta
        self.chunks = chunks
        self.embedder = embedder
        self.faiss_index = None
        if vectors is not None and embedder is not None and len(chunks):
            import faiss

            self.faiss_index = faiss.IndexFlatIP(vectors.shape[1])
            self.faiss_index.add(vectors.astype("float32"))
        from rank_bm25 import BM25Okapi

        self._bm25_docs = [bm25_terms(c.embed_text) for c in chunks]
        self.bm25 = BM25Okapi(self._bm25_docs) if chunks else None
        self.vocab = {t for c in chunks for t in tokenize(c.embed_text)}
        self._reranker = None

    # ------------------------------------------------------------------ build / load
    @classmethod
    def build(cls, pdf_path: Path, original_name: str, progress=None) -> "DocumentIndex":
        t0 = time.time()
        doc_hash = file_sha256(pdf_path)
        embedder = get_embedder()
        embed_id = embedder.model_id if embedder else "none"
        directory = config.INDEX_DIR / f"{doc_hash[:16]}_{_settings_key(embed_id)}"
        if (directory / "meta.json").exists():
            try:
                idx = cls.load(directory)
                idx.meta["reused"] = True
                idx.meta["display_name"] = original_name
                log.info("Reusing existing index %s", directory.name)
                return idx
            except Exception as e:  # noqa: BLE001
                log.warning("Saved index unreadable (%s); rebuilding", e)
                shutil.rmtree(directory, ignore_errors=True)

        if progress:
            progress("Extracting text")
        ext = extract(pdf_path, config.MAX_PAGES, progress=progress)
        if progress:
            progress("Chunking")
        chunks = chunk_blocks(ext.blocks, ext.title, doc_hash)
        printed_map = getattr(ext, "printed_pages", {}) or {}
        for c in chunks:
            a, b = printed_map.get(str(c.page_start)), printed_map.get(str(c.page_end))
            if a and a != str(c.page_start):
                c.printed = a if (c.page_start == c.page_end or not b) else f"{a}-{b}"
        if not chunks:
            raise IngestError("No usable text chunks were produced from this PDF.")
        t_extract = time.time() - t0
        if embedder is None and config.EMBED_PROVIDER != "none":
            ext.warnings.append(
                "Semantic embeddings were unavailable, so this document uses keyword (BM25) retrieval only. "
                "Free some memory or check the setup, then remove and re-upload the document."
            )
        elif embedder is not None and embedder.name != config.EMBED_PROVIDER:
            ext.warnings.append(f"Preferred embeddings '{config.EMBED_PROVIDER}' unavailable; using '{embedder.model_id}'.")

        vectors = None
        if embedder is not None:
            if progress:
                progress(f"Embedding {len(chunks)} chunks")
            try:
                vectors = embedder.embed_docs([c.embed_text for c in chunks])
            except Exception as e:  # noqa: BLE001
                log.error("Embedding failed (%s); continuing with BM25 only", e)
                ext.warnings.append("Embedding failed; using keyword (BM25) retrieval only.")
                embedder, embed_id = None, "none"
                directory = config.INDEX_DIR / f"{doc_hash[:16]}_{_settings_key(embed_id)}"

        directory.mkdir(parents=True, exist_ok=True)
        stored_pdf = directory / "source.pdf"
        if Path(pdf_path).resolve() != stored_pdf.resolve():
            shutil.copyfile(pdf_path, stored_pdf)
        heads = [c.section for c in chunks]
        meta = {
            "doc_hash": doc_hash,
            "display_name": original_name,
            "title": ext.title,
            "pages": ext.page_count,
            "chars": ext.char_count,
            "chunks": len(chunks),
            "extractor": ext.extractor,
            "warnings": ext.warnings,
            "embed_model": embed_id,
            "settings": config.chunk_settings(),
            "sections": list(dict.fromkeys(h for h in heads if h != "(no heading)"))[:200],
            "tables": sum(c.has_table for c in chunks),
            "suspicious_chunks": [c.id for c in chunks if c.suspicious],
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build_seconds": round(time.time() - t0, 2),
            "extract_seconds": round(t_extract, 2),
            "reused": False,
        }
        (directory / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")
        if vectors is not None:
            np.save(directory / "vectors.npy", vectors)
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return cls(directory, meta, chunks, vectors, embedder)

    @classmethod
    def load(cls, directory: Path) -> "DocumentIndex":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        vectors, embedder = None, None
        if meta.get("embed_model", "none") != "none" and (directory / "vectors.npy").exists():
            embedder = get_embedder_by_id(meta["embed_model"])
            if embedder is not None:
                vectors = np.load(directory / "vectors.npy")
            else:
                meta.setdefault("warnings", []).append(
                    f"Embedding model {meta['embed_model']} unavailable at load time; using keyword retrieval only."
                )
        return cls(directory, meta, chunks, vectors, embedder)

    # ------------------------------------------------------------------ search
    def spelling_hints(self, query: str) -> list[str]:
        """Likely intended document words for misspelled query words ("semestr" -> "semester").
        Only words absent from the document are considered; matches must be very close (difflib ratio >= 0.8)."""
        import difflib

        if not hasattr(self, "_raw_vocab"):
            words = {w for c in self.chunks for w in re.findall(r"[a-z]{4,}", c.text.lower())}
            self._raw_vocab = sorted(words)
            self._raw_vocab_set = words
        hints = []
        for w in re.findall(r"[A-Za-z]{4,}", query):
            lw = w.lower()
            if lw in self._raw_vocab_set or lw in STOPWORDS:
                continue
            m = difflib.get_close_matches(lw, self._raw_vocab, n=1, cutoff=0.8)
            if m and m[0] != lw:
                hints.append(m[0])
        return hints[:5]

    @property
    def mode(self) -> str:
        return "hybrid" if self.faiss_index is not None else "bm25"

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not config.USE_RERANKER or not hits:
            return hits
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(config.RERANK_MODEL, device="cpu")
            scores = self._reranker.predict([(query, h.chunk.text) for h in hits])
            for h, s in zip(hits, scores):
                h.rerank = float(s)
            return sorted(hits, key=lambda h: h.rerank, reverse=True)
        except Exception as e:  # noqa: BLE001
            log.warning("Reranker unavailable (%s); using fused order", e)
            return hits

    def search(self, query: str, top_k: int | None = None) -> SearchResult:
        top_k = top_k or config.TOP_K
        timings = {}
        t0 = time.time()
        n = len(self.chunks)
        cand = min(config.CANDIDATES, n)
        hits: dict[int, Hit] = {}

        if self.faiss_index is not None:
            qv = self.embedder.embed_query(query)
            scores, ids = self.faiss_index.search(qv, cand)
            for rank, (i, s) in enumerate(zip(ids[0], scores[0])):
                if i < 0:
                    continue
                h = hits.setdefault(int(i), Hit(self.chunks[int(i)]))
                h.dense, h.dense_rank = float(s), rank
        timings["dense_ms"] = round((time.time() - t0) * 1000, 1)

        t1 = time.time()
        q_tokens = bm25_terms(query)
        if self.bm25 is not None and q_tokens:
            bm = self.bm25.get_scores(q_tokens)
            order = np.argsort(-bm)[:cand]
            for rank, i in enumerate(order):
                if bm[i] <= 0:
                    break
                h = hits.setdefault(int(i), Hit(self.chunks[int(i)]))
                h.bm25, h.bm25_rank = float(bm[i]), rank
        timings["bm25_ms"] = round((time.time() - t1) * 1000, 1)

        for h in hits.values():
            if h.dense_rank is not None:
                h.fused += config.DENSE_WEIGHT / (config.RRF_K + h.dense_rank + 1)
            if h.bm25_rank is not None:
                h.fused += config.BM25_WEIGHT / (config.RRF_K + h.bm25_rank + 1)
        # fill missing dense scores for BM25-only candidates so the inspector shows comparable numbers
        if self.faiss_index is not None:
            missing = [i for i, h in hits.items() if h.dense_rank is None]
            if missing:
                vecs = np.vstack([self.faiss_index.reconstruct(i) for i in missing])
                sims = vecs @ qv[0]
                for i, s in zip(missing, sims):
                    hits[i].dense = float(s)

        pool = config.RERANK_POOL if config.USE_RERANKER else max(top_k * 2, 10)
        ranked = sorted(hits.values(), key=lambda h: h.fused, reverse=True)[:pool]
        t2 = time.time()
        ranked = self._rerank(query, ranked)
        timings["rerank_ms"] = round((time.time() - t2) * 1000, 1)

        top_dense = max((h.dense for h in hits.values()), default=0.0)
        selected = []
        for h in ranked:
            if len(selected) >= top_k:
                break
            # context filter: drop chunks much weaker than the best one unless BM25 strongly agrees
            if self.faiss_index is not None and selected and h.dense < top_dense * config.RELATIVE_SCORE_CUTOFF and (h.bm25_rank is None or h.bm25_rank > 2):
                continue
            selected.append(h)

        # Neighbour expansion: rules and lists often continue across a chunk boundary (a heading line ending one
        # chunk, its list starting the next). Add the chunks just before/after the strongest hits, placed next to
        # their anchor so the model reads them in document order. Ranking of the original hits is preserved.
        if config.NEIGHBOR_TOP > 0 and selected:
            taken = {h.chunk.index for h in selected}
            budget = max(0, config.MAX_PASSAGES - len(selected))
            expanded: list[Hit] = []
            for rank, h in enumerate(selected):
                before, after = [], []
                if rank < config.NEIGHBOR_TOP:
                    for j, side in ((h.chunk.index - 1, before), (h.chunk.index + 1, after)):
                        if budget and 0 <= j < n and j not in taken and abs(self.chunks[j].page_start - h.chunk.page_end) <= 1:
                            nh = Hit(self.chunks[j], neighbor=True)
                            if self.faiss_index is not None:
                                nh.dense = float(self.faiss_index.reconstruct(j) @ qv[0])
                            side.append(nh)
                            taken.add(j)
                            budget -= 1
                expanded += before + [h] + after
            selected = expanded

        qt = set(content_terms(query))
        top_terms = set()
        for h in [x for x in selected if not x.neighbor][:3]:
            top_terms |= set(tokenize(h.chunk.embed_text))
        coverage = len(qt & top_terms) / len(qt) if qt else 0.0
        timings["total_ms"] = round((time.time() - t0) * 1000, 1)
        missing = sorted(t for t in qt if t not in self.vocab)
        return SearchResult(selected, top_dense, coverage, sorted(qt), self.mode, timings, missing)


# ---------------------------------------------------------------------- active document
class ActiveDocument:
    """Holds the currently loaded document; thread-safe swap on upload/reset."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.index: DocumentIndex | None = None

    def load_saved(self) -> None:
        if not config.ACTIVE_FILE.exists():
            return
        try:
            info = json.loads(config.ACTIVE_FILE.read_text(encoding="utf-8"))
            d = Path(info["dir"])
            if not d.is_absolute():
                d = config.INDEX_DIR / d
            idx = DocumentIndex.load(d)
            idx.meta["display_name"] = info.get("display_name", idx.meta.get("display_name"))
            self.index = idx
            log.info("Loaded active document: %s (%d chunks)", idx.meta["display_name"], len(idx.chunks))
        except Exception as e:  # noqa: BLE001
            log.warning("Could not restore active document: %s", e)

    def set(self, idx: DocumentIndex) -> None:
        with self._lock:
            self.index = idx
            config.ACTIVE_FILE.write_text(
                json.dumps({"dir": idx.dir.name, "display_name": idx.meta.get("display_name")}), encoding="utf-8"
            )

    def clear(self) -> None:
        with self._lock:
            self.index = None
            config.ACTIVE_FILE.unlink(missing_ok=True)


ACTIVE = ActiveDocument()
