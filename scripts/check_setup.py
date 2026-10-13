"""Pre-flight check. Verifies packages, keys (without printing them), model access, Ollama and the embedding model.

    python scripts/check_setup.py
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def line(tag, msg):
    print(f"{tag} {msg}")


def main() -> int:
    failures = 0
    print(f"Python {sys.version.split()[0]}")
    for mod in ["fastapi", "uvicorn", "multipart", "dotenv", "pymupdf", "pdfplumber", "pypdf", "numpy", "faiss",
                "rank_bm25", "openai", "anthropic", "httpx"]:
        try:
            importlib.import_module(mod)
            line(OK, f"package {mod}")
        except Exception as e:  # noqa: BLE001
            line(BAD, f"package {mod}: {e}")
            failures += 1

    from app import config, llm, rag

    print("\n-- Keys (values are never printed) --")
    for name, val in [("GEMINI_API_KEY", config.GEMINI_API_KEY), ("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY),
                      ("OPENAI_API_KEY", config.OPENAI_API_KEY)]:
        line(OK if val else WARN, f"{name}: {'set (' + str(len(val)) + ' chars)' if val else 'not set'}")

    print("\n-- LLM providers (live test call) --")
    chain = rag.provider_chain()
    print("chain:", " -> ".join(f"{p}({llm.model_name(p)})" for p in chain))
    for p in chain:
        if p == "extractive":
            line(OK, "extractive fallback (always available)")
            continue
        if p == "ollama" and not rag._ollama_up():
            line(WARN, "ollama: server not running (start the Ollama app for the offline fallback)")
            continue
        t0 = time.time()
        try:
            out = llm.complete(p, "Reply with exactly: pong", [{"role": "user", "content": "ping"}])
            line(OK, f"{p} ({llm.model_name(p)}): replied {out.strip()[:30]!r} in {time.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            line(BAD, f"{p} ({llm.model_name(p)}): {e}")
            if p == config.LLM_PROVIDER:
                failures += 1

    print("\n-- Embeddings --")
    from app.embeddings import get_embedder

    t0 = time.time()
    emb = get_embedder()
    if emb is None:
        line(WARN, "no embedder available -> BM25-only retrieval")
    else:
        v = emb.embed_query("test")
        line(OK, f"{emb.model_id}: dim={v.shape[1]} ({time.time() - t0:.1f}s incl. load)")

    print("\n-- Data --")
    line(OK, f"data dir: {config.DATA_DIR}")
    from app.store import ACTIVE

    ACTIVE.load_saved()
    if ACTIVE.index:
        line(OK, f"active document: {ACTIVE.index.meta['display_name']} ({len(ACTIVE.index.chunks)} chunks, {ACTIVE.index.mode})")
    else:
        line(WARN, "no active document (upload one in the UI or run scripts/ingest.py)")
    print("\nRESULT:", "READY" if failures == 0 else f"{failures} problem(s) - see [FAIL] lines")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
