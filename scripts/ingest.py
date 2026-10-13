"""Ingest a PDF from the command line and make it the active document.

    python scripts/ingest.py path/to/manual.pdf          # build (or reuse) index, set active
    python scripts/ingest.py path/to/manual.pdf --show   # also print every chunk for inspection
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag  # noqa: E402
from app.ingest import IngestError  # noqa: E402
from app.store import ACTIVE, DocumentIndex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--show", action="store_true", help="print all chunks")
    ap.add_argument("--no-samples", action="store_true", help="skip LLM-generated sample questions")
    a = ap.parse_args()
    pdf = Path(a.pdf)
    if not pdf.exists():
        print(f"File not found: {pdf}")
        return 2
    t0 = time.time()
    try:
        idx = DocumentIndex.build(pdf, pdf.name, progress=lambda s: print(f"  - {s}..."))
    except IngestError as e:
        print(f"INGESTION FAILED: {e}")
        return 1
    if not idx.meta.get("samples"):
        idx.meta["samples"] = rag.sample_questions(idx, use_llm=not a.no_samples)
        (idx.dir / "meta.json").write_text(json.dumps(idx.meta, ensure_ascii=False, indent=1), encoding="utf-8")
    ACTIVE.set(idx)
    m = idx.meta
    print(f"\nOK  '{m['display_name']}'  title='{m['title']}'")
    print(f"    pages={m['pages']} chars={m['chars']} chunks={m['chunks']} tables={m['tables']} extractor={m['extractor']}")
    print(f"    embeddings={m['embed_model']} retrieval={idx.mode} reused={m.get('reused')} time={time.time() - t0:.1f}s")
    print(f"    index dir: {idx.dir}")
    if m.get("suspicious_chunks"):
        print(f"    !! {len(m['suspicious_chunks'])} chunk(s) contain instruction-like text (treated as untrusted data)")
    for w in m.get("warnings", []):
        print(f"    WARNING: {w}")
    print("    sample questions:", *[f"\n      - {q}" for q in m.get("samples", [])])
    if a.show:
        for c in idx.chunks:
            print(f"\n--- [{c.index}] {c.id} {c.pages_label} | {c.section}{' | TABLE' if c.has_table else ''}{' | SUSPICIOUS' if c.suspicious else ''}")
            print(c.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
