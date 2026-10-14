"""OCR for scanned PDFs with Gemini vision.

Image-only pages have no text layer, so PyMuPDF/pdfplumber return nothing. Each such page is rendered to JPEG and
Gemini transcribes it verbatim as Markdown (headings, tables, lists, reading order for multi-column layouts).
Pages with a real text layer are read directly. Results are cached per page (data/pages/<doc hash>/pNNNN.json),
so re-ingestion is free and an interrupted run resumes.
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import json
import logging
import re
import time
from pathlib import Path

from . import config

log = logging.getLogger("veridesk.ocr")

OCR_PROMPT = """Transcribe ALL text on this page of a document exactly as written.
Rules:
- Output Markdown only. Use '#' / '##' / '###' for headings and section titles as they appear.
- Keep the natural reading order. For multi-column layouts, finish the left column top-to-bottom, then the right column.
- Tables become Markdown tables (keep every row and column). Lists become '-' bullets (keep numbering if shown).
- Copy names, numbers, dates, fees, codes and units exactly. Do not correct, summarise, translate, or add anything.
- Skip repeated page decorations (running header/footer banners, social-media icons). Ignore photos and logos.
- If some text is unreadable write [illegible].
- First line must be: PRINTED_PAGE: <the page number printed on the page, or none>
"""


def _client():
    from openai import OpenAI

    return OpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL, timeout=120, max_retries=2)


def _ocr_image(client, jpeg: bytes, model: str) -> str:
    b64 = base64.b64encode(jpeg).decode()
    r = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    )
    return (r.choices[0].message.content or "").strip()


def ocr_pdf(pdf_path: Path, doc_hash16: str, progress=None, workers: int = 8, dpi: int = 150) -> list[dict]:
    """Return [{"page", "printed_page", "text", "source"}] for every page, in order."""
    import pymupdf

    cache = config.DATA_DIR / "pages" / doc_hash16
    cache.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    results: dict[int, dict] = {}
    todo: list[tuple[int, bytes]] = []
    for n in range(1, doc.page_count + 1):
        cpath = cache / f"p{n:04d}.json"
        if cpath.exists():
            rec = json.loads(cpath.read_text(encoding="utf-8"))
            if rec.get("source") != "failed":
                results[n] = rec
                continue
        page = doc[n - 1]
        text = page.get_text().strip()
        if len(text) > 100:
            rec = {"page": n, "printed_page": None, "text": text, "source": "text"}
            cpath.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            results[n] = rec
        else:
            todo.append((n, page.get_pixmap(dpi=dpi).tobytes("jpeg", jpg_quality=80)))
    doc.close()

    if todo:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("This PDF is scanned and needs OCR, but GEMINI_API_KEY is not set.")
        client = _client()
        models = [config.GEMINI_MODEL] + [m for m in [config.GEMINI_FALLBACK_MODEL] if m and m != config.GEMINI_MODEL]

        def work(item):
            n, img = item
            last = None
            for model in models:
                for attempt in range(3):
                    try:
                        md = _ocr_image(client, img, model)
                        m = re.match(r"\s*PRINTED_PAGE:\s*(\S+)", md)
                        printed = m.group(1) if m and m.group(1).lower() != "none" else None
                        md = re.sub(r"^\s*PRINTED_PAGE:.*\n?", "", md).strip()
                        md = re.sub(r"^```(markdown)?\s*|\s*```$", "", md).strip()
                        rec = {"page": n, "printed_page": printed, "text": md, "source": "ocr", "model": model}
                        (cache / f"p{n:04d}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                        return rec
                    except Exception as e:  # noqa: BLE001
                        last = e
                        time.sleep(2 * (attempt + 1))
            log.error("OCR failed for page %d: %s", n, last)
            return {"page": n, "printed_page": None, "text": "", "source": "failed"}

        done, t0 = 0, time.time()
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for rec in ex.map(work, todo):
                results[rec["page"]] = rec
                done += 1
                if progress:
                    progress(f"OCR {done}/{len(todo)} scanned pages ({time.time() - t0:.0f}s)")
    return [results[n] for n in sorted(results)]
