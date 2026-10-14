"""PDF -> structured blocks (headings, paragraphs, list items, tables) with page numbers.

Primary extractor: PyMuPDF (fast, gives font sizes/bold flags for heading detection, and table finding).
Fallback extractors: pdfplumber, then pypdf (plain text per page, headings guessed from text patterns).
"""
from __future__ import annotations

import hashlib
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


class IngestError(Exception):
    """A problem the user must know about (scanned PDF, encrypted, empty...)."""


@dataclass
class Block:
    kind: str  # heading | para | list | table
    text: str
    page: int  # 1-based
    level: int = 0  # heading level (1 = top)


@dataclass
class ExtractResult:
    blocks: list[Block]
    page_count: int
    title: str
    extractor: str
    char_count: int
    warnings: list[str] = field(default_factory=list)
    page_chars: list[int] = field(default_factory=list)


BULLET_RE = re.compile(r"^\s*([•●▪◦‣⁃∙\-\*–]|\(?\d{1,2}[\.\)]|\(?[a-zA-Z][\.\)])\s+")
NUMBERED_HEADING_RE = re.compile(r"^(\d+(\.\d+){0,3}|[A-Z]\.|Chapter\s+\d+|Section\s+\d+|Appendix\s+[A-Z0-9]+)[\.:)]?\s+\S", re.I)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphen
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    # bullet glyphs that fonts often map to odd code points (or U+FFFD) -> a plain bullet
    text = re.sub(r"^[�▪●‣⁃∙·]\s*", "• ", text)
    text = re.sub(r"[ \t ]+", " ", text)
    return text.strip()


def _norm_repeat_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


# --------------------------------------------------------------------------
# PyMuPDF extractor
# --------------------------------------------------------------------------

def _table_to_markdown(rows: list[list]) -> str:
    rows = [[_clean(str(c or "")).replace("\n", " ") for c in r] for r in rows if r]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _extract_pymupdf(path: Path) -> ExtractResult:
    import pymupdf as fitz  # PyMuPDF

    try:
        doc = fitz.open(path)
    except Exception as e:  # noqa: BLE001
        raise IngestError(f"The file could not be opened as a PDF ({e}).") from e
    if doc.needs_pass:
        raise IngestError("The PDF is password-protected. Please provide an unlocked copy.")
    if doc.page_count == 0:
        raise IngestError("The PDF has no pages.")

    lines: list[dict] = []  # {text,size,bold,page,y0,y1,height}
    tables: list[tuple[int, float, str]] = []  # (page, y0, markdown)
    page_chars: list[int] = []
    warnings: list[str] = []

    for pno in range(doc.page_count):
        page = doc[pno]
        ph = page.rect.height or 1
        table_boxes = []
        try:
            # The table finder's default strategy needs ruling lines; skip pages without vector drawings (fast path).
            tables_found = page.find_tables().tables if len(page.get_drawings()) >= 3 else []
            for t in tables_found:
                md = _table_to_markdown(t.extract())
                if md and md.count("\n") >= 2:
                    tables.append((pno + 1, t.bbox[1] / ph, md))
                    table_boxes.append(fitz.Rect(t.bbox))
        except Exception:  # noqa: BLE001 - table finding is best-effort
            pass

        d = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_DEHYPHENATE)
        chars_on_page = 0
        for b in d.get("blocks", []):
            if b.get("type") != 0:
                continue
            for ln in b.get("lines", []):
                spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = _clean("".join(s["text"] for s in ln["spans"]))
                if not text:
                    continue
                bbox = fitz.Rect(ln["bbox"])
                if any(tb.intersects(bbox) and (tb & bbox).get_area() > 0.5 * bbox.get_area() for tb in table_boxes):
                    continue  # text already captured inside a table
                size = max(s["size"] for s in spans)
                bold_chars = sum(len(s["text"]) for s in spans if (s["flags"] & 16) or "bold" in s["font"].lower())
                total = sum(len(s["text"]) for s in spans) or 1
                lines.append(
                    {
                        "text": text,
                        "size": round(size, 1),
                        "bold": bold_chars / total > 0.6,
                        "page": pno + 1,
                        "y0": bbox.y0 / ph,
                        "y1": bbox.y1 / ph,
                        "x0": bbox.x0,
                    }
                )
                chars_on_page += len(text)
        page_chars.append(chars_on_page + sum(len(m) for p, _, m in tables if p == pno + 1))

    title = (doc.metadata or {}).get("title") or ""
    page_count = doc.page_count
    doc.close()
    blocks = _lines_to_blocks(lines, tables, page_count)
    return ExtractResult(blocks, page_count, title, "pymupdf", sum(page_chars), warnings, page_chars)


def _lines_to_blocks(lines: list[dict], tables: list[tuple[int, float, str]], page_count: int) -> list[Block]:
    if not lines and not tables:
        return []

    # 1) Remove running headers/footers: short lines in the top/bottom 8% repeated on many pages.
    if page_count >= 3:
        edge = [l for l in lines if l["y0"] < 0.08 or l["y1"] > 0.92]
        counts = Counter(_norm_repeat_key(l["text"]) for l in edge)
        repeated = {k for k, c in counts.items() if c >= max(3, int(page_count * 0.5))}
        lines = [l for l in lines if not ((l["y0"] < 0.08 or l["y1"] > 0.92) and _norm_repeat_key(l["text"]) in repeated)]
    # bare page numbers ("3", "Page 3 of 10")
    lines = [l for l in lines if not re.fullmatch(r"(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?", l["text"].lower())]

    # 2) Body font size = most common size weighted by characters.
    size_weight: Counter = Counter()
    for l in lines:
        size_weight[l["size"]] += len(l["text"])
    body = size_weight.most_common(1)[0][0] if size_weight else 10.0
    heading_sizes = sorted({l["size"] for l in lines if l["size"] >= body * 1.15}, reverse=True)

    def heading_level(l: dict) -> int:
        t = l["text"]
        if len(t) > 110 or len(t) < 2 or t.endswith((",", ";")):
            return 0
        if l["size"] >= body * 1.15 and not t.endswith("."):
            return min(heading_sizes.index(l["size"]) + 1, 4) if l["size"] in heading_sizes else 2
        if l["bold"] and len(t) <= 80 and not t.endswith(".") and not BULLET_RE.match(t):
            return 3 if not NUMBERED_HEADING_RE.match(t) else (2 if t.count(".") <= 1 else 3)
        return 0

    # 3) Merge consecutive lines into paragraphs / list items; split on headings and vertical gaps.
    events: list[tuple[int, float, Block]] = []
    cur: list[str] = []
    cur_kind = "para"
    cur_page = 1
    cur_y = 0.0
    prev = None

    def flush():
        nonlocal cur
        if cur:
            text = " ".join(cur)
            text = re.sub(r"(\w)- (\w)", r"\1\2", text) if cur_kind == "para" else text
            events.append((cur_page, cur_y, Block(cur_kind, text.strip(), cur_page)))
        cur = []

    for l in lines:
        lvl = heading_level(l)
        if lvl:
            flush()
            # merge two-line headings (same style, consecutive)
            if events and prev is not None and prev.get("_h") and prev["page"] == l["page"] and abs(prev["size"] - l["size"]) < 0.2 and l["y0"] - prev["y1"] < 0.02:
                events[-1][2].text += " " + l["text"]
            else:
                events.append((l["page"], l["y0"], Block("heading", l["text"], l["page"], lvl)))
            l["_h"] = True
            prev = l
            continue
        new_para = (
            prev is None
            or prev.get("_h")
            or l["page"] != prev["page"]
            or (l["y0"] - prev["y1"]) > 0.012  # visible vertical gap
            or BULLET_RE.match(l["text"])
        )
        # a page break in the middle of a sentence continues the paragraph
        if prev is not None and l["page"] != prev["page"] and cur and not cur[-1].rstrip().endswith((".", ":", "!", "?")) and not BULLET_RE.match(l["text"]):
            new_para = False
        if new_para:
            flush()
            cur_kind = "list" if BULLET_RE.match(l["text"]) else "para"
            cur_page, cur_y = l["page"], l["y0"]
        cur.append(l["text"])
        prev = l
    flush()

    for p, y, md in tables:
        events.append((p, y, Block("table", md, p)))
    events.sort(key=lambda e: (e[0], e[1]))
    return [e[2] for e in events if e[2].text]


# --------------------------------------------------------------------------
# Fallback extractors (plain text)
# --------------------------------------------------------------------------

def _text_pages_to_blocks(pages: list[str]) -> list[Block]:
    blocks: list[Block] = []
    for i, txt in enumerate(pages, start=1):
        for para in re.split(r"\n\s*\n", txt or ""):
            para_lines = [_clean(x) for x in para.splitlines() if _clean(x)]
            if not para_lines:
                continue
            buf: list[str] = []
            for ln in para_lines:
                is_heading = (
                    len(ln) <= 80
                    and not ln.endswith(".")
                    and (NUMBERED_HEADING_RE.match(ln) or (ln.isupper() and len(ln) > 3))
                )
                if is_heading:
                    if buf:
                        blocks.append(Block("para", " ".join(buf), i))
                        buf = []
                    blocks.append(Block("heading", ln, i, 2))
                elif BULLET_RE.match(ln):
                    if buf:
                        blocks.append(Block("para", " ".join(buf), i))
                        buf = []
                    blocks.append(Block("list", ln, i))
                else:
                    buf.append(ln)
            if buf:
                blocks.append(Block("para", " ".join(buf), i))
    return blocks


def _extract_pdfplumber(path: Path) -> ExtractResult:
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
        title = (pdf.metadata or {}).get("Title", "") or ""
    pc = [len(p) for p in pages]
    return ExtractResult(_text_pages_to_blocks(pages), len(pages), title, "pdfplumber", sum(pc), [], pc)


def _extract_pypdf(path: Path) -> ExtractResult:
    from pypdf import PdfReader

    r = PdfReader(str(path))
    if r.is_encrypted:
        raise IngestError("The PDF is password-protected. Please provide an unlocked copy.")
    pages = [(p.extract_text() or "") for p in r.pages]
    pc = [len(p) for p in pages]
    return ExtractResult(_text_pages_to_blocks(pages), len(pages), "", "pypdf", sum(pc), [], pc)


def _markdown_pages_to_blocks(pages: list[dict]) -> list[Block]:
    """Convert OCR Markdown (per page) into blocks: '#' headings, '|' tables, '-'/'1.' lists (indented lines continue
    the item), '**Title**' lines as headings, everything else paragraphs."""
    # Running banners (e.g. the document title printed on every page) must not become section headings.
    head_pages: Counter = Counter()
    min_level: dict[str, int] = {}
    for rec in pages:
        seen = set()
        for ln in (rec.get("text") or "").splitlines():
            m = re.match(r"^\s*(#{1,6})\s+(.*)", ln)
            if m:
                k = _norm_repeat_key(m.group(2).strip("* "))
                seen.add(k)
                min_level[k] = min(min_level.get(k, 9), len(m.group(1)))
        head_pages.update(seen)
    # A heading repeated on several pages is a banner if it is top-level (#/##) or long. Short repeated
    # sub-headings like "### Professors" are real structure and are kept.
    banner = {k for k, c in head_pages.items()
              if c >= max(3, len(pages) * 0.04) and (min_level[k] <= 2 or len(k.split()) >= 3)}

    blocks: list[Block] = []
    for rec in pages:
        n = rec["page"]
        lines = (rec.get("text") or "").splitlines()
        para: list[str] = []
        table: list[str] = []

        def flush_para():
            if para:
                blocks.append(Block("para", _clean(" ".join(para)), n))
                para.clear()

        def flush_table():
            if table:
                rows = [r for r in table if r.strip()]
                if len(rows) >= 2:
                    blocks.append(Block("table", "\n".join(rows), n))
                else:
                    blocks.append(Block("para", _clean(" ".join(rows).strip("| ")), n))
                table.clear()

        for raw in lines:
            line = raw.rstrip()
            s = line.strip()
            if s.startswith("|"):
                flush_para()
                table.append(s)
                continue
            flush_table()
            if not s:
                flush_para()
                continue
            m = re.match(r"^(#{1,6})\s+(.*)", s)
            bold = re.fullmatch(r"\*\*([^*]{2,100})\*\*:?", s)
            if m or bold:
                flush_para()
                text = (m.group(2) if m else bold.group(1)).strip("* ")
                level = min(len(m.group(1)), 4) if m else 3
                key = _norm_repeat_key(text)
                if key in banner or (len(key) >= 6 and any(key in b for b in banner)):
                    continue  # banner, or a fragment of one (e.g. "NED UNIVERSITY")
                if text:
                    blocks.append(Block("heading", _clean(text), n, level))
                continue
            if BULLET_RE.match(s):
                flush_para()
                blocks.append(Block("list", _clean(s.replace("**", "")), n))
                continue
            if raw[:1] in (" ", "\t") and blocks and blocks[-1].kind == "list" and blocks[-1].page == n and not para:
                blocks[-1].text += " " + _clean(s.replace("**", ""))  # continuation of a list item
                continue
            para.append(s.replace("**", ""))
        flush_table()
        flush_para()
    return [b for b in blocks if b.text]


def _needs_ocr(r: ExtractResult) -> bool:
    if not r.page_count:
        return False
    empty = sum(1 for c in r.page_chars if c < 50)
    return empty / r.page_count > 0.5


def extract(path: Path, max_pages: int = 800, progress=None) -> ExtractResult:
    """Extract structured blocks; scanned PDFs are OCR'd with Gemini vision when a key is configured."""
    r = _extract_text_layer(path, max_pages)
    if _needs_ocr(r):
        from . import config
        from .ocr import ocr_pdf

        if not config.GEMINI_API_KEY:
            raise IngestError(
                "This PDF is scanned (no text layer) and needs OCR. Set GEMINI_API_KEY in .env to enable Gemini OCR."
            )
        pages = ocr_pdf(Path(path), file_sha256(Path(path))[:16], progress=progress)
        failed = [p["page"] for p in pages if p["source"] == "failed"]
        blocks = _markdown_pages_to_blocks(pages)
        page_chars = [len(p.get("text") or "") for p in pages]
        warnings = ["Scanned PDF: text was extracted with Gemini OCR (verbatim transcription of each page image)."]
        if failed:
            warnings.append(f"OCR failed on {len(failed)} page(s): {', '.join(map(str, failed[:20]))}. Re-run ingestion to retry them.")
        printed = {str(p["page"]): p.get("printed_page") for p in pages if p.get("printed_page")}
        title = r.title if r.title and not r.title.lower().endswith(".pdf") else ""
        title = config.DOC_TITLE or title
        cover = [m.group(1).strip("* ") for ln in (pages[0].get("text") or "").splitlines()
                 if (m := re.match(r"^\s*#{1,3}\s+(.*)", ln))] if pages else []
        if not title and 1 <= len(cover) <= 4 and len(" ".join(cover)) <= 120:
            title = " ".join(cover)  # cover page headings, e.g. "NED" + "UNIVERSITY OF ..." + "PROSPECTUS 2025"
        if not title:  # first multi-word heading on the opening pages (raw OCR text, before banner filtering)
            for p in pages[:4]:
                for ln in (p.get("text") or "").splitlines():
                    m = re.match(r"^\s*#{1,3}\s+(.*)", ln)
                    if m and len(m.group(1).split()) >= 3:
                        title = m.group(1).strip("* ")
                        break
                if title:
                    break
            title = title or Path(path).stem
        res = ExtractResult(blocks, len(pages), title[:120], "gemini-ocr", sum(page_chars), warnings, page_chars)
        res.printed_pages = printed  # type: ignore[attr-defined]
        if res.char_count < 30:
            raise IngestError("OCR produced no text. The pages may be blank or unreadable.")
        return res
    return _finish_checks(r, path)


def _extract_text_layer(path: Path, max_pages: int = 800) -> ExtractResult:
    """Extract structured blocks, trying extractors in order and validating quality."""
    path = Path(path)
    with open(path, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise IngestError("This file is not a valid PDF (missing %PDF header).")

    errors: list[str] = []
    result: ExtractResult | None = None
    for fn in (_extract_pymupdf, _extract_pdfplumber, _extract_pypdf):
        try:
            r = fn(path)
        except IngestError:
            raise
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fn.__name__}: {e}")
            continue
        # Keep the first (structure-aware) extractor unless a fallback finds clearly more text.
        if result is None or r.char_count > result.char_count * 1.5:
            result = r
        if result.char_count >= 50 * result.page_count:
            break  # good enough; no need for fallbacks
    if result is None:
        raise IngestError("Text extraction failed with every extractor: " + "; ".join(errors))
    if result.page_count > max_pages:
        raise IngestError(f"The PDF has {result.page_count} pages; the limit is {max_pages}.")
    return result


def _finish_checks(result: ExtractResult, path: Path) -> ExtractResult:
    """Quality checks for text-layer PDFs."""
    path = Path(path)
    if result.char_count < 30:
        raise IngestError(
            "No extractable text was found. This looks like a scanned or image-only PDF. "
            "Please provide a text-based PDF (or run OCR first, e.g. `ocrmypdf in.pdf out.pdf`)."
        )
    empty_pages = [i + 1 for i, c in enumerate(result.page_chars) if c < 20]
    if result.page_count and len(empty_pages) / result.page_count > 0.3:
        result.warnings.append(
            f"{len(empty_pages)} of {result.page_count} pages contain little or no text (possibly scanned images): "
            f"pages {', '.join(map(str, empty_pages[:15]))}{'...' if len(empty_pages) > 15 else ''}. "
            "Answers cannot cover content on those pages."
        )
    all_text = " ".join(b.text for b in result.blocks)
    if all_text.count("(cid:") > 20 or (all_text and sum(ch == "�" for ch in all_text) / len(all_text) > 0.01):
        result.warnings.append("Some text could not be decoded (font encoding issue); a few passages may be garbled.")
    if not any(b.kind == "heading" for b in result.blocks):
        result.warnings.append("No headings were detected; chunks will be split by paragraphs only.")
    if not result.title:
        heads = [b for b in result.blocks if b.kind == "heading" and b.page == 1]
        result.title = heads[0].text if heads else path.stem
    result.title = result.title[:120]
    return result
