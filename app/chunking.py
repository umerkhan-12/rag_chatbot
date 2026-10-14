"""Heading-aware chunking.

Rules (see docs/TECHNICAL_NOTES.md for the rationale):
  * A chunk never spans two sections, unless the first section is tiny (< MIN_CHUNK_CHARS),
    in which case it is merged forward so short sections are not orphaned.
  * Inside a section, blocks are packed up to CHUNK_SIZE characters. When a section needs several
    chunks, the next chunk starts with ~CHUNK_OVERLAP characters (whole sentences) of the previous one.
  * Tables up to MAX_TABLE_CHARS are atomic. Larger tables are split by rows, repeating the header row.
  * Every chunk keeps its section path, page range and a stable ID. The section path is prepended
    to the text that is embedded ("contextual chunk headers"), which helps retrieval of short chunks.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from . import config
from .guard import looks_like_instruction
from .ingest import Block

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    id: str
    index: int
    text: str
    embed_text: str
    section: str
    page_start: int
    page_end: int
    has_table: bool
    suspicious: bool  # contains instruction-like text aimed at AI systems (treated as untrusted)
    printed: str = ""  # page number printed on the paper page (scanned/OCR documents), if different

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def pages_label(self) -> str:
        label = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        return f"{label} (printed {self.printed})" if self.printed else label


@dataclass
class _Unit:
    text: str
    page: int
    atomic: bool = False  # tables
    is_table: bool = False


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT.split(text) if p.strip()]
    return parts or [text]


def _split_long_text(text: str, size: int) -> list[str]:
    """Split a long paragraph on sentence boundaries; hard-wrap sentences that are still too long."""
    out, buf = [], ""
    for s in _split_sentences(text):
        while len(s) > size:
            cut = s.rfind(" ", 0, size)
            cut = cut if cut > size * 0.5 else size
            if buf:
                out.append(buf)
                buf = ""
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if len(buf) + len(s) + 1 > size and buf:
            out.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        out.append(buf)
    return out


def _split_table(md: str, max_chars: int) -> list[str]:
    lines = md.splitlines()
    header, rows = lines[:2], lines[2:]
    parts, buf = [], list(header)
    for r in rows:
        if sum(len(x) + 1 for x in buf) + len(r) > max_chars and len(buf) > 2:
            parts.append("\n".join(buf))
            buf = list(header)
        buf.append(r)
    if len(buf) > 2:
        parts.append("\n".join(buf))
    return parts


def _sections(blocks: list[Block]) -> list[tuple[str, list[Block]]]:
    heads = [b for b in blocks if b.kind == "heading"]
    # A single top-level heading is the document title: keep it as text, not as part of every section path.
    title_block = None
    if heads:
        top = min(b.level for b in heads)
        tops = [b for b in heads if b.level == top]
        if len(tops) == 1 and len(heads) > 2 and tops[0] is heads[0]:
            title_block = tops[0]
    stack: list[tuple[int, str]] = []
    out: list[tuple[str, list[Block]]] = []
    current: list[Block] = []
    current_path = ""
    for b in blocks:
        if b is title_block:
            current.append(Block("para", b.text, b.page))
            continue
        if b.kind == "heading":
            if current:
                out.append((current_path, current))
                current = []
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text))
            current_path = " > ".join(t for _, t in stack)
        else:
            current.append(b)
    if current:
        out.append((current_path, current))
    return out


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or "\n|" in text[-overlap * 2:]:
        return ""  # never carry partial tables
    tail = ""
    for s in reversed(_split_sentences(text.replace("\n", " "))):
        if len(tail) + len(s) > overlap:
            break
        tail = f"{s} {tail}".strip()
    return tail


def chunk_blocks(blocks: list[Block], doc_title: str, doc_hash: str) -> list[Chunk]:
    size, overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    max_table, min_chunk = config.MAX_TABLE_CHARS, config.MIN_CHUNK_CHARS

    raw: list[dict] = []  # {section, units}

    for path, sec_blocks in _sections(blocks):
        units: list[_Unit] = []
        for b in sec_blocks:
            if b.kind == "table":
                parts = [b.text] if len(b.text) <= max_table else _split_table(b.text, max_table)
                units += [_Unit(p, b.page, atomic=True, is_table=True) for p in parts]
            elif len(b.text) > size:
                units += [_Unit(p, b.page) for p in _split_long_text(b.text, size)]
            else:
                units.append(_Unit(b.text, b.page))
        raw.append({"section": path, "units": units})

    # Pack units into chunks
    packed: list[dict] = []
    pending_small: dict | None = None
    for sec in raw:
        sec_chunks: list[dict] = []
        cur = {"sections": [sec["section"]], "parts": [], "pages": [], "table": False}
        if pending_small is not None:  # merge a tiny previous section forward
            cur = pending_small
            cur["sections"].append(sec["section"])
            cur["parts"].append(f"## {sec['section'].split(' > ')[-1]}" if sec["section"] else "")
            pending_small = None
        for u in sec["units"]:
            cur_len = sum(len(p) + 1 for p in cur["parts"])
            if cur["parts"] and cur_len + len(u.text) > size and cur_len > min_chunk * 0.5:
                sec_chunks.append(cur)
                tail = _overlap_tail("\n".join(cur["parts"]), overlap) if not u.atomic else ""
                cur = {"sections": [sec["section"]], "parts": [tail] if tail else [], "pages": [cur["pages"][-1]] if tail else [], "table": False}
            cur["parts"].append(u.text)
            cur["pages"].append(u.page)
            cur["table"] = cur["table"] or u.is_table
        if cur["parts"]:
            sec_chunks.append(cur)
        if sec_chunks:
            last = sec_chunks[-1]
            if sum(len(p) for p in last["parts"]) < min_chunk and len(sec_chunks) == 1:
                pending_small = sec_chunks.pop()
        packed += sec_chunks
    if pending_small is not None:
        if packed:  # attach to the previous chunk instead of leaving an orphan
            prev = packed[-1]
            prev["parts"] += [f"## {pending_small['sections'][0].split(' > ')[-1]}"] + pending_small["parts"]
            prev["pages"] += pending_small["pages"]
            prev["sections"] += pending_small["sections"]
        else:
            packed.append(pending_small)

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for c in packed:
        text = "\n".join(p for p in c["parts"] if p).strip()
        if not text or not c["pages"]:
            continue
        key = re.sub(r"\s+", " ", text.lower())
        if key in seen:  # exact duplicates (repeated boilerplate) are dropped
            continue
        seen.add(key)
        sections = [s for s in dict.fromkeys(c["sections"]) if s]
        section = " ; ".join(sections) if sections else "(no heading)"
        idx = len(chunks)
        cid = hashlib.sha1(f"{doc_hash}:{idx}:{key}".encode()).hexdigest()[:12]
        embed_text = f"{doc_title} | {section}\n{text}"
        chunks.append(
            Chunk(
                id=cid,
                index=idx,
                text=text,
                embed_text=embed_text,
                section=section,
                page_start=min(c["pages"]),
                page_end=max(c["pages"]),
                has_table=c["table"],
                suspicious=looks_like_instruction(text),
            )
        )
    return chunks
