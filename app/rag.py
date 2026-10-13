"""The RAG pipeline: guard -> retrieve -> evidence gate -> generate -> validate.

`answer_stream()` yields events consumed by the API (Server-Sent Events) and by the evaluation suite:
  {"type": "meta",  ...}   retrieved evidence, gate decision, input flags
  {"type": "token", "text": ...}
  {"type": "final", ...}   validated answer, status, citations, warnings, timings
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator

import httpx

from . import config, guard, llm
from .prompts import build_system_prompt, build_user_turn
from .store import ACTIVE, DocumentIndex, SearchResult, content_terms, tokenize

log = logging.getLogger("veridesk.rag")

STATUSES = {"ANSWERED", "PARTIAL", "NOT_FOUND", "CLARIFY", "OUT_OF_SCOPE"}
CITE_RE = re.compile(r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]")
UNIT = r"(?:%|percent|gb|mb|tb|kb|days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|weeks?|months?|years?|users?|students?|learners?|seats?|members?|courses?|attempts?|files?|pages?|characters?|participants?|times?|am|pm)"
NUM_RE = re.compile(rf"(?<![\w\[])((?:[$€£]|rs\.?|pkr|usd)\s?)?(\d[\d,]*(?:\.\d+)?)(\s?{UNIT}\b)?", re.I)
FOLLOWUP_RE = re.compile(r"\b(it|its|that|this|those|these|they|them|same|there|above|previous|what about|how about|and the|else|more)\b", re.I)


def _ollama_up() -> bool:
    try:
        base = config.OLLAMA_BASE_URL.rsplit("/v1", 1)[0]
        return httpx.get(base + "/api/tags", timeout=1.5).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def provider_chain() -> list[str]:
    chain = []
    for p in [config.LLM_PROVIDER, *config.LLM_FALLBACKS, "extractive"]:
        if p not in chain and llm.provider_available(p):
            chain.append(p)
    return chain


def doc_topics(index: DocumentIndex, n: int = 5) -> list[str]:
    tops = []
    for s in index.meta.get("sections", []):
        head = s.split(" > ")[0].split(" ; ")[0].strip()
        head = re.sub(r"^\d+(\.\d+)*[.)]?\s*", "", head)
        if re.match(r"(?i)^(introduction|overview|contents|table of contents|preface|foreword|glossary|index|appendix\b|about this (manual|guide|document))", head):
            continue
        if head and head.lower() not in {t.lower() for t in tops} and len(head) < 60:
            tops.append(head)
    title = index.meta.get("title", "").lower()
    tops = [t for t in tops if t.lower() != title]
    return tops[:n]


# ---------------------------------------------------------------------- validation
def _normalize_num(s: str) -> str:
    return s.replace(",", "").rstrip(".")


def unsupported_numbers(answer: str, evidence_text: str, question: str) -> list[str]:
    """Numbers in the answer that do not appear anywhere in the evidence (or the user's own question)."""
    body = CITE_RE.sub(" ", answer)
    ev = _normalize_num(evidence_text.replace(",", ""))
    q = question.replace(",", "")
    out = []
    for m in NUM_RE.finditer(body):
        cur, num, unit = m.group(1), m.group(2), m.group(3)
        n = _normalize_num(num)
        if not n or (len(n) == 1 and not cur and not unit):
            continue  # ordinals / list numbers like "step 2"
        if re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", ev) or re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", q):
            continue
        out.append(m.group(0).strip())
    return list(dict.fromkeys(out))


def _support_ratio(sentence: str, cited_texts: list[str]) -> float:
    terms = set(content_terms(CITE_RE.sub(" ", sentence)))
    if not terms:
        return 1.0
    ev_terms = set()
    for t in cited_texts:
        ev_terms |= set(tokenize(t))
    return len(terms & ev_terms) / len(terms)


def validate_answer(answer: str, hits, question: str) -> dict:
    """Check citations, numbers and lexical support. Removes citations that point to non-existent passages."""
    n_hits = len(hits)
    invalid: list[int] = []

    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*[,;]\s*", m.group(1))]
        good = [x for x in nums if 1 <= x <= n_hits]
        invalid.extend(x for x in nums if not 1 <= x <= n_hits)
        return "".join(f"[{x}]" for x in dict.fromkeys(good))

    cleaned = CITE_RE.sub(fix, answer)
    used = sorted({int(x) for x in re.findall(r"\[(\d{1,2})\]", cleaned)})
    citations = []
    for n in used:
        c = hits[n - 1].chunk
        citations.append({"n": n, "chunk_id": c.id, "page_start": c.page_start, "page_end": c.page_end,
                          "pages_label": c.pages_label, "section": c.section, "excerpt": c.text[:700]})

    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if s.strip()]
    ratios, weak = [], []
    for s in sentences:
        refs = [int(x) for x in re.findall(r"\[(\d{1,2})\]", s)]
        if not refs:
            continue
        r = _support_ratio(s, [hits[i - 1].chunk.text for i in refs])
        ratios.append(r)
        if r < 0.34:
            weak.append(s.strip()[:160])
    evidence_text = "\n".join(h.chunk.text for h in hits)
    return {
        "answer": cleaned,
        "citations": citations,
        "invalid_citations": sorted(set(invalid)),
        "unsupported_numbers": unsupported_numbers(cleaned, evidence_text, question),
        "support_score": round(sum(ratios) / len(ratios), 3) if ratios else None,
        "weakly_supported": weak,
    }


# ---------------------------------------------------------------------- extractive fallback
def _table_sentences(text: str) -> list[str]:
    """Turn markdown table rows into 'Header: value' sentences so they can be quoted on their own."""
    lines = [l for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return []
    cells = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]  # noqa: E731
    head = cells(lines[0])
    out = []
    for row in lines[2:]:
        vals = cells(row)
        if vals and vals[0]:
            pairs = [f"{h}: {v}" for h, v in zip(head[1:], vals[1:]) if v]
            out.append(f"{head[0]} {vals[0]} - " + "; ".join(pairs) + ".")
    return out


def missing_words(question: str, missing_stems: list[str]) -> list[str]:
    stems = set(missing_stems)
    words = []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.'-]*", question):
        t = tokenize(w)
        if t and t[0] in stems and w.lower() not in {x.lower() for x in words}:
            words.append(w.strip(".'"))
    return words


def extractive_answer(question: str, hits, missing: list[str] | None = None, analysis: guard.InputAnalysis | None = None) -> str:
    """No-LLM fallback: quote the most relevant sentences, with citations. Cannot hallucinate."""
    q = set(content_terms(question))
    scored = []
    for n, h in enumerate(hits[:4], start=1):
        if h.chunk.suspicious:
            continue  # never quote instruction-like document text as an answer
        prose = "\n".join(l for l in h.chunk.text.splitlines() if not l.strip().startswith("|"))
        sentences = re.split(r"(?<=[.!?])\s+|\n+", prose) + _table_sentences(h.chunk.text)
        for s in sentences:
            s = s.strip(" -•")
            if len(s) < 25 or s.startswith("## "):
                continue
            overlap = len(q & set(tokenize(s)))
            if overlap:
                scored.append((overlap + (0.5 if n == 1 else 0) - 0.1 * n, n, s))
    if not scored:
        return "STATUS: NOT_FOUND\n\nI couldn't find a passage in the document that answers this question."
    scored.sort(key=lambda x: x[0], reverse=True)
    picked, seen = [], set()
    for _, n, s in scored:
        if s in seen:
            continue
        seen.add(s)
        picked.append(f"- {s} [{n}]")
        if len(picked) == 3:
            break
    lead = ""
    if analysis and (analysis.authority or analysis.fabrication_pressure):
        lead = "I can't confirm anything the document doesn't state. "
    if missing:
        quoted = ", ".join(f'"{w}"' for w in missing[:4])
        return (f"STATUS: PARTIAL\n\n{lead}The document doesn't mention {quoted}, so I can't answer that part. "
                "The closest passages say (quoted directly, offline mode):\n" + "\n".join(picked))
    return f"STATUS: ANSWERED\n\n{lead}Here is what the document says (quoted directly, offline mode):\n" + "\n".join(picked)


# ---------------------------------------------------------------------- canned replies
def _canned(kind: str, index: DocumentIndex | None, analysis: guard.InputAnalysis) -> tuple[str, str]:
    name = config.BOT_NAME
    doc = (index.meta.get("title") or index.meta.get("display_name")) if index else None
    topics = doc_topics(index) if index else []
    topic_line = f" For example, you can ask me about {', '.join(topics[:4])}." if topics else ""
    if kind == "no_doc":
        return "NOT_FOUND", f"Hi, I'm {name}. No document is loaded yet. Please upload the product manual (PDF) and I'll answer questions about it."
    if kind == "greeting":
        return "ANSWERED", f"Hello! I'm {name}, your support assistant for **{doc}**. How can I help you today?{topic_line}"
    if kind == "thanks":
        return "ANSWERED", "You're welcome! Is there anything else I can help you with?"
    if kind == "identity":
        return "ANSWERED", (f"I'm {name}, a support assistant that answers questions using only **{doc}**. "
                            f"Every answer cites the page it came from, and if the document doesn't cover something, I'll tell you rather than guess.{topic_line}")
    if kind == "extraction":
        lead = "I understand you'd like to change how I work, but " if analysis.authority or analysis.injection else ""
        return "OUT_OF_SCOPE", (f"{lead}I can't share or change my internal instructions. I'm here to answer questions about **{doc}** "
                                f"using what the document actually says.{topic_line}")
    raise ValueError(kind)


def _gate_reply(index: DocumentIndex, analysis: guard.InputAnalysis) -> str:
    topics = doc_topics(index)
    lead = "I'm sorry this has been frustrating. " if analysis.hostile else ""
    doc = index.meta.get("title") or index.meta.get("display_name")
    t = f" I can help with topics such as {', '.join(topics[:4])}." if topics else ""
    return (f"{lead}I couldn't find anything about that in **{doc}**, so I can't answer it without guessing. "
            f"I only answer from this document.{t}")


# ---------------------------------------------------------------------- main entry point
def retrieval_query(question: str, history: list[dict]) -> str:
    last_user = next((h["content"] for h in reversed(history) if h.get("role") == "user"), "")
    if last_user and (len(question.split()) <= 7 or FOLLOWUP_RE.search(question)) and len(question) < 160:
        return f"{last_user}\n{question}"
    return question


def evidence_gate(sr: SearchResult) -> tuple[bool, str]:
    if not sr.hits:
        return False, "no passages matched"
    if sr.mode == "hybrid":
        if sr.top_dense < config.GATE_DENSE_MIN and sr.lexical_coverage < config.GATE_LEXICAL_MIN:
            return False, f"top similarity {sr.top_dense:.2f} < {config.GATE_DENSE_MIN} and keyword coverage {sr.lexical_coverage:.2f} < {config.GATE_LEXICAL_MIN}"
        return True, f"top similarity {sr.top_dense:.2f}, keyword coverage {sr.lexical_coverage:.2f}"
    if sr.lexical_coverage < 0.5:
        return False, f"keyword coverage {sr.lexical_coverage:.2f} < 0.50 (BM25-only mode)"
    return True, f"keyword coverage {sr.lexical_coverage:.2f} (BM25-only mode)"


def _final(status: str, answer: str, t0: float, **kw) -> dict:
    return {"type": "final", "status": status, "answer": answer, "citations": kw.pop("citations", []),
            "warnings": kw.pop("warnings", []), "latency_ms": round((time.time() - t0) * 1000), **kw}


def answer_stream(question: str, history: list[dict] | None = None, index: DocumentIndex | None = None) -> Iterator[dict]:
    t0 = time.time()
    history = [h for h in (history or []) if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str)]
    history = history[-config.MAX_HISTORY_TURNS * 2:]
    question = (question or "").strip()[: config.MAX_QUESTION_CHARS]
    index = index or ACTIVE.index
    analysis = guard.analyze_input(question)
    base_meta = {"type": "meta", "flags": analysis.flags, "evidence": [], "gate": None}

    if not question:
        yield base_meta
        yield _final("CLARIFY", "Please type a question about the document.", t0, provider="rules")
        return
    if index is None:
        yield base_meta
        s, a = _canned("no_doc", None, analysis)
        yield _final(s, a, t0, provider="rules")
        return
    if analysis.small_talk and not analysis.flags:
        yield base_meta
        s, a = _canned(analysis.small_talk, index, analysis)
        yield _final(s, a, t0, provider="rules")
        return
    if guard.is_pure_extraction(question, analysis):
        yield base_meta
        s, a = _canned("extraction", index, analysis)
        yield _final(s, a, t0, provider="rules", guard="prompt-extraction/injection blocked before retrieval")
        return

    rq = retrieval_query(question, history)
    sr = index.search(rq)
    passed, gate_reason = evidence_gate(sr)
    yield {**base_meta, "evidence": [h.to_dict() for h in sr.hits], "retrieval_query": rq,
           "gate": {"passed": passed, "reason": gate_reason, "top_dense": round(sr.top_dense, 4),
                    "lexical_coverage": round(sr.lexical_coverage, 3), "mode": sr.mode,
                    "missing_terms": missing_words(question, sr.missing_terms)}, "timings": sr.timings}

    if not passed:
        yield _final("NOT_FOUND", _gate_reply(index, analysis), t0, provider="rules",
                     guard=f"evidence gate: {gate_reason}")
        return

    missing = missing_words(question, sr.missing_terms)
    notes = list(analysis.notes)
    if missing:
        notes.append(
            "These words from the question appear nowhere in the document: " + ", ".join(missing) + ". They may be "
            "paraphrases of document terms; but if the evidence does not clearly cover them, say the document does not "
            "mention them instead of answering from general knowledge."
        )
    system = build_system_prompt(index.meta.get("title") or index.meta.get("display_name", "the document"))
    messages = []
    for h in history:
        content = re.sub(r"^STATUS:\s*\w+\s*", "", h["content"]).strip()[:1500]
        messages.append({"role": h["role"], "content": content or "(empty)"})
    if messages and messages[0]["role"] == "assistant":
        messages = messages[1:]
    messages.append({"role": "user", "content": build_user_turn(question, sr.hits, notes)})

    raw, used_provider, errors = "", None, []
    for provider in provider_chain():
        if provider == "ollama" and not _ollama_up():
            errors.append("ollama: server not running")
            continue
        if provider == "extractive":
            raw, used_provider = extractive_answer(question, sr.hits, missing, analysis), "extractive"
            yield {"type": "token", "text": re.sub(r"^STATUS:\s*\w+\s*", "", raw)}
            break
        try:
            stream = llm.stream_chat(provider, system, messages)
            buf, header_done, got_any = "", False, False
            for delta in stream:
                got_any = True
                raw += delta
                if not header_done:
                    buf += delta
                    if "\n" in buf or (len(buf) > 30 and not buf.lstrip().upper().startswith("STATUS")):
                        header_done = True
                        rest = re.sub(r"^\s*STATUS:\s*[A-Z_]+[ \t]*\n?", "", buf, flags=re.I).lstrip("\n")
                        if rest:
                            yield {"type": "token", "text": rest}
                    continue
                yield {"type": "token", "text": delta}
            if not got_any or not raw.strip():
                raise llm.LLMError("empty response")
            if not header_done:
                rest = re.sub(r"^\s*STATUS:\s*[A-Z_]+\s*", "", buf, flags=re.I)
                if rest:
                    yield {"type": "token", "text": rest}
            used_provider = provider
            break
        except llm.LLMError as e:
            log.warning("Provider %s failed: %s", provider, e)
            errors.append(f"{provider}: {e}")
            if raw:  # failed mid-stream: tell the UI to discard partial text
                yield {"type": "reset"}
            raw = ""
            continue

    m = re.match(r"^\s*STATUS:\s*([A-Z_]+)", raw, re.I)
    status = m.group(1).upper() if m and m.group(1).upper() in STATUSES else "ANSWERED"
    body = re.sub(r"^\s*STATUS:\s*[A-Z_]+\s*", "", raw, flags=re.I).strip()

    warnings = []
    if guard.leaks_prompt(body):
        body = "I can't share my internal instructions, but I'm happy to help with questions about the document."
        status = "OUT_OF_SCOPE"
        warnings.append("Output blocked: possible system-prompt leakage.")
    v = validate_answer(body, sr.hits, question + " " + " ".join(h["content"] for h in history if h["role"] == "user"))
    if v["invalid_citations"]:
        warnings.append(f"Removed citations to non-existent passages: {v['invalid_citations']}")
    if v["unsupported_numbers"] and status in ("ANSWERED", "PARTIAL"):
        warnings.append("Could not verify these figures in the cited document text: " + ", ".join(v["unsupported_numbers"]))
    if status in ("ANSWERED", "PARTIAL") and not v["citations"] and used_provider != "extractive":
        warnings.append("This answer has no citation to the document; treat it with caution.")
    if v["weakly_supported"]:
        warnings.append(f"{len(v['weakly_supported'])} sentence(s) are only weakly supported by their cited passage.")
    if errors and used_provider:
        warnings.append(f"Primary model unavailable; answered with fallback '{used_provider}'.")

    yield _final(status, v["answer"], t0, citations=v["citations"], warnings=warnings,
                 provider=used_provider, model=llm.model_name(used_provider or ""), provider_errors=errors,
                 support_score=v["support_score"], unsupported_numbers=v["unsupported_numbers"],
                 invalid_citations=v["invalid_citations"], weakly_supported=v["weakly_supported"],
                 gate=gate_reason, timings=sr.timings)


def answer(question: str, history: list[dict] | None = None, index: DocumentIndex | None = None) -> dict:
    """Non-streaming helper used by the CLI and evaluation suite."""
    meta, final = {}, {}
    for ev in answer_stream(question, history, index):
        if ev["type"] == "meta":
            meta = ev
        elif ev["type"] == "final":
            final = ev
    return {**final, "evidence": meta.get("evidence", []), "gate_info": meta.get("gate"), "flags": meta.get("flags", [])}


# ---------------------------------------------------------------------- sample questions
def sample_questions(index: DocumentIndex, use_llm: bool = True, n: int = 6) -> list[str]:
    """Suggested questions derived from the uploaded document (LLM-generated if available, else from headings)."""
    qs: list[str] = []
    chain = [p for p in provider_chain() if p not in ("extractive", "ollama")]
    if use_llm and chain:
        snippets = "\n\n".join(f"[{c.section}] {c.text[:300]}" for c in index.chunks[:: max(1, len(index.chunks) // 10)][:10])
        prompt = (f"Here are excerpts from a document titled '{index.meta.get('title')}':\n\n{snippets}\n\n"
                  f"Write {n + 2} short, natural questions a customer might ask that are answered by these excerpts. "
                  "One per line, no numbering, no extra text.")
        try:
            out = llm.complete(chain[0], "You write realistic customer support questions.", [{"role": "user", "content": prompt}])
            for line in out.splitlines():
                line = re.sub(r"^[\s\-*\d.)]+", "", line).strip()
                if 10 < len(line) < 120 and line.endswith("?"):
                    sr = index.search(line, top_k=3)
                    if evidence_gate(sr)[0]:
                        qs.append(line)
        except llm.LLMError as e:
            log.warning("Sample question generation failed: %s", e)
    if len(qs) < 3:
        for t in doc_topics(index, 8):
            qs.append(f"What does the document say about {t}?")
    return list(dict.fromkeys(qs))[:n]
