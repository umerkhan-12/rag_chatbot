"""FastAPI server: JSON/SSE API plus the static single-page UI. One process, one command."""
from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, llm, rag
from .ingest import IngestError
from .store import ACTIVE, DocumentIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("veridesk.server")

app = FastAPI(title=config.APP_NAME, docs_url=None, redoc_url=None)
STATIC = config.ROOT / "static"
_ingest_lock = threading.Lock()
_ingest_state = {"busy": False, "stage": "", "error": None}

# --- tiny in-memory rate limiter (per client IP) -----------------------------
_hits: dict[str, deque] = defaultdict(deque)


def _rate_limit(request: Request, limit: int = 40, window: int = 60) -> None:
    ip = request.client.host if request.client else "?"
    q, now = _hits[ip], time.time()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many requests - please wait a moment.")
    q.append(now)


@app.on_event("startup")
def _startup() -> None:
    ACTIVE.load_saved()
    chain = rag.provider_chain()
    log.info("LLM provider chain: %s", " -> ".join(f"{p}({llm.model_name(p)})" for p in chain))


def _doc_info() -> dict | None:
    idx = ACTIVE.index
    if idx is None:
        return None
    m = idx.meta
    return {
        "name": m.get("display_name"), "title": m.get("title"), "pages": m.get("pages"), "chunks": m.get("chunks"),
        "tables": m.get("tables"), "extractor": m.get("extractor"), "embed_model": m.get("embed_model"),
        "retrieval_mode": idx.mode, "warnings": m.get("warnings", []), "built_at": m.get("built_at"),
        "build_seconds": m.get("build_seconds"), "reused": m.get("reused", False),
        "suspicious_chunks": len(m.get("suspicious_chunks", [])), "samples": m.get("samples", []),
        "topics": rag.doc_topics(idx, 8), "doc_hash": m.get("doc_hash", "")[:12],
    }


@app.get("/api/status")
def status() -> dict:
    chain = rag.provider_chain()
    return {
        "app": config.APP_NAME, "bot": config.BOT_NAME, "document": _doc_info(),
        "llm_chain": [{"provider": p, "model": llm.model_name(p)} for p in chain],
        "ingest": _ingest_state, "allow_upload": config.ALLOW_UPLOAD, "upload_token_required": bool(config.UPLOAD_TOKEN),
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "settings": {**config.chunk_settings(), "top_k": config.TOP_K, "reranker": config.USE_RERANKER,
                     "gate_dense_min": config.GATE_DENSE_MIN, "gate_lexical_min": config.GATE_LEXICAL_MIN},
    }


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...), x_upload_token: str | None = Header(default=None)) -> JSONResponse:
    _rate_limit(request, limit=10)
    if not config.ALLOW_UPLOAD:
        raise HTTPException(403, "Uploads are disabled on this deployment.")
    if config.UPLOAD_TOKEN and x_upload_token != config.UPLOAD_TOKEN:
        raise HTTPException(403, "Upload token required.")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a .pdf file.")
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(409, "Another document is being processed. Please wait.")
    tmp_path = None
    try:
        _ingest_state.update(busy=True, stage="Receiving file", error=None)
        limit = config.MAX_UPLOAD_MB * 1024 * 1024
        size = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=config.UPLOAD_DIR) as tmp:
            tmp_path = Path(tmp.name)
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File too large (limit {config.MAX_UPLOAD_MB} MB).")
                tmp.write(chunk)
        if size == 0:
            raise HTTPException(400, "The uploaded file is empty.")
        display = Path(file.filename).name[:150]  # never used as a filesystem path

        def progress(stage: str) -> None:
            _ingest_state["stage"] = stage

        import anyio

        idx = await anyio.to_thread.run_sync(lambda: DocumentIndex.build(tmp_path, display, progress))
        _ingest_state["stage"] = "Preparing suggested questions"
        if not idx.meta.get("samples"):
            idx.meta["samples"] = await anyio.to_thread.run_sync(lambda: rag.sample_questions(idx))
            (idx.dir / "meta.json").write_text(json.dumps(idx.meta, ensure_ascii=False, indent=1), encoding="utf-8")
        ACTIVE.set(idx)
        return JSONResponse({"ok": True, "document": _doc_info()})
    except IngestError as e:
        _ingest_state["error"] = str(e)
        raise HTTPException(422, str(e)) from e
    except HTTPException as e:
        _ingest_state["error"] = e.detail
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("Ingestion failed")
        _ingest_state["error"] = "Unexpected error while processing the PDF."
        raise HTTPException(500, f"Unexpected error while processing the PDF: {type(e).__name__}") from e
    finally:
        _ingest_state.update(busy=False, stage="")
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        _ingest_lock.release()


@app.post("/api/reset")
def reset(request: Request, x_upload_token: str | None = Header(default=None)) -> dict:
    _rate_limit(request, limit=10)
    if config.UPLOAD_TOKEN and x_upload_token != config.UPLOAD_TOKEN:
        raise HTTPException(403, "Upload token required.")
    ACTIVE.clear()
    return {"ok": True}


class ChatTurn(BaseModel):
    role: str
    content: str = Field(max_length=6000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    _rate_limit(request)
    history = [t.model_dump() for t in req.history]

    def events():
        try:
            for ev in rag.answer_stream(req.message, history):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            log.exception("Chat failed")
            err = {"type": "error", "message": f"Something went wrong while answering ({type(e).__name__}). Please try again."}
            yield f"data: {json.dumps(err)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/source/{chunk_id}")
def source(chunk_id: str) -> dict:
    idx = ACTIVE.index
    if idx is None:
        raise HTTPException(404, "No document loaded")
    for c in idx.chunks:
        if c.id == chunk_id:
            d = c.to_dict()
            d.pop("embed_text", None)
            d["pages_label"] = c.pages_label
            return d
    raise HTTPException(404, "Unknown chunk")


@app.get("/api/pdf")
def pdf() -> FileResponse:
    idx = ACTIVE.index
    if idx is None or not (idx.dir / "source.pdf").exists():
        raise HTTPException(404, "No document loaded")
    return FileResponse(idx.dir / "source.pdf", media_type="application/pdf")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
