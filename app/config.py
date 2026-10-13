"""Central configuration. Every setting can be overridden from the environment or a .env file.

Secrets (API keys) are read here and never sent to the browser or written to logs.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)
# Plain HTTPS model downloads (the Xet transfer backend stalled on some networks during testing).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Never phone home for telemetry.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    v = _env(name, "1" if default else "0").lower()
    return v in {"1", "true", "yes", "on"}


DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
ACTIVE_FILE = DATA_DIR / "active.json"
for _d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

APP_NAME = _env("APP_NAME", "VeriDesk")
BOT_NAME = _env("BOT_NAME", "Vera")

# ---- Upload limits -------------------------------------------------------
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 40)
MAX_PAGES = _int("MAX_PAGES", 800)
MAX_QUESTION_CHARS = _int("MAX_QUESTION_CHARS", 2000)
MAX_HISTORY_TURNS = _int("MAX_HISTORY_TURNS", 6)
ALLOW_UPLOAD = _bool("ALLOW_UPLOAD", True)
UPLOAD_TOKEN = _env("UPLOAD_TOKEN")  # optional: required header for uploads on a public deployment

# ---- Chunking ------------------------------------------------------------
CHUNK_SIZE = _int("CHUNK_SIZE", 1000)  # characters (~200-250 tokens)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 150)  # characters carried over between chunks in the same section
MAX_TABLE_CHARS = _int("MAX_TABLE_CHARS", 2500)  # tables up to this size are kept whole
MIN_CHUNK_CHARS = _int("MIN_CHUNK_CHARS", 200)  # tiny sections are merged with their neighbour

# ---- Embeddings ----------------------------------------------------------
# local  = sentence-transformers on CPU (offline after the first model download)
# gemini = Gemini embedding API, openai = OpenAI text-embedding-3-small, none = BM25 only
EMBED_PROVIDER = _env("EMBED_PROVIDER", "local").lower()
LOCAL_EMBED_MODEL = _env("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# If the model is already cached, never contact the Hub: without this, each load makes "update check" requests
# that retry for ~45 s when the network is down (observed during testing).
_hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
if (_hf_cache / ("models--" + LOCAL_EMBED_MODEL.replace("/", "--")) / "snapshots").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
LOCAL_EMBED_QUERY_PREFIX = _env(
    "LOCAL_EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)
GEMINI_EMBED_MODEL = _env("GEMINI_EMBED_MODEL", "gemini-embedding-001")
OPENAI_EMBED_MODEL = _env("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# ---- Retrieval -----------------------------------------------------------
TOP_K = _int("TOP_K", 6)  # chunks given to the LLM
CANDIDATES = _int("CANDIDATES", 24)  # per-retriever candidates before fusion
RRF_K = _int("RRF_K", 60)
DENSE_WEIGHT = _float("DENSE_WEIGHT", 1.0)
BM25_WEIGHT = _float("BM25_WEIGHT", 1.0)
USE_RERANKER = _bool("USE_RERANKER", False)
RERANK_MODEL = _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Evidence gate (calibrated with scripts/calibrate.py; see docs/TECHNICAL_NOTES.md).
# Abstain without calling the LLM only when BOTH signals say the document is irrelevant.
GATE_DENSE_MIN = _float("GATE_DENSE_MIN", 0.52)
GATE_LEXICAL_MIN = _float("GATE_LEXICAL_MIN", 0.34)
# Context budget: drop chunks that are far weaker than the best one
RELATIVE_SCORE_CUTOFF = _float("RELATIVE_SCORE_CUTOFF", 0.80)

# ---- LLM -----------------------------------------------------------------
# Primary provider then comma-separated fallbacks. "extractive" never fails (no LLM, quotes the manual).
LLM_PROVIDER = _env("LLM_PROVIDER", "gemini").lower()
LLM_FALLBACKS = [p.strip().lower() for p in _env("LLM_FALLBACKS", "extractive").split(",") if p.strip()]
LLM_TIMEOUT = _float("LLM_TIMEOUT", 45)
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.1)
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 900)

GEMINI_API_KEY = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODEL = _env("GEMINI_FALLBACK_MODEL", "gemini-flash-latest")  # tried if the primary model fails
GEMINI_BASE_URL = _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = _env("ANTHROPIC_MODEL", "claude-sonnet-5-5")

OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL")  # optional, e.g. Groq or any OpenAI-compatible server

OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen2.5:1.5b")

HOST = _env("HOST", "127.0.0.1")
PORT = _int("PORT", 8000)


def chunk_settings() -> dict:
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_table_chars": MAX_TABLE_CHARS,
        "min_chunk_chars": MIN_CHUNK_CHARS,
        "version": 3,
    }
