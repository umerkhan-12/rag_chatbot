# VeriDesk — grounded RAG support assistant

Built for **The Engineer's Code 2026 — RAG Chatbot Builder** (SENTEC, NED UET).
Upload a product manual (PDF). VeriDesk answers customer questions **only from that document**, cites the page for
every claim, says so when the manual doesn't cover something, and stays composed under adversarial users.

## Quick start
```bash
pip install -r requirements.txt
copy .env.example .env          # then paste GEMINI_API_KEY=... (optional: works without a key in extractive mode)
python scripts/check_setup.py   # verifies packages, keys (masked), live model ping, embeddings
python run.py                   # http://127.0.0.1:8000
```

## Pipeline
PDF → PyMuPDF extraction (pages, headings from font sizes, tables as markdown; pdfplumber/pypdf fallbacks) →
heading-aware chunks (~1000 chars, 150 overlap, tables atomic, section path on every chunk) →
`BAAI/bge-small-en-v1.5` embeddings (local CPU) in **FAISS** + **BM25** → Reciprocal Rank Fusion top-6 →
evidence gate → **Gemini** (Claude/OpenAI/Ollama switchable) with numbered evidence and mandatory `[n]` citations →
validation (invalid citations removed, unverified numbers flagged, prompt-leak canary) → streamed UI with page
citations and an evidence inspector. Details: [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md).

## Repository map
| Path | What |
|---|---|
| `app/ingest.py` | PDF extraction + quality checks |
| `app/chunking.py` | structure-aware chunking |
| `app/embeddings.py`, `app/store.py` | embeddings, FAISS + BM25 index, hybrid search, persistence |
| `app/guard.py` | injection / hostility / extraction detection, document-injection flags, canary |
| `app/prompts.py` | system prompt and evidence formatting |
| `app/rag.py` | orchestration, evidence gate, validation, extractive fallback |
| `app/llm.py` | Gemini / Claude / OpenAI / Ollama streaming |
| `app/server.py`, `static/index.html` | FastAPI API + single-page UI |
| `scripts/eval.py` | evaluation suite (Recall@k, MRR, behaviour, citation validity, latency) |
| `scripts/make_test_pdfs.py` | **synthetic** rehearsal PDFs (not the competition document) |
| `eval/dataset_synthetic.json` | 53 labelled test questions for the synthetic manual |

## Team docs
[COMPETITION_PREFLIGHT.md](COMPETITION_PREFLIGHT.md) · [RUNBOOK.md](RUNBOOK.md) · [DEMO_SCRIPT.md](DEMO_SCRIPT.md) ·
[ADVERSARIAL_TESTS.md](ADVERSARIAL_TESTS.md) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
