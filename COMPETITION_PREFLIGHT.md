# COMPETITION PREFLIGHT — VeriDesk (NED Engineer's Code 2026, RAG Chatbot Builder)

Finish this tonight. Every item marked ✅ was verified on the team laptop on 13–14 Oct 2026.
Items marked ⏳ are still pending at the time of writing.

## 1. What the project needs

| Component | Version tested | Purpose | Essential? |
|---|---|---|---|
| Windows 11 laptop, 7.2 GB RAM, Ryzen 5 7520U, no GPU | — | runs everything | ✅ essential |
| Python | 3.13.1 (global install) | backend | ✅ essential |
| fastapi / uvicorn / python-multipart | 0.121 / 0.38 / 0.0.28 | web server + upload | ✅ essential |
| pymupdf / pdfplumber / pypdf | 1.28 / 0.11 / 6.16 | PDF extraction (+ fallbacks) | ✅ essential |
| faiss-cpu / rank-bm25 / numpy | 1.15 / 0.2.2 / 2.3 | vector + keyword search | ✅ essential |
| sentence-transformers (+ torch 2.10 CPU, transformers 5.19) | 6.1.0 | local embeddings, offline | strongly recommended (falls back to BM25) |
| Embedding model `BAAI/bge-small-en-v1.5` | cached at `~/.cache/huggingface` (129 MB) | embeddings | ✅ already downloaded |
| openai SDK | 3.29.0 (**≥1.60 required**; 1.54 crashed with httpx 0.28) | Gemini (OpenAI-compatible endpoint), OpenAI, Ollama | essential for the LLM |
| anthropic SDK | 0.84 | Claude (optional alternative) | optional |
| Ollama + `qwen2.5:1.5b` | installed | offline LLM fallback (**~50 s/answer; not in the default chain**) | optional |
| Node.js | not needed | — | — |
| Browser (Edge/Chrome) | any | UI | ✅ essential |

No Node build, no database server, no Docker needed locally.

## 2. Install (only if starting on a fresh machine)
```bash
cd C:\Users\anas_\Desktop\ned_compt_rag
pip install -r requirements.txt
python scripts/check_setup.py
```
On the team laptop, everything is **already installed**. Don't reinstall tonight unless `check_setup.py` fails.
Disk has ~3.5 GB free, so avoid installing anything big.

## 3. API providers and environment variables
All settings live in `.env` (git-ignored). The template is `.env.example`.

| Variable | Needed? | Notes |
|---|---|---|
| `GEMINI_API_KEY` | **yes (primary LLM)** | from https://aistudio.google.com → Get API key |
| `GEMINI_MODEL` | default `gemini-2.5-flash` | change if the key's project lacks access |
| `LLM_PROVIDER` | default `gemini` | `anthropic`, `openai`, `ollama`, `extractive` also work |
| `LLM_FALLBACKS` | default `extractive` | `ollama,extractive` adds the slow local model |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | optional | only if you switch providers |
| `EMBED_PROVIDER` | default `local` | `gemini` uses API embeddings (no torch); `none` = BM25 only |

## 4. Verify each key without exposing it
```bash
python scripts/check_setup.py
```
It prints `set (39 chars)`, never the key, and makes one tiny live call per provider (`ping → pong`).
Expected: `[ OK ] gemini (gemini-2.5-flash): replied 'pong'`.

## 5. Account / billing tasks for tonight
- [ ] Create the Gemini key in Google AI Studio, paste it into `.env`, run `check_setup.py`.
- [ ] Free tier limits are per-minute and per-day and change over time. **Check your project's limits in AI Studio.**
      The evaluation suite sends ~50 requests. If you hit 429 errors, use `--sleep 4` or a second key.
- [ ] Optional: a second key (from a teammate's Google account) written down on paper as a spare. Never put it in chat or Git.
- [ ] Create an empty **private** GitHub repo and push (commands in RUNBOOK.md §6).

## 6. Optional vs essential
* **Essential:** Python packages above, the PDF, a browser. The app **starts and answers without any key** (extractive mode).
* **Strongly recommended:** Gemini key (natural answers, clarifying questions, false-premise handling).
* **Optional:** Ollama (offline LLM, slow), Claude/OpenAI keys, hosted deployment (Render/Railway via `Dockerfile`).

## 7. Offline vs online

| Feature | Offline? | Verified |
|---|---|---|
| Upload, extraction, chunking, indexing | ✅ yes | ✅ |
| Local embeddings (bge-small, cached) | ✅ yes | ✅ (model cached) |
| Hybrid retrieval, evidence gate, inspector | ✅ yes | ✅ |
| Extractive answers (no LLM) | ✅ yes | ✅ 88.7% on the synthetic eval |
| Ollama local LLM | ✅ yes | ✅ works, ~50 s per answer |
| Gemini / Claude / OpenAI answers | ❌ needs internet | ⏳ pending key |
| First-time embedding model download | ❌ needs internet | done already |

## 8. Startup (one command)
```bash
cd C:\Users\anas_\Desktop\ned_compt_rag
python run.py
```
The browser opens http://127.0.0.1:8000. The first start takes **~60–90 s** (torch import on this laptop). Wait for
`Application startup complete`. Preload a PDF at startup with `python run.py --ingest "path\to\manual.pdf"`.

## 9. Most likely failures → see TROUBLESHOOTING.md

## 10. Final checklist before leaving for NED
- [ ] `python scripts/check_setup.py` → `RESULT: READY`
- [ ] `python run.py` → UI loads → upload `samples/SYNTHETIC_LearnLoop_Manual.pdf` → ask 3 questions
- [ ] `python scripts/eval.py` with the Gemini key → note the overall pass rate in DEMO_SCRIPT.md
- [ ] Laptop charger, phone hotspot (plan for a backup connection), HDMI adapter
- [ ] Project pushed to GitHub (private) **and** copied to a USB drive
- [ ] Gemini key works on a phone hotspot too (venue Wi-Fi may block things)
- [ ] Everyone has read DEMO_SCRIPT.md and ADVERSARIAL_TESTS.md
- [ ] Laptop: Windows Update paused, sleep disabled while plugged in, `wsl --shutdown` before demo (frees RAM)
