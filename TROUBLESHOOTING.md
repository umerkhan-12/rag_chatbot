# TROUBLESHOOTING

Fast fixes, most likely first. The app is built to **degrade instead of crash**: a failed LLM falls back to
extractive quoting, and failed embeddings fall back to keyword (BM25) retrieval. In both cases the UI shows a warning.

| Symptom | Likely cause | Fix |
|---|---|---|
| Page doesn't load / "Server unreachable" | server still starting (torch import takes 60–90 s), or crashed | Wait for `Application startup complete` in the terminal. If it crashed, read the last lines and run `python run.py` again. |
| Header pill says `extractive (no LLM)` | no `GEMINI_API_KEY` in `.env`, or `.env` not saved | Paste the key, save, **restart** `python run.py`. Check with `python scripts/check_setup.py`. |
| Answers say "Primary model unavailable; answered with fallback" | Gemini error: bad key, quota (429), network, timeout | `python scripts/check_setup.py` shows the exact error. 429 → wait 1 min or switch key. Network → phone hotspot. |
| `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` | old openai SDK with new httpx | `pip install -U "openai>=1.60"` |
| Gemini error mentioning `reasoning` | model rejects the reasoning setting | add `GEMINI_REASONING_EFFORT=` (empty) to `.env`, or try `GEMINI_MODEL=gemini-2.5-flash-lite` |
| Gemini 404 "model not found" | model name not available to the key | set `GEMINI_MODEL` to a listed model (AI Studio shows names) |
| Document card says "keyword (BM25) retrieval only" | embedding model failed to load (low RAM) | Close Chrome/VS Code, run `wsl --shutdown`, restart the app, **Remove document** and upload again. |
| Upload: "not a valid PDF" | file isn't a real PDF | Re-download the file; check it opens in a PDF viewer. |
| Upload: "scanned or image-only PDF" | no text layer | Ask organisers for a text PDF. Last resort: OCR it (`ocrmypdf`, not installed), or copy text into a PDF via Word → Save as PDF. |
| Upload: "password-protected" | encrypted PDF | Ask for an unlocked copy, or print to PDF. |
| Warning "N pages contain little or no text" | some scanned pages | Answers won't cover those pages. Mention it honestly in the demo. |
| Answers miss an obvious fact | retrieval or chunking issue for this PDF | Open the **Evidence inspector**: is the right passage retrieved? If not, try `TOP_K=8`; check chunks with `python scripts/ingest.py file.pdf --show`. |
| Bot abstains on a valid question ("couldn't find anything") | evidence gate too strict for this PDF | `python scripts/eval.py --calibrate` with your own questions; lower `GATE_DENSE_MIN` in `.env` (e.g. 0.45) and restart. |
| Port 8000 already in use | old server still running | close the old terminal, or set `PORT=8001` in `.env` |
| Very slow everything / fork errors | RAM exhausted (7 GB laptop) | Close apps, `wsl --shutdown`, don't run eval and the server at the same time. |
| Ollama fallback hangs | the local model is slow (~50 s) | Keep `LLM_FALLBACKS=extractive` (default). |
| Need a clean slate | corrupted index/state | Stop the server, delete the `data` folder, restart, re-upload. |

## Diagnostic commands
```bash
python scripts/check_setup.py                       # packages, keys (masked), live LLM ping, embeddings, active doc
python scripts/ingest.py "manual.pdf" --show        # see every chunk with page + section
python scripts/eval.py --retrieval-only --dataset eval/real_doc.json   # Recall@k without spending API calls
curl http://127.0.0.1:8000/api/status               # server state as JSON (no secrets in it)
```
