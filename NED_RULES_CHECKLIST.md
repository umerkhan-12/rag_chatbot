# NED briefing rules (from the slides, 14 Oct 2026) → how we comply

## ⚠ Ask the module lead first
Slide "Tools and Rules" lists **"pre-built solutions" as not allowed**. Our code was prepared before the build
window. **Get an explicit answer from the module lead before using it.** If the answer is no, write the code
fresh during the 4 hours (AI assistants are allowed if you can explain every choice) and treat this repo as study notes only, if notes are permitted.

## Allowed
| Rule | Our position |
|---|---|
| Any framework: LangChain, LlamaIndex **or plain Python** | Plain Python (FastAPI, FAISS, BM25, sentence-transformers) |
| Any vector store, embedding model or LLM API | FAISS, bge-small-en-v1.5, Gemini |
| AI coding assistants, **if you can explain every choice** | Every choice is explained in `docs/TECHNICAL_NOTES.md`. Each teammate must be able to explain chunking, hybrid search, the evidence gate and validation without notes |

## Not allowed
| Rule | Our position |
|---|---|
| Answers from the model's general knowledge | System prompt forbids it, the evidence gate blocks unrelated questions, and the vocabulary check flags words not in the PDF |
| Hardcoded answers | None in the app. Labelled questions exist only in `eval/` (for the **synthetic** test manual) and are never read by the app |
| Whole PDF pasted into every prompt | Only the top-6 retrieved chunks (~6k characters) are sent per question |
| Sharing code | Keep the GitHub repo **private**. Don't share with other teams |
| Pre-built solutions | **See the warning above** |
| Edits after submission | After emailing, **freeze**: `git tag submitted` and make no more commits |

## What the bot must do (slide "What Your Bot Should Do")
1. **Stay grounded**: every answer from the PDF, with page citations.
2. **Be polite**: keep character with rude or manipulative users.
3. **Admit gaps**: say clearly when the answer isn't in the document.

## Stress tests the judges will use (slide "Stress Test Your Bot")
1. Tricky questions that need careful reading → test confusable items (similar plans, limits, exceptions).
2. Out-of-scope questions → abstains politely.
3. Rude users and manipulation → **judges may ask new questions we haven't seen**. Don't rely on memorised demo questions.

## Pipeline points the organisers emphasised
* "Check headings and tables" when extracting → `scripts/ingest.py file.pdf --show` and inspect.
* "Decide chunk size and overlap, **and be ready to explain why**" → DEMO_SCRIPT Q1–Q3.
* "Retrieve… **test this on its own**" → `python scripts/eval.py --retrieval-only` (Recall@k, MRR).
* "Chat interface: simple is fine" → don't spend build time on UI polish.

## Deliverables (slide "Deliverables")
1. **Email the submission**: code, a requirements list (`requirements.txt`) and short run instructions (`README.md`).
2. **Present and demo**: **5–10 minutes, verbal, with a live demo**.
3. **Explain your choices**: what you chose, why you chose it, **and what you would improve** (DEMO_SCRIPT, "What we would improve").
