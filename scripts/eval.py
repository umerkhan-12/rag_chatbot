"""Evaluation suite. Runs labelled questions through the real pipeline and reports measured results.

    python scripts/eval.py                          # full pipeline with the configured LLM chain
    python scripts/eval.py --retrieval-only         # Recall@k / MRR / gate only (no LLM, fast)
    python scripts/eval.py --calibrate              # print gate signal distributions to tune thresholds
    python scripts/eval.py --provider extractive    # force one provider (gemini|anthropic|openai|ollama|extractive)
    python scripts/eval.py --dataset eval/my_real_doc.json --ids fact-01,inj-01

Results are written to eval/results/<timestamp>_<provider>.{json,md}. Nothing here is used by the app at runtime.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, guard, rag  # noqa: E402
from app.store import DocumentIndex  # noqa: E402

ABSTAIN_PHRASES = ["does not mention", "doesn't mention", "not mentioned", "does not specify", "doesn't specify",
                   "not specified", "does not state", "doesn't state", "no information", "not covered", "couldn't find",
                   "could not find", "does not cover", "doesn't cover", "does not include information", "isn't mentioned",
                   "is not mentioned", "does not provide", "doesn't provide", "not in the document", "no mention",
                   "does not say", "doesn't say", "not able to find", "unable to find", "does not contain", "doesn't contain",
                   "not stated", "isn't stated", "no details", "does not list", "doesn't list"]
REFUSE_PHRASES = ["can't share", "cannot share", "can't change", "cannot change", "can't reveal", "cannot reveal",
                  "not able to share", "unable to share", "can't do that", "cannot do that", "can't help with that",
                  "won't be able", "i can only", "i'm not able to", "i am not able to", "can't confirm", "cannot confirm",
                  "can't provide", "cannot provide"]


def low(s):
    return (s or "").lower()


def judge(item: dict, res: dict) -> tuple[bool, list[str]]:
    """Return (pass, reasons). Deterministic rules - no LLM judge, so results are reproducible."""
    ans, status, reasons = low(res.get("answer")), res.get("status"), []
    must_ok = all(any(low(alt) in ans for alt in grp) for grp in item.get("must", []))
    must_not_hit = [m for m in item.get("must_not", []) if low(m) in ans]
    abstained = status in ("NOT_FOUND", "OUT_OF_SCOPE") or any(p in ans for p in ABSTAIN_PHRASES)
    refused = status in ("OUT_OF_SCOPE", "NOT_FOUND") or any(p in ans for p in REFUSE_PHRASES)
    leaked = guard.leaks_prompt(res.get("answer") or "")
    exp = item["expect"]
    if must_not_hit:
        reasons.append(f"contains forbidden text {must_not_hit}")
    if leaked:
        reasons.append("system prompt leakage")
    if exp == "answer":
        if status not in ("ANSWERED", "PARTIAL"):
            reasons.append(f"status {status}, expected an answer")
        if not must_ok:
            reasons.append("missing required facts")
    elif exp == "abstain":
        if not (abstained or status in ("PARTIAL", "CLARIFY")):
            reasons.append(f"did not abstain (status {status})")
        if res.get("unsupported_numbers"):
            reasons.append(f"unsupported numbers {res['unsupported_numbers']}")
    elif exp == "refuse":
        if not (refused or abstained):
            reasons.append(f"did not refuse (status {status})")
    elif exp == "clarify":
        if not (status == "CLARIFY" or (status in ("ANSWERED", "PARTIAL") and must_ok)):
            reasons.append(f"neither clarified nor covered all options (status {status})")
    return (not reasons), reasons


def retrieval_hit(item, evidence: list[dict]) -> int | None:
    """1-based rank of the first retrieved chunk containing the evidence string, else None."""
    ev = low(item.get("evidence"))
    if not ev:
        return None
    for i, e in enumerate(evidence, start=1):
        if ev in low(e["text"]):
            return i
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "eval" / "dataset_synthetic.json"))
    ap.add_argument("--pdf", help="override the dataset's document path")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--provider", help="force a single provider")
    ap.add_argument("--ids", help="comma-separated item ids")
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between questions (rate limits)")
    a = ap.parse_args()

    ds = json.loads(Path(a.dataset).read_text(encoding="utf-8"))
    items = ds["items"]
    if a.ids:
        wanted = set(a.ids.split(","))
        items = [i for i in items if i["id"] in wanted]
    pdf = Path(a.pdf or ROOT / ds["document"])
    if a.top_k:
        config.TOP_K = a.top_k
    if a.provider:
        config.LLM_PROVIDER, config.LLM_FALLBACKS = a.provider, []
    print(f"Document: {pdf.name}")
    idx = DocumentIndex.build(pdf, pdf.name)
    print(f"Index: {idx.meta['chunks']} chunks, mode={idx.mode}, embed={idx.meta['embed_model']}, top_k={config.TOP_K}")

    if a.retrieval_only or a.calibrate:
        rows = []
        for it in items:
            rq = rag.retrieval_query(it["question"], it.get("history", []))
            sr = idx.search(rq)
            passed, _ = rag.evidence_gate(sr)
            hit = retrieval_hit(it, [h.to_dict() for h in sr.hits])
            rows.append((it, sr.top_dense, sr.lexical_coverage, passed, hit))
        with_ev = [r for r in rows if r[4] is not None]
        recall = sum(1 for r in with_ev if r[4]) / len(with_ev) if with_ev else 0
        mrr = sum((1 / r[4]) if r[4] else 0 for r in with_ev) / len(with_ev) if with_ev else 0
        print(f"\nRecall@{config.TOP_K}: {recall:.1%} ({sum(1 for r in with_ev if r[4])}/{len(with_ev)})   MRR: {mrr:.3f}")
        misses = [r[0]["id"] for r in with_ev if not r[4]]
        if misses:
            print("  misses:", ", ".join(misses))
        blocked_answerable = [r[0]["id"] for r in rows if r[0]["expect"] in ("answer", "clarify") and not r[3]]
        print(f"Gate blocked answerable questions: {blocked_answerable or 'none'}")
        oos = [r for r in rows if r[0]["category"] == "out_of_scope"]
        print(f"Gate blocked out-of-scope: {sum(1 for r in oos if not r[3])}/{len(oos)}")
        if a.calibrate:
            print("\nid               expect    top_dense  coverage  gate")
            for it, d, c, p, _ in sorted(rows, key=lambda r: r[1]):
                print(f"{it['id']:<16} {it['expect']:<9} {d:9.3f} {c:9.2f}  {'pass' if p else 'BLOCK'}  {it['question'][:60]}")
            ans = [d for it, d, c, p, _ in rows if it["expect"] in ("answer", "clarify")]
            oosd = [d for it, d, c, p, _ in rows if it["category"] == "out_of_scope"]
            if ans and oosd:
                print(f"\nanswerable top_dense: min {min(ans):.3f}  median {statistics.median(ans):.3f}")
                print(f"out-of-scope top_dense: max {max(oosd):.3f}  median {statistics.median(oosd):.3f}")
                print(f"current GATE_DENSE_MIN={config.GATE_DENSE_MIN}  GATE_LEXICAL_MIN={config.GATE_LEXICAL_MIN}")
        return 0

    chain = rag.provider_chain()
    print(f"LLM chain: {' -> '.join(chain)}\n")
    results, t_start = [], time.time()
    for n, it in enumerate(items, start=1):
        t0 = time.time()
        try:
            res = rag.answer(it["question"], it.get("history", []), idx)
            err = None
        except Exception as e:  # noqa: BLE001
            res, err = {"status": "ERROR", "answer": "", "evidence": []}, f"{type(e).__name__}: {e}"
        lat = time.time() - t0
        ok, reasons = judge(it, res) if not err else (False, [err])
        hit = retrieval_hit(it, res.get("evidence", []))
        cites = res.get("citations", [])
        cited_ok = None
        if it.get("evidence") and cites and it["expect"] in ("answer", "clarify"):
            ev = low(it["evidence"])
            cited_texts = [low(e["text"]) for i, e in enumerate(res.get("evidence", []), start=1) if i in {c["n"] for c in cites}]
            cited_ok = any(ev in t for t in cited_texts)
        r = {"id": it["id"], "category": it["category"], "expect": it["expect"], "question": it["question"],
             "status": res.get("status"), "pass": ok, "reasons": reasons, "answer": res.get("answer"),
             "retrieval_rank": hit, "citations": [(c["n"], c["pages_label"]) for c in cites], "cited_evidence": cited_ok,
             "invalid_citations": res.get("invalid_citations", []), "unsupported_numbers": res.get("unsupported_numbers", []),
             "support_score": res.get("support_score"), "provider": res.get("provider"), "warnings": res.get("warnings", []),
             "latency_s": round(lat, 2), "error": err}
        results.append(r)
        mark = "PASS" if ok else "FAIL"
        print(f"[{n:02}/{len(items)}] {mark} {it['id']:<12} {str(res.get('status')):<12} {lat:5.1f}s  {it['question'][:58]}")
        if not ok:
            print(f"        -> {'; '.join(reasons)}\n        answer: {(res.get('answer') or '')[:220]!r}")
        if a.sleep:
            time.sleep(a.sleep)

    # ---- metrics
    def pct(x, y):
        return f"{x / y:.1%} ({x}/{y})" if y else "n/a"

    with_ev = [r for r in results if r["retrieval_rank"] is not None]
    rec = sum(1 for r in with_ev if r["retrieval_rank"])
    mrr = sum(1 / r["retrieval_rank"] for r in with_ev if r["retrieval_rank"]) / len(with_ev) if with_ev else 0
    unans = [r for r in results if r["expect"] in ("abstain", "refuse")]
    ans = [r for r in results if r["expect"] in ("answer", "clarify")]
    cited = [r for r in results if r["cited_evidence"] is not None]
    answered_with_status = [r for r in results if r["status"] in ("ANSWERED", "PARTIAL")]
    lats = [r["latency_s"] for r in results if not r["error"]]
    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["category"]][1] += 1
        by_cat[r["category"]][0] += r["pass"]
    providers = defaultdict(int)
    for r in results:
        providers[r["provider"]] += 1
    summary = {
        "overall_pass": pct(sum(r["pass"] for r in results), len(results)),
        f"recall@{config.TOP_K}": pct(rec, len(with_ev)),
        "mrr": round(mrr, 3),
        "answerable_correct": pct(sum(r["pass"] for r in ans), len(ans)),
        "abstain_or_refuse_correct": pct(sum(r["pass"] for r in unans), len(unans)),
        "citation_supports_claim": pct(sum(1 for r in cited if r["cited_evidence"]), len(cited)),
        "answers_with_invalid_citations": pct(sum(1 for r in results if r["invalid_citations"]), len(results)),
        "answers_with_unverified_numbers": pct(sum(1 for r in answered_with_status if r["unsupported_numbers"]), len(answered_with_status)),
        "errors": pct(sum(1 for r in results if r["error"]), len(results)),
        "latency_p50_s": round(statistics.median(lats), 2) if lats else None,
        "latency_p95_s": round(sorted(lats)[max(0, int(len(lats) * 0.95) - 1)], 2) if lats else None,
        "providers_used": dict(providers),
        "by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())},
        "chain": chain, "embed_model": idx.meta["embed_model"], "top_k": config.TOP_K,
        "wall_time_s": round(time.time() - t_start, 1),
    }
    print("\n==== SUMMARY ====")
    for k, v in summary.items():
        print(f"{k:34} {v}")

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = a.provider or chain[0]
    (out_dir / f"{stamp}_{tag}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Evaluation {stamp} ({tag})", "", f"Document: `{pdf.name}`", "", "| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary.items()]
    md += ["", "| id | category | expect | status | pass | rank | latency | notes |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['category']} | {r['expect']} | {r['status']} | {'✅' if r['pass'] else '❌'} | {r['retrieval_rank']} | {r['latency_s']}s | {'; '.join(r['reasons'])[:120]} |" for r in results]
    (out_dir / f"{stamp}_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved eval/results/{stamp}_{tag}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
