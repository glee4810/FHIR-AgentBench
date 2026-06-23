#!/usr/bin/env python3
"""Resumable scorer for the n=200 code-vs-resource full run. Judges each answer with gpt-5-mini,
CACHING every label to disk immediately so a killed/restarted run resumes instead of re-judging.
Writes runs/full/_summary.json + a DONE marker when complete. Run detached:
  setsid nohup python3 score_full.py > runs/full/score.log 2>&1 < /dev/null &
"""
import json, os, sys, math

sys.argv = [sys.argv[0], "--input", "_unused"]
from evaluation_metrics import check_answer_correctness_with_llm

JUDGE = "gpt-5-mini-2025-08-07"
D = os.environ.get("RUN_DIR", "runs/full")
CACHE = f"{D}/judged_cache.json"
labels = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def judged(arm, qid, ans, true):
    key = f"{arm}|{qid}"
    if key in labels:
        return labels[key]
    try:
        v = int(check_answer_correctness_with_llm(ans or "", true or "", "", model=JUDGE))
    except Exception:
        v = None
    labels[key] = v
    json.dump(labels, open(CACHE, "w"))  # persist immediately -> resumable
    return v


res = {r["question_id"]: r for r in json.load(open(f"{D}/multi_turn_resource.json"))}
cod = {r["question_id"]: r for r in json.load(open(f"{D}/multi_turn_code_resource.json"))}
ids = [q for q in res if q in cod]
print(f"scoring {len(ids)} paired questions ({len(labels)} cached)...", flush=True)

rj, cj = {}, {}
for i, q in enumerate(ids, 1):
    rj[q] = judged("resource", q, res[q].get("agent_answer"), res[q].get("true_answer"))
    cj[q] = judged("code", q, cod[q].get("agent_answer"), cod[q].get("true_answer"))
    if i % 25 == 0:
        print(f"  {i}/{len(ids)}", flush=True)

paired = [q for q in ids if rj[q] in (0, 1) and cj[q] in (0, 1)]
n = len(paired)
ra = sum(rj[q] for q in paired)
ca = sum(cj[q] for q in paired)
b01 = sum(1 for q in paired if rj[q] == 0 and cj[q] == 1)  # code fixed
b10 = sum(1 for q in paired if rj[q] == 1 and cj[q] == 0)  # code broke
nd, k = b01 + b10, min(b01, b10)
p = min(1.0, 2.0 * sum(math.comb(nd, i) for i in range(k + 1)) / (2 ** nd)) if nd else 1.0
rc = sum((res[q].get("usage") or {}).get("cost", 0) or 0 for q in ids)
cc = sum((cod[q].get("usage") or {}).get("cost", 0) or 0 for q in ids)

summary = {"n_paired": n, "n_total": len(ids),
           "resource_acc": round(ra / n, 4), "code_acc": round(ca / n, 4),
           "delta": round((ca - ra) / n, 4), "code_fixed": b01, "code_broke": b10,
           "mcnemar_p": round(p, 6), "significant": p < 0.05,
           "cost_resource": round(rc, 2), "cost_code": round(cc, 2), "cost_total": round(rc + cc, 2)}
json.dump(summary, open(f"{D}/_summary.json", "w"), indent=2)
open(f"{D}/SCORE_DONE", "w").write("done")
print("\n=== RESULT ===")
print(f"  resource (no code): {ra}/{n} = {ra/n:.1%}")
print(f"  code (+interpreter): {ca}/{n} = {ca/n:.1%}  delta={(ca-ra)/n:+.1%}")
print(f"  McNemar: fixed {b01}, broke {b10}, p={p:.4g} {'SIGNIFICANT' if p<0.05 else 'n.s.'}")
print(f"  cost total ${rc+cc:.2f}")
