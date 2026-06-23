#!/usr/bin/env python3
"""Recover the n=409 score after the judge ran out of OpenAI quota mid-scoring.

The agent answers are all saved (durable backup: medplum-eval/full409_answers.json). The first
scoring pass judged only the 200 cached-from-n=200 labels; the ~209 new questions erred to None
(quota). This re-judges ONLY the None/missing labels (reuses the 400 valid ones), so it costs <$1.

Run once OpenAI billing/quota is restored (or with a funded key):
  cd FHIR-AgentBench
  set -a; . ~/.codex/local-secrets/openai.env; set +a   # or export a funded OPENAI_API_KEY
  /Users/aanishsachdev/miniconda3/bin/python3 rejudge_409.py
"""
import json, os, sys, math

sys.argv = [sys.argv[0], "--input", "_unused"]
from evaluation_metrics import check_answer_correctness_with_llm

JUDGE = "gpt-5-mini-2025-08-07"
D = "runs/full409"
CACHE = f"{D}/judged_cache.json"
BACKUP = "medplum-eval/full409_answers.json"  # durable; survives runs/ clobbering

os.makedirs(D, exist_ok=True)
ans = json.load(open(BACKUP))                       # {'resource':{qid:{answer,true}}, 'code':{...}}
labels = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def judged(arm, qid, a, t):
    key = f"{arm}|{qid}"
    if labels.get(key) in (0, 1):                   # reuse only VALID labels; re-judge None/missing
        return labels[key]
    try:
        v = int(check_answer_correctness_with_llm(a or "", t or "", "", model=JUDGE))
    except Exception as e:
        print(f"  judge error on {key}: {type(e).__name__} {str(e)[:80]}", flush=True)
        v = None
    labels[key] = v
    json.dump(labels, open(CACHE, "w"))             # persist each -> resumable
    return v


ids = [q for q in ans["resource"] if q in ans["code"]]
need = sum(labels.get(f"resource|{q}") not in (0, 1) for q in ids) + \
       sum(labels.get(f"code|{q}") not in (0, 1) for q in ids)
print(f"{len(ids)} paired qids; {need} judge calls needed (rest already valid in cache)...", flush=True)

rj, cj = {}, {}
for i, q in enumerate(ids, 1):
    rj[q] = judged("resource", q, ans["resource"][q]["answer"], ans["resource"][q]["true"])
    cj[q] = judged("code", q, ans["code"][q]["answer"], ans["code"][q]["true"])
    if i % 50 == 0:
        print(f"  {i}/{len(ids)}", flush=True)

paired = [q for q in ids if rj[q] in (0, 1) and cj[q] in (0, 1)]
n = len(paired)
if n == 0:
    print("\nNO valid pairs — judge still failing (quota?). Cache preserved; just re-run when funded.")
    sys.exit(1)
ra = sum(rj[q] for q in paired)
ca = sum(cj[q] for q in paired)
b01 = sum(1 for q in paired if rj[q] == 0 and cj[q] == 1)   # code fixed
b10 = sum(1 for q in paired if rj[q] == 1 and cj[q] == 0)   # code broke
nd, k = b01 + b10, min(b01, b10)
p = min(1.0, 2.0 * sum(math.comb(nd, i) for i in range(k + 1)) / (2 ** nd)) if nd else 1.0

summary = {"n_paired": n, "n_total": len(ids),
           "resource_acc": round(ra / n, 4), "code_acc": round(ca / n, 4),
           "delta": round((ca - ra) / n, 4), "code_fixed": b01, "code_broke": b10,
           "mcnemar_p": round(p, 6), "significant": p < 0.05,
           "note": "cost (agent runs) = $34.38 from the original run; this re-judge is judge-only"}
json.dump(summary, open(f"{D}/_summary.json", "w"), indent=2)
open(f"{D}/SCORE_DONE", "w").write("done")
print("\n=== REAL n=409 RESULT ===")
print(f"  paired n = {n} (of {len(ids)})")
print(f"  resource (no code): {ra}/{n} = {ra/n:.1%}")
print(f"  code (+interp):     {ca}/{n} = {ca/n:.1%}   delta = {(ca-ra)/n:+.1%}")
print(f"  McNemar: fixed {b01}, broke {b10}, exact p = {p:.4g} {'SIGNIFICANT' if p<0.05 else 'n.s.'}")
print(json.dumps(summary, indent=2))
