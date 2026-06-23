#!/usr/bin/env python3
"""Canonical + STRATIFIED scorer for the 409 code-vs-resource run.

Fixes the harness bug found in adversarial review: score_full.py/rejudge_409.py called the LLM judge
directly, skipping the benchmark's own error pre-filter (evaluation_metrics.py:133-137 marks any
error / 'Input tokens exceeded' as WRONG before the judge). This replicates that pre-filter, then
reports the result the way the data actually warrants:

  - CANONICAL pooled accuracy (errors/overflow/empty = 0).
  - STRATIFIED table, which is the honest primary result:
      * matched-budget stratum  = questions where BOTH arms produced a real answer
        (the controlled comparison: does code help when the no-code agent can also operate?)
      * large-record stratum     = questions where the resource arm OVERFLOWED the 32k cap
        (the architecture effect: code routes data to a sandbox; resource can't fit it)

Run (needs a funded gpt-5-mini key to judge any newly re-run answers; reuses cached labels):
  set -a; . ~/.codex/local-secrets/openai.env; set +a
  /Users/aanishsachdev/miniconda3/bin/python3 rescore_canonical.py
"""
import json, os, re, sys, math

sys.argv = [sys.argv[0], "--input", "_unused"]
from evaluation_metrics import check_answer_correctness_with_llm

JUDGE = "gpt-5-mini-2025-08-07"
D = "runs/full409"
CACHE = f"{D}/judged_cache.json"
FAIL = re.compile(r"Input tokens exceeded|Max retries exceeded|RateLimitError|exceeded your current quota|Expected .* tool call, but got|Traceback \(most recent", re.I)
OVERFLOW = re.compile(r"Input tokens exceeded", re.I)

labels = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def load(arm):
    f = f"{D}/multi_turn_{arm}.json"
    return {r["question_id"]: r for r in json.load(open(f))}


def judged(arm, qid, ans, true):
    """LLM-judge label for a REAL answer, cached. Only called when the answer is not an error/empty."""
    key = f"{arm}|{qid}"
    if labels.get(key) in (0, 1):
        return labels[key]
    try:
        v = int(check_answer_correctness_with_llm(ans or "", true or "", "", model=JUDGE))
    except Exception as e:
        print(f"  judge error {key}: {type(e).__name__} {str(e)[:60]}", flush=True)
        v = None
    labels[key] = v
    json.dump(labels, open(CACHE, "w"))
    return v


def canon(arm, rec):
    """Benchmark-canonical correctness: error/overflow/empty -> 0; else the LLM judge label."""
    a = rec.get("agent_answer") or ""
    if not a.strip() or FAIL.search(a):
        return 0
    lab = judged(arm, rec["question_id"], a, rec.get("true_answer"))
    return 1 if lab == 1 else 0


def mcnemar(pairs):
    n = len(pairs)
    ra = sum(r for r, c in pairs); ca = sum(c for r, c in pairs)
    b01 = sum(1 for r, c in pairs if r == 0 and c == 1)
    b10 = sum(1 for r, c in pairs if r == 1 and c == 0)
    nd, k = b01 + b10, min(b01, b10)
    p = min(1.0, 2.0 * sum(math.comb(nd, i) for i in range(k + 1)) / (2 ** nd)) if nd else 1.0
    return {"n": n, "resource": ra, "code": ca, "resource_acc": round(ra / n, 4) if n else 0,
            "code_acc": round(ca / n, 4) if n else 0, "delta": round((ca - ra) / n, 4) if n else 0,
            "fixed": b01, "broke": b10, "mcnemar_p": round(p, 8), "significant": p < 0.05}


res, cod = load("resource"), load("code_resource")
ids = [q for q in res if q in cod]

rc = {q: canon("resource", res[q]) for q in ids}
cc = {q: canon("code", cod[q]) for q in ids}

def is_real(rec):
    a = rec.get("agent_answer") or ""
    return bool(a.strip()) and not FAIL.search(a)
def overflowed(rec):
    return bool(OVERFLOW.search(rec.get("agent_answer") or ""))

pooled = mcnemar([(rc[q], cc[q]) for q in ids])
matched = [q for q in ids if is_real(res[q]) and is_real(cod[q])]
matched_t = mcnemar([(rc[q], cc[q]) for q in matched])
largerec = [q for q in ids if overflowed(res[q]) and is_real(cod[q])]  # resource can't fit; code answered
lr = {"n": len(largerec), "resource_acc": 0.0,
      "code_acc": round(sum(cc[q] for q in largerec) / len(largerec), 4) if largerec else 0,
      "note": "resource = 0% by construction (overflowed 32k cap); code routes payload to sandbox"}

summary = {
    "scoring": "benchmark-canonical (error/overflow/empty -> 0, per evaluation_metrics.py:133-137), gpt-5-mini judge on real answers",
    "pooled_409": pooled,
    "matched_budget_stratum": matched_t,
    "large_record_stratum": lr,
    "code_arm_failures": {
        "answered_real": sum(1 for q in ids if is_real(cod[q])),
        "overflow": sum(1 for q in ids if overflowed(cod[q])),
        "empty_or_error": sum(1 for q in ids if not is_real(cod[q]) and not overflowed(cod[q])),
    },
    "resource_arm_overflow": sum(1 for q in ids if overflowed(res[q])),
    "model": "gpt-5.5-2026-04-23", "judge": JUDGE,
}
json.dump(summary, open(f"{D}/_canonical_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print("\n=== HONEST TWO-PART RESULT ===")
print(f"  matched budget (both arms answer, n={matched_t['n']}): "
      f"resource {matched_t['resource_acc']:.1%} vs code {matched_t['code_acc']:.1%}, "
      f"delta {matched_t['delta']:+.1%}, McNemar p={matched_t['mcnemar_p']:.3g} "
      f"-> {'NULL' if not matched_t['significant'] else 'SIGNIFICANT'}")
print(f"  large records (resource overflows, n={lr['n']}): "
      f"resource 0% (can't fit) vs code {lr['code_acc']:.1%} -> architecture/plumbing effect")
print(f"  pooled canonical (uninterpretable mix): resource {pooled['resource_acc']:.1%} vs "
      f"code {pooled['code_acc']:.1%}, delta {pooled['delta']:+.1%}")
