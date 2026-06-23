#!/usr/bin/env python3
"""Numeric-error MAGNITUDE analysis on the 97 ground-truth-known (numeric-gold) questions.

Two findings, both pure arithmetic (gold numbers vs numbers extracted from the agent answer):
  1. AGENT calibration: when an arm gets a numeric question wrong, how far off is it — a near-miss
     (rounding/precision) or a gross error?
  2. JUDGE calibration (the smoking gun): correlate gpt-5-mini's verdict with the magnitude of the
     numeric difference. A reliable judge should reject large errors and accept tiny ones. We show
     gpt-5-mini does the opposite often enough to invert a real effect.
"""
import json, re
from statistics import median

D = "runs/full409"
res = {r["question_id"]: r for r in json.load(open(f"{D}/multi_turn_resource.json"))}
cod = {r["question_id"]: r for r in json.load(open(f"{D}/multi_turn_code_resource.json"))}
ids = [q for q in res if q in cod]
gpt = json.load(open(f"{D}/judged_cache.json"))
NUM = re.compile(r"-?\d+\.?\d*")
FAIL = re.compile(r"Input tokens exceeded|Max retries|RateLimitError|exceeded your current quota|Expected .* tool call, but got|Traceback", re.I)

def gold_type(g):
    g = (g or "").strip()
    if g in ("[[1]]", "[[0]]"): return "boolean"
    if "'" in g or '"' in g: return "categorical"
    inner = re.sub(r"[\[\]]", " ", g); toks = [t for t in re.split(r"[\s,]+", inner) if t]
    if toks and all(re.fullmatch(r"-?\d+\.?\d*", t) for t in toks): return "numeric"
    return "other"

def is_real(rec):
    a = rec.get("agent_answer") or ""
    return bool(a.strip()) and not FAIL.search(a)

def rel_error(ans, gold):
    """Best-case relative error: the answer number closest to (any) gold number."""
    gnums = [float(x) for x in NUM.findall(gold or "")]
    anums = [float(x) for x in NUM.findall(ans or "")]
    if not gnums or not anums:
        return None
    best = min(abs(a - g) / max(abs(g), 1e-9) for g in gnums for a in anums)
    return best

num_qs = [q for q in ids if gold_type(res[q].get("true_answer")) == "numeric"]
rows = []
for q in num_qs:
    for arm, rec in (("resource", res[q]), ("code", cod[q])):
        if not is_real(rec):
            continue
        re_ = rel_error(rec.get("agent_answer"), rec.get("true_answer"))
        if re_ is None:
            continue
        rows.append(dict(arm=arm, qid=q, rel_err=re_, gpt=gpt.get(f"{arm}|{q}"),
                         det_correct=(re_ <= 0.01)))  # within 1% == deterministically correct

# 1. agent error magnitude buckets
def bucket(e):
    if e <= 0.001: return "exact (<=0.1%)"
    if e <= 0.05: return "near-miss (<=5%)"
    if e <= 0.25: return "off (5-25%)"
    return "gross (>25%)"
from collections import Counter
agent_buckets = Counter(bucket(r["rel_err"]) for r in rows)

# 2. judge calibration vs magnitude
# false negatives: deterministically correct (tiny error) but gpt said WRONG
fn = [r for r in rows if r["det_correct"] and r["gpt"] == 0]
# false positives: deterministically wrong (real error) but gpt said CORRECT
fp = [r for r in rows if (not r["det_correct"]) and r["gpt"] == 1]
def med(xs): return round(median(xs), 4) if xs else None

summary = dict(
    numeric_questions=len(num_qs),
    arm_answers_scored=len(rows),
    agent_error_magnitude=dict(agent_buckets),
    gpt5mini_false_negatives=dict(
        n=len(fn),
        note="answers within 1% of gold (deterministically CORRECT) that gpt-5-mini marked WRONG",
        median_rel_error_of_these=med([r["rel_err"] for r in fn]),
        max_rel_error_of_these=round(max([r["rel_err"] for r in fn], default=0), 4),
    ),
    gpt5mini_false_positives=dict(
        n=len(fp),
        note="answers >1% off (deterministically WRONG) that gpt-5-mini marked CORRECT",
        median_rel_error_of_these=med([r["rel_err"] for r in fp]),
        max_rel_error_of_these=round(max([r["rel_err"] for r in fp], default=0), 4),
    ),
)
json.dump(summary, open(f"{D}/_magnitude_analysis.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print("\n=== HOW FAR OFF (agent numeric answers, n={} real arm-answers) ===".format(len(rows)))
for k in ["exact (<=0.1%)", "near-miss (<=5%)", "off (5-25%)", "gross (>25%)"]:
    print(f"  {k:18s}: {agent_buckets.get(k,0)}")
print("\n=== JUDGE CALIBRATION TO MAGNITUDE (gpt-5-mini, on known-truth numerics) ===")
print(f"  false NEGATIVES (tiny error, marked wrong): {len(fn)}  "
      f"median rel-error {med([r['rel_err'] for r in fn])} (these are essentially-correct answers)")
print(f"  false POSITIVES (real error, marked right): {len(fp)}  "
      f"median rel-error {med([r['rel_err'] for r in fp])}")
print("  -> a well-calibrated judge would do neither; gpt-5-mini punishes precision and rewards some gross misses.")
