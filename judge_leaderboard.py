#!/usr/bin/env python3
"""Judge-accuracy LEADERBOARD on the 97 numeric-gold questions, where we have NON-LLM ground truth
(deterministic tolerance match against the known gold). This answers: do the judge MODELS differ on
accuracy? — and validates the panels against truth (not just against each other).

For each judge (gpt-5-mini, codex/GPT panel, Claude panel) on every REAL numeric arm-answer:
  accuracy vs ground truth, plus the directional error split:
    false negative = ground-truth CORRECT but judge said WRONG  (punishing precision)
    false positive = ground-truth WRONG  but judge said CORRECT  (rewarding a miss)
"""
import json, re, glob, os
from collections import defaultdict

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
def numeric_truth(ans, gold):
    gnums = [float(x) for x in NUM.findall(gold or "")]
    anums = [float(x) for x in NUM.findall(ans or "")]
    if not gnums or not anums: return 0
    for gn in gnums:
        tol = max(0.05, 0.01 * abs(gn))
        if not any(abs(an - gn) <= tol for an in anums): return 0
    return 1

def panel_majority(files):
    votes = defaultdict(list)
    for f in files:
        try: obj = json.load(open(f))
        except Exception: continue
        rows = obj.get("grades") if isinstance(obj, dict) else None
        if rows is None and isinstance(obj, list):  # panel_votes_*.json shape: [{batch,votes:[[...]]}]
            for b in obj:
                for judge in b["votes"]:
                    for row in judge:
                        for arm in ("resource", "code"):
                            if row.get(arm) in (0, 1): votes[f'{arm}|{row["qid"]}'].append(row[arm])
            continue
        for row in rows or []:
            for arm in ("resource", "code"):
                if row.get(arm) in (0, 1): votes[f'{arm}|{row["qid"]}'].append(row[arm])
    out = {}
    for k, vs in votes.items():
        s = sum(vs); out[k] = 1 if s * 2 > len(vs) else (0 if s * 2 < len(vs) else None)
    return out

codex_num = panel_majority(sorted(glob.glob(f"{D}/codex_votes_numeric/p*_b*.json")))
claude_num = panel_majority([f"{D}/panel_votes_numeric.json"]) if os.path.exists(f"{D}/panel_votes_numeric.json") else {}

num_qs = [q for q in ids if gold_type(res[q].get("true_answer")) == "numeric"]
truth = {}
for q in num_qs:
    for arm, rec in (("resource", res[q]), ("code", cod[q])):
        if is_real(rec):
            truth[f"{arm}|{q}"] = numeric_truth(rec.get("agent_answer"), rec.get("true_answer"))

def score(judge_labels, name):
    keys = [k for k in truth if k in judge_labels and judge_labels[k] in (0, 1)]
    n = len(keys)
    correct = sum(1 for k in keys if judge_labels[k] == truth[k])
    fn = sum(1 for k in keys if truth[k] == 1 and judge_labels[k] == 0)
    fp = sum(1 for k in keys if truth[k] == 0 and judge_labels[k] == 1)
    return dict(judge=name, scored=n, accuracy=round(correct/n, 4) if n else None,
                false_neg_precision_punished=fn, false_pos_miss_rewarded=fp)

board = [score(gpt, "gpt-5-mini (benchmark default)"),
         score(codex_num, "codex/GPT panel (3-vote)")]
if claude_num:
    board.append(score(claude_num, "Claude panel (3-vote)"))
out = dict(ground_truth="deterministic numeric tolerance vs known gold (non-LLM)",
           numeric_questions=len(num_qs), leaderboard=board)
json.dump(out, open(f"{D}/_judge_leaderboard.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print("\n=== JUDGE ACCURACY vs NON-LLM GROUND TRUTH (numeric subset) ===")
for r in board:
    print(f"  {r['judge']:32s} acc {r['accuracy']:.1%} on {r['scored']}  "
          f"(precision-punished FN={r['false_neg_precision_punished']}, miss-rewarded FP={r['false_pos_miss_rewarded']})")
