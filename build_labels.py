#!/usr/bin/env python3
"""Rebuild the deterministic label layer CORRECTLY and emit the set of questions that must go to the
LLM judge panel.

Fixes the grading-integrity bug found in adversarial review: boolean golds `[[1]]`/`[[0]]` (the
benchmark's Yes/No encoding — see evaluation_metrics.py:176-178, "evaluate based on meaning, not
syntax") were being graded by the NUMERIC tolerance path, so free-text "Yes"/"No" answers (no digit to
extract) were auto-scored 0 and `final_grade.py` (deterministic-preferred) never let the panel correct
them. 99/230 boolean arm-labels were wrong.

Corrected deterministic policy — a label is deterministic ONLY when it is genuinely unambiguous:
  * FAILURE (empty / error / 'Input tokens exceeded' / quota / malformed tool call) -> 0, for ANY gold.
  * NUMERIC gold that is NOT 0/1 (counts >=2, decimals, multi-value) AND the answer contains a matching
    number within tolerance -> 1; numeric gold with a real answer but no matching number -> 0.
Everything else with a REAL answer -> NULL here (handed to the Claude + codex judge panels):
  * boolean 0/1 golds (ambiguous: Yes/No vs a count of 1 — the panel sees the QUESTION and decides),
  * categorical / string / list golds,
  * 'no answer'/null golds (semantic).

Outputs:
  runs/full409/det_labels.json        - the corrected deterministic labels (failures + clean numerics)
  runs/full409/panel_needed.json      - qids with >=1 real answer that need a panel label, by reason
  runs/full409/new_judge_batches/     - batches for questions NOT already covered by panel_votes.json
"""
import json, re, os
from collections import Counter, defaultdict

D = "runs/full409"
res = {r["question_id"]: r for r in json.load(open(f"{D}/multi_turn_resource.json"))}
cod = {r["question_id"]: r for r in json.load(open(f"{D}/multi_turn_code_resource.json"))}
ids = [q for q in res if q in cod]

FAIL = re.compile(r"Input tokens exceeded|Max retries|RateLimitError|exceeded your current quota|Expected .* tool call, but got|Traceback", re.I)
def is_failure(rec):
    a = rec.get("agent_answer") or ""
    return (not a.strip()) or bool(FAIL.search(a))

NUM = re.compile(r"-?\d+\.?\d*")
def gold_type(g):
    g = (g or "").strip()
    if g in ("[[1]]", "[[0]]", "[['1']]", "[['0']]", "[1]", "[0]"):
        return "boolean"
    if "'" in g or '"' in g:
        return "categorical"
    inner = re.sub(r"[\[\]]", " ", g)
    toks = [t for t in re.split(r"[\s,]+", inner) if t]
    if toks and all(re.fullmatch(r"-?\d+\.?\d*", t) for t in toks):
        return "numeric"
    return "other"

def gold_numbers(g):
    return [float(x) for x in NUM.findall(g or "")]

def numeric_match(ans, gold):
    """True if every gold number appears in the answer within tolerance.

    Known edge cases (audited: did NOT mislabel any committed answer, but fix before reuse):
      - NUM treats a hyphen as a minus sign, so an ISO date like '2142-05-10' parses to spurious
        negatives (-05, -10). Harmless here because numeric golds are never dates; gate on gold shape
        if you extend this to date golds.
      - a gold of exactly 0 with the floor tol=0.05 matches any literal '0' in a verbose answer. Fine
        for the count-0 golds in this set; tighten if 0-valued measurements are added.
    """
    gnums = gold_numbers(gold)
    anums = [float(x) for x in NUM.findall(ans or "")]
    if not gnums:
        return None
    for gn in gnums:
        tol = max(0.05, 0.01 * abs(gn))
        if not any(abs(an - gn) <= tol for an in anums):
            return False
    return True

det = {}
panel_needed = defaultdict(list)   # reason -> [qid]
type_counter = Counter()
for q in ids:
    gt = gold_type(res[q].get("true_answer"))
    type_counter[gt] += 1
    for arm, rec in (("resource", res[q]), ("code", cod[q])):
        key = f"{arm}|{q}"
        if is_failure(rec):
            det[key] = 0
            continue
        if gt == "numeric":
            m = numeric_match(rec.get("agent_answer"), rec.get("true_answer"))
            det[key] = 1 if m else 0      # clean, unambiguous numeric
        else:
            panel_needed[gt].append(q)    # boolean / categorical / other -> panel

json.dump(det, open(f"{D}/det_labels.json", "w"), indent=1)

# which panel-needed qids are NOT yet covered by the existing Claude panel_votes.json?
covered = set()
if os.path.exists(f"{D}/panel_votes.json"):
    for b in json.load(open(f"{D}/panel_votes.json")):
        for row in b["votes"][0]:
            covered.add(row["qid"])
need_qids = sorted({q for lst in panel_needed.values() for q in lst})
new_qids = [q for q in need_qids if q not in covered]

# build batches for the NEW questions (with both arms' real answers; failed arms marked)
def ans_for(rec):
    if is_failure(rec):
        a = rec.get("agent_answer") or ""
        if "Input tokens exceeded" in a:
            return "[FAILED: context overflow — no answer produced]"
        return "[FAILED: error/empty — no usable answer]"
    return (rec.get("agent_answer") or "")[:4000]

os.makedirs(f"{D}/new_judge_batches", exist_ok=True)
for f in os.listdir(f"{D}/new_judge_batches"):
    os.remove(f"{D}/new_judge_batches/{f}")
B = 15
batches = [new_qids[i:i+B] for i in range(0, len(new_qids), B)]
for bi, chunk in enumerate(batches):
    items = [dict(qid=q, question=res[q].get("question"), gold=res[q].get("true_answer"),
                  resource_answer=ans_for(res[q]), code_answer=ans_for(cod[q])) for q in chunk]
    json.dump(items, open(f"{D}/new_judge_batches/batch_{bi:02d}.json", "w"), indent=2)

print("gold types:", dict(type_counter))
print("deterministic labels:", len(det),
      "(failures + clean numerics);  numeric questions:", type_counter["numeric"])
print("panel-needed reasons:", {k: len(set(v)) for k, v in panel_needed.items()})
print("already covered by Claude panel_votes:", len(covered))
print("NEW questions needing judging:", len(new_qids), "-> batches:", len(batches))
