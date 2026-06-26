#!/usr/bin/env python3
"""Three-way verdict: A0 (raw FHIR) vs A0' (projection-only) vs A5 (code interpreter), on the overflow /
matched-answerable / pooled strata. Answers the decisive question: does projection-alone recover the code
arm's lift on the overflow stratum?

A0' is graded with the SAME trustworthy method: deterministic for numeric golds + failures; the non-numeric
(boolean/categorical) real answers are graded by the codex panel written to runs/a0prime/codex_votes/ (run
codex_panel on the a0prime batches first). Where a non-numeric A0' answer has no panel label yet, it is left
out of the trustworthy number and counted in `a0prime_ungraded`.
"""
import json, re, glob
from collections import defaultdict

R = "runs/full409"
A0P_FILE = "runs/a0prime/multi_turn_projected_resource.json"  # scp'd from the mini

res = {r["question_id"]: r for r in json.load(open(f"{R}/multi_turn_resource.json"))}
cod = {r["question_id"]: r for r in json.load(open(f"{R}/multi_turn_code_resource.json"))}
a0p = {r["question_id"]: r for r in json.load(open(A0P_FILE))}
det = json.load(open(f"{R}/det_labels.json"))

# --- panel labels (Claude) for resource/code ---
pvotes = defaultdict(list)
for pf in (f"{R}/panel_votes.json", f"{R}/panel_votes_new.json"):
    for b in json.load(open(pf)):
        for j in b["votes"]:
            for row in j:
                for arm in ("resource", "code"):
                    if row.get(arm) in (0, 1):
                        pvotes[f'{arm}|{row["qid"]}'].append(row[arm])
def pmaj(k):
    v = pvotes.get(k, [])
    return None if not v else (1 if sum(v) * 2 > len(v) else (0 if sum(v) * 2 < len(v) else None))
def L(arm, q):
    k = f"{arm}|{q}"
    return det[k] if k in det else (pmaj(k) or 0)

# --- grade A0' (deterministic numeric/failures + its own codex panel for non-numeric) ---
NUM = re.compile(r"-?\d+\.?\d*")
FAIL = re.compile(r"Input tokens exceeded|Max retries|RateLimitError|exceeded your current quota|Expected .* tool call, but got|Traceback", re.I)
def gold_type(g):
    g = (g or "").strip()
    if g in ("[[1]]", "[[0]]"): return "boolean"
    if "'" in g or '"' in g: return "categorical"
    inner = re.sub(r"[\[\]]", " ", g); toks = [t for t in re.split(r"[\s,]+", inner) if t]
    return "numeric" if toks and all(re.fullmatch(r"-?\d+\.?\d*", t) for t in toks) else "other"
def is_fail(rec):
    a = rec.get("agent_answer") or ""
    return (not a.strip()) or bool(FAIL.search(a))
def numeric_ok(ans, gold):
    gs = [float(x) for x in NUM.findall(gold or "")]; as_ = [float(x) for x in NUM.findall(ans or "")]
    if not gs or not as_: return 0
    return int(all(any(abs(a - g) <= max(0.05, 0.01 * abs(g)) for a in as_) for g in gs))

# A0' codex panel (if present): runs/a0prime/codex_votes/*.json -> majority
a0p_panel = defaultdict(list)
for f in glob.glob("runs/a0prime/codex_votes/*.json"):
    try:
        for row in json.load(open(f)).get("grades", []):
            if row.get("label") in (0, 1): a0p_panel[row["qid"]].append(row["label"])
    except Exception: pass
def a0p_panel_label(q):
    v = a0p_panel.get(q, [])
    return None if not v else (1 if sum(v) * 2 > len(v) else 0)

ungraded = []
def La0p(q):
    rec = a0p.get(q)
    if rec is None: return None
    if is_fail(rec): return 0
    gt = gold_type(rec.get("true_answer"))
    if gt == "numeric": return numeric_ok(rec.get("agent_answer"), rec.get("true_answer"))
    lab = a0p_panel_label(q)
    if lab is None: ungraded.append(q); return None
    return lab

strata = json.load(open(f"{R}/_strata.json"))
def acc(fn, qs):
    vals = [fn(q) for q in qs]; vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (0, 0)

print(f"{'arm':<14}{'overflow(262)':>16}{'matched(147)':>16}{'pooled(409)':>16}")
for name, fn in [("A0 raw", lambda q: L("resource", q)), ("A5 code", lambda q: L("code", q)),
                 ("A0' projected", La0p)]:
    o, on = acc(fn, strata["overflow"]); m, mn = acc(fn, strata["matched"]); p, pn = acc(fn, strata["ids"])
    print(f"{name:<14}{o:>13.1%}({on}){m:>13.1%}({mn}){p:>13.1%}({pn})")

# A0' overflow behavior
ov = sum(1 for q in strata["overflow"] if a0p.get(q) and "Input tokens exceeded" in (a0p[q].get("agent_answer") or ""))
print(f"\nA0' still overflows on {ov}/{len(strata['overflow'])} of the overflow stratum")
print(f"A0' non-numeric answers awaiting panel label: {len(ungraded)}")
o_a0p = acc(La0p, strata["overflow"])[0]; o_a5 = acc(lambda q: L('code', q), strata["overflow"])[0]
print(f"\n>>> VERDICT: overflow-stratum recovery = A0' {o_a0p:.1%} vs code {o_a5:.1%} vs raw 0%")
print("   projection-alone recovers " + (f"{o_a0p/o_a5:.0%}" if o_a5 else "n/a") + " of the code arm's overflow-stratum accuracy")
