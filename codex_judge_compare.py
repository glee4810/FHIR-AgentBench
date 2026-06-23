#!/usr/bin/env python3
"""Aggregate the independent codex/GPT judge panel and triangulate it against the Claude panel and the
benchmark's gpt-5-mini judge. The point: show the trustworthy conclusion (matched-budget NULL + overflow
architecture effect) is INDEPENDENT of which judge model family you trust.

Inputs:
  runs/full409/codex_votes/p{1,2,3}_b{00..12}.json  - codex judge, 3 passes x 13 batches ({"grades":[...]})
  runs/full409/panel_votes.json                     - Claude judge panel
  runs/full409/det_labels.json                      - deterministic numeric + failure labels
  runs/full409/judged_cache.json                    - the old gpt-5-mini labels
  runs/full409/multi_turn_{resource,code_resource}.json
"""
import json, glob, re, math, os
from collections import defaultdict

D = "runs/full409"
res = {r["question_id"]: r for r in json.load(open(f"{D}/multi_turn_resource.json"))}
cod = {r["question_id"]: r for r in json.load(open(f"{D}/multi_turn_code_resource.json"))}
ids = [q for q in res if q in cod]
det = json.load(open(f"{D}/det_labels.json"))
gpt = json.load(open(f"{D}/judged_cache.json"))

# --- codex panel majority ---
codex_votes = defaultdict(list)
files = sorted(glob.glob(f"{D}/codex_votes/p*_b*.json")) + sorted(glob.glob(f"{D}/codex_votes_new/p*_b*.json"))
parsed = 0
for f in files:
    try:
        obj = json.load(open(f))
    except Exception:
        # tolerate a stray ```json fence or trailing text
        txt = open(f).read()
        m = re.search(r"\{.*\}", txt, re.S)
        obj = json.loads(m.group(0)) if m else {"grades": []}
    for row in obj.get("grades", []):
        for arm in ("resource", "code"):
            v = row.get(arm)
            if v in (0, 1):
                codex_votes[f'{arm}|{row["qid"]}'].append(v)
    parsed += 1

def majority(votes_list):
    if not votes_list:
        return None
    s = sum(votes_list)
    return 1 if s * 2 > len(votes_list) else (0 if s * 2 < len(votes_list) else None)

codex_label = {k: majority(v) for k, v in codex_votes.items()}

# --- Claude panel majority (original 188 + boolean-fix 111) ---
claude_votes = defaultdict(list)
import os as _os
for pf in [f"{D}/panel_votes.json"] + ([f"{D}/panel_votes_new.json"] if _os.path.exists(f"{D}/panel_votes_new.json") else []):
    for batch in json.load(open(pf)):
        for judge in batch["votes"]:
            for row in judge:
                for arm in ("resource", "code"):
                    v = row.get(arm)
                    if v in (0, 1):
                        claude_votes[f'{arm}|{row["qid"]}'].append(v)
claude_label = {k: majority(v) for k, v in claude_votes.items()}

# --- triangulation on the 188 non-numeric pairs both panels judged ---
shared = [k for k in claude_label if claude_label[k] is not None and codex_label.get(k) is not None]
agree = sum(1 for k in shared if claude_label[k] == codex_label[k])
# vs gpt-5-mini on the same shared keys
gpt_vs_claude = [(k, gpt.get(k), claude_label[k]) for k in shared if gpt.get(k) in (0, 1)]
gpt_vs_codex = [(k, gpt.get(k), codex_label[k]) for k in shared if gpt.get(k) in (0, 1)]
gpt_claude_agree = sum(1 for _, g, c in gpt_vs_claude if g == c)
gpt_codex_agree = sum(1 for _, g, c in gpt_vs_codex if g == c)

# --- recompute the stratified result using CODEX labels (det where present, else codex panel) ---
def final_codex(arm, q):
    k = f"{arm}|{q}"
    if k in det:
        return det[k]
    lab = codex_label.get(k)
    return 0 if lab is None else lab

FAIL = re.compile(r"Input tokens exceeded|Max retries|RateLimitError|exceeded your current quota|Expected .* tool call, but got|Traceback", re.I)
def real(rec):
    a = rec.get("agent_answer") or ""
    return bool(a.strip()) and not FAIL.search(a)
def overflowed(rec):
    return "Input tokens exceeded" in (rec.get("agent_answer") or "")

def mcnemar(pairs):
    n = len(pairs); ra = sum(r for r, c in pairs); ca = sum(c for r, c in pairs)
    b01 = sum(1 for r, c in pairs if r == 0 and c == 1)
    b10 = sum(1 for r, c in pairs if r == 1 and c == 0)
    nd, k = b01 + b10, min(b01, b10)
    p = min(1.0, 2.0 * sum(math.comb(nd, i) for i in range(k + 1)) / (2 ** nd)) if nd else 1.0
    return dict(n=n, resource_acc=round(ra/n, 4) if n else 0, code_acc=round(ca/n, 4) if n else 0,
                delta=round((ca-ra)/n, 4) if n else 0, fixed=b01, broke=b10,
                mcnemar_p=round(p, 6), significant=p < 0.05)

matched = [q for q in ids if real(res[q]) and real(cod[q])]
overflow = [q for q in ids if overflowed(res[q])]
mb = mcnemar([(final_codex("resource", q), final_codex("code", q)) for q in matched])
pooled = mcnemar([(final_codex("resource", q), final_codex("code", q)) for q in ids])
lr = dict(n=len(overflow), resource_acc=0.0,
          code_acc=round(sum(final_codex("code", q) for q in overflow)/len(overflow), 4) if overflow else 0)

summary = dict(
    codex_files_parsed=parsed,
    codex_panel_passes=3,
    triangulation=dict(
        shared_nonnumeric_labels=len(shared),
        claude_vs_codex_agreement=round(agree/len(shared), 4) if shared else 0,
        gpt5mini_vs_claude_agreement=round(gpt_claude_agree/len(gpt_vs_claude), 4) if gpt_vs_claude else 0,
        gpt5mini_vs_codex_agreement=round(gpt_codex_agree/len(gpt_vs_codex), 4) if gpt_vs_codex else 0,
    ),
    codex_graded_result=dict(matched_budget=mb, large_record=lr, pooled=pooled),
)
json.dump(summary, open(f"{D}/_codex_triangulation.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print("\n=== JUDGE TRIANGULATION (non-numeric labels) ===")
print(f"  Claude panel vs codex panel agree:   {agree}/{len(shared)} = {agree/len(shared):.1%}  (two different model families)")
print(f"  gpt-5-mini vs Claude panel agree:    {gpt_claude_agree}/{len(gpt_vs_claude)} = {gpt_claude_agree/len(gpt_vs_claude):.1%}")
print(f"  gpt-5-mini vs codex panel agree:     {gpt_codex_agree}/{len(gpt_vs_codex)} = {gpt_codex_agree/len(gpt_vs_codex):.1%}")
print("\n=== STRATIFIED RESULT USING CODEX (GPT) LABELS — robustness check ===")
print(f"  matched budget (n={mb['n']}): resource {mb['resource_acc']:.1%} vs code {mb['code_acc']:.1%}, "
      f"delta {mb['delta']:+.1%}, fixed {mb['fixed']}/broke {mb['broke']}, p={mb['mcnemar_p']:.3g} -> "
      f"{'SIGNIFICANT' if mb['significant'] else 'NULL'}")
print(f"  large records (n={lr['n']}): resource 0% vs code {lr['code_acc']:.1%}")
print(f"  pooled (n={pooled['n']}): resource {pooled['resource_acc']:.1%} vs code {pooled['code_acc']:.1%}, delta {pooled['delta']:+.1%}")
