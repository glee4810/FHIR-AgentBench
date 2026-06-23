#!/usr/bin/env python3
"""Combine the corrected deterministic layer + the 3-Claude-judge panel into TRUSTWORTHY labels, then
run the stratified analysis honestly (multiple stratum definitions, CIs, not just a bare p-value), and
quantify how unreliable the original gpt-5-mini judge was — WITHOUT conflating it with the harness bug.

Inputs (all committed/derivable):
  runs/full409/det_labels.json        - deterministic: failures->0 (any gold) + clean NUMERIC golds (build_labels.py)
  runs/full409/panel_votes.json       - Claude panel for the original 188 categorical/'other' questions
  runs/full409/panel_votes_new.json   - Claude panel for the 111 BOOLEAN questions (the grading-bug fix); optional
  runs/full409/judged_cache.json      - the old gpt-5-mini labels (for the disagreement audit)
  runs/full409/multi_turn_{resource,code_resource}.json
"""
import json, re, math, os
from collections import defaultdict

D = "runs/full409"
res = {r["question_id"]: r for r in json.load(open(f"{D}/multi_turn_resource.json"))}
cod = {r["question_id"]: r for r in json.load(open(f"{D}/multi_turn_code_resource.json"))}
ids = [q for q in res if q in cod]
det = json.load(open(f"{D}/det_labels.json"))
gpt = json.load(open(f"{D}/judged_cache.json"))

# Claude panel majority per (qid, arm), ignoring -1, across BOTH the original and the boolean-fix panels
votes = defaultdict(list)
panel_files = [f"{D}/panel_votes.json"] + ([f"{D}/panel_votes_new.json"] if os.path.exists(f"{D}/panel_votes_new.json") else [])
for pf in panel_files:
    for batch in json.load(open(pf)):
        for judge in batch["votes"]:
            for row in judge:
                for arm in ("resource", "code"):
                    v = row.get(arm)
                    if v in (0, 1):
                        votes[f'{arm}|{row["qid"]}'].append(v)

def panel_label(key):
    vs = votes.get(key, [])
    if not vs:
        return None
    return 1 if sum(vs) * 2 > len(vs) else (0 if sum(vs) * 2 < len(vs) else None)

# FINAL trustworthy label: deterministic where present (failures + clean numerics), else panel majority
final, unresolved = {}, 0
for q in ids:
    for arm in ("resource", "code"):
        key = f"{arm}|{q}"
        if key in det:
            final[key] = det[key]
        else:
            lab = panel_label(key)
            if lab is None:
                final[key] = 0; unresolved += 1
            else:
                final[key] = lab

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
    delta = (ca - ra) / n if n else 0
    # Wald CI on the paired difference of proportions (McNemar): var = (b01+b10 - (b01-b10)^2/n)/n^2
    var = ((b01 + b10) - (b01 - b10) ** 2 / n) / n ** 2 if n else 0
    half = 1.96 * math.sqrt(var) if var > 0 else 0
    return dict(n=n, resource=ra, code=ca, resource_acc=round(ra/n, 4) if n else 0,
                code_acc=round(ca/n, 4) if n else 0, delta=round(delta, 4),
                ci95=[round(delta - half, 4), round(delta + half, 4)],
                fixed=b01, broke=b10, mcnemar_p=round(p, 6), significant=p < 0.05)

L = lambda arm, q: final[f"{arm}|{q}"]
pooled = mcnemar([(L("resource", q), L("code", q)) for q in ids])
# stratum A (conditioned on BOTH answering — post-treatment filter; conservative read of "reasoning")
both_real = [q for q in ids if real(res[q]) and real(cod[q])]
matched = mcnemar([(L("resource", q), L("code", q)) for q in both_real])
# stratum B (PREDEFINED by resource success only — the cleaner control the stats review asked for)
res_real = [q for q in ids if real(res[q])]
resource_real = mcnemar([(L("resource", q), L("code", q)) for q in res_real])
# stratum C (resource overflow — code routes payload to a sandbox; resource 0 by construction)
overflow_ids = [q for q in ids if overflowed(res[q])]
large = dict(n=len(overflow_ids), resource_acc=0.0,
             code_acc=round(sum(L("code", q) for q in overflow_ids)/len(overflow_ids), 4) if overflow_ids else 0,
             code_real=sum(1 for q in overflow_ids if real(cod[q])))

# how unreliable was gpt-5-mini? Separate TRUE judge error (on real answers) from the harness-bug rows
# (failure answers it should never have judged). Report both, headline the clean one.
def is_fail_key(arm, q):
    return not real({ "resource": res, "code": cod }[arm][q])
disagree_all = disagree_realonly = compared_all = compared_realonly = 0
for q in ids:
    for arm in ("resource", "code"):
        k = f"{arm}|{q}"
        g = gpt.get(k)
        if g not in (0, 1):
            continue
        compared_all += 1
        if final[k] != g:
            disagree_all += 1
        if not is_fail_key(arm, q):           # gpt judged a REAL answer -> genuine judge (un)reliability
            compared_realonly += 1
            if final[k] != g:
                disagree_realonly += 1

summary = dict(
    grading="deterministic(clean numerics + failures) + 3-Claude-judge panel majority (boolean/categorical/other)",
    pooled=pooled,
    matched_budget_BOTH_real=matched,
    resource_real_PREDEFINED=resource_real,
    large_record=large,
    panel_unresolved_ties=unresolved,
    gpt5mini_disagree_realanswers=dict(rate=round(disagree_realonly/compared_realonly, 4) if compared_realonly else 0,
                                       n=f"{disagree_realonly}/{compared_realonly}",
                                       note="genuine judge unreliability: gpt-5-mini vs trustworthy on answers that were REAL (not auto-failed)"),
    gpt5mini_disagree_allpairs=dict(rate=round(disagree_all/compared_all, 4) if compared_all else 0,
                                    n=f"{disagree_all}/{compared_all}",
                                    note="includes failure rows gpt mis-credited due to the harness bug (overstates pure judge error)"),
)
json.dump(summary, open(f"{D}/_trustworthy_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print("\n=== TRUSTWORTHY STRATIFIED RESULT (deterministic + Claude panel, boolean bug fixed) ===")
def line(tag, m):
    print(f"  {tag} (n={m['n']}): resource {m['resource_acc']:.1%} vs code {m['code_acc']:.1%}, "
          f"delta {m['delta']:+.1%} (95% CI {m['ci95'][0]:+.1%}..{m['ci95'][1]:+.1%}), "
          f"fixed {m['fixed']}/broke {m['broke']}, McNemar p={m['mcnemar_p']:.3g} -> "
          f"{'significant' if m['significant'] else 'no sig. difference'}")
line("matched budget (both answered)", matched)
line("resource-real (predefined)    ", resource_real)
print(f"  large records (resource overflows, n={large['n']}): resource 0% vs code {large['code_acc']:.1%}")
line("pooled                        ", pooled)
print(f"  >> gpt-5-mini judge error on REAL answers: {disagree_realonly}/{compared_realonly} = "
      f"{disagree_realonly/compared_realonly:.1%}  (all-pairs incl. harness-bug rows: {disagree_all}/{compared_all} = {disagree_all/compared_all:.1%})")
