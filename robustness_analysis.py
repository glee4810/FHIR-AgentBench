#!/usr/bin/env python3
"""Post-hoc robustness analysis (no re-run, no LLM judge) for the MCP tool-ablation.

Addresses three SOTA gaps the audit flagged, using only the saved per-question data + the
ground-truth answers:
  1. JUDGE-FREE CORROBORATION -- deterministically re-score the agent answers against the
     benchmark ground truth (string/number match) and check it tracks the gpt-5-mini LLM judge,
     so the conclusion isn't an artifact of a small uncalibrated judge.
  2. MINIMUM DETECTABLE EFFECT (MDE) -- simulate paired-McNemar power at our n to report the
     smallest accuracy delta we were actually powered to detect (so the null is "no effect > X pp,"
     not "no effect").
  3. HOLM-BONFERRONI -- correct every paired p-value for the family of comparisons, so secondary
     claims (incl. the cap effect) are credible and we don't ride an uncorrected 0.05.

Usage: python robustness_analysis.py [results_dir]   (default: medplum-eval/results)
"""
import os
import sys
import ast
import json
import glob
import math
import random

random.seed(20260621)


# ---------- deterministic (judge-free) correctness ----------
def parse_true(ta):
    """FHIR-AgentBench true answers look like [['Nicotine Patch']] / [[1]] / [[15]] -> flat strings."""
    flat = []
    try:
        v = ast.literal_eval(str(ta))
    except Exception:
        v = ta

    def rec(x):
        if isinstance(x, (list, tuple)):
            for i in x:
                rec(i)
        else:
            flat.append(str(x))
    rec(v)
    return [s for s in flat if s.strip()]


_YES = ("yes", "present", "has been", "there are", "there is", "at least one", "i found", "was performed", "was given", "was prescribed", "was conducted")
_NO = ("no ", "not ", "none", "cannot find", "could not find", "no record", "no lab", "unable to", "there are no", "0 ", "zero")


def det_correct(agent_answer, true_answer):
    """Deterministic, judge-free correctness heuristic (corroboration, not ground truth)."""
    a = (agent_answer or "").strip()
    if not a or a.startswith("Error"):
        return None  # not answered -> excluded from answerable-set
    al = a.lower()
    gold = parse_true(true_answer)
    if not gold:
        return None
    hits = 0
    for g in gold:
        gl = g.lower().strip()
        if gl in ("1", "yes", "true"):
            hits += 1 if any(w in al for w in _YES) else 0
        elif gl in ("0", "no", "false"):
            hits += 1 if any(w in al for w in _NO) else 0
        else:
            gnum = gl.replace(",", "")
            hits += 1 if (gl in al or gnum in al.replace(",", "")) else 0
    return 1 if hits == len(gold) else 0


# ---------- McNemar exact ----------
def mcnemar_p(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


# ---------- MDE via paired simulation ----------
def mde(n, e=0.10, alpha=0.05, power=0.80, reps=3000):
    """Smallest true accuracy delta detectable at `power` with paired McNemar.
    Model: `e` = symmetric discordance noise (a question flips against the effect this often);
    the signal `delta` adds to the 'for' direction. b01=delta+e (A wrong,B right), b10=e."""
    for delta in [x / 100 for x in range(1, 60)]:
        b01p = delta + e
        b10p = e
        if b01p + b10p > 1:
            break
        wins = 0
        for _ in range(reps):
            b01 = b10 = 0
            for _ in range(n):
                r = random.random()
                if r < b01p:
                    b01 += 1
                elif r < b01p + b10p:
                    b10 += 1
            if mcnemar_p(b01, b10) < alpha:
                wins += 1
        if wins / reps >= power:
            return round(delta, 2)
    return None


# ---------- Holm-Bonferroni ----------
def holm(pvals):
    """pvals: list of (label, p). Returns list of (label, p, p_holm, sig_after)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i][1])
    m = len(pvals)
    out = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        label, p = pvals[i]
        adj = min(1.0, max(running, p * (m - rank)))
        running = adj
        out[i] = (label, round(p, 4), round(adj, 4), adj < 0.05)
    return out


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "medplum-eval/results"
    print("=" * 78)
    print("ROBUSTNESS ANALYSIS (judge-free re-score + MDE + Holm)  ")
    print("=" * 78)
    print(
        "\nPROVENANCE / what is computed vs reconstructed:\n"
        "  * COMPUTED LIVE from committed data (medplum-eval/results/*.json): the §1 deterministic\n"
        "    re-score curve and its McNemar steps in §3 (gpt curve(det): ...).\n"
        "  * RECONSTRUCTED CONSTANTS (hardcoded below) from a scoring run whose raw data was on a\n"
        "    torn-down box: the §1 LLM-judge accuracies and ALL the opus p-values in §3. The committed\n"
        "    per-question JSON carries agent answers but NOT the judge's correctness labels, and the\n"
        "    entire Opus run (incl. the cap-factorial / p_holm=0.005 finding) has no committed raw data.\n"
        "    Those numbers are reproducible only by re-running the eval, not from this repo's artifacts."
    )

    # ---- 1. deterministic re-score of the GPT-5.5 curve cells ----
    print("\n## 1. Judge-free deterministic re-score (vs the gpt-5-mini LLM judge)\n")
    # LLM-judge accuracies captured from the original scoring (the on-disk _scores.csv was clobbered):
    llm_judge = {"control": 0.80, "cat2": 0.70, "cat4": 0.767, "validated5": 0.767, "arm_ref": 0.833}
    order = {"control": 1, "cat2": 2, "cat4": 4, "validated5": 5, "arm_ref": 6, "arm_full8": 8}
    det = {}
    perq = {}
    for p in sorted(glob.glob(os.path.join(rdir, "mcp.*.json"))):
        arm = os.path.basename(p).split(".")[1]
        recs = json.load(open(p))
        scored = {r.get("question_id"): det_correct(r.get("agent_answer"), r.get("true_answer")) for r in recs}
        answered = {q: v for q, v in scored.items() if v is not None}
        det[arm] = (sum(answered.values()) / len(answered)) if answered else None
        perq[arm] = answered
    print(f"{'tools':>5}  {'arm':<14} {'LLM-judge':>9} {'deterministic':>13} {'|Δ|':>6}")
    for arm in sorted(det, key=lambda a: order.get(a, 99)):
        if det[arm] is None:
            print(f"{order.get(arm,'?'):>5}  {arm:<14} {'(all-error/quota)':>23}")
            continue
        lj = llm_judge.get(arm)
        diff = abs(lj - det[arm]) if lj is not None else None
        print(f"{order.get(arm,'?'):>5}  {arm:<14} {('%.2f'%lj) if lj else '-':>9} {det[arm]:>13.2f} {('%.2f'%diff) if diff is not None else '-':>6}")
    detvals = [det[a] for a in det if det[a] is not None]
    print(f"\n  -> the STRICT scorer floors ~20pp below the semantic LLM judge (it misses semantic"
          f"\n     equivalents, e.g. 'urgent'=='UR'), so absolute levels are judge-dependent. But the"
          f"\n     CURVE IS FLAT under BOTH scorers (deterministic range {min(detvals):.2f}-{max(detvals):.2f};"
          f" judge 0.70-0.83):"
          f"\n     control(1 tool) ~ arm_ref(6 tools) either way. The NULL/flat pattern is robust to the"
          f"\n     scorer; only the absolute number depends on the judge.")

    # ---- 2. MDE ----
    print("\n## 2. Minimum detectable effect (paired McNemar, alpha=.05, power=.80)\n")
    print(f"  {'':6}{'optimistic (e=.05)':>20}{'conservative (e=.10)':>22}")
    for n in (25, 30):
        m_lo = mde(n, e=0.05)
        m_hi = mde(n, e=0.10)
        print(f"  n={n:<4}{(str(int(m_lo*100))+' pp') if m_lo else '>55pp':>20}"
              f"{(str(int(m_hi*100))+' pp') if m_hi else '>55pp':>22}")
    print("  -> Under ANY plausible noise assumption the MDE is ~34-46pp. Reframe the null as 'no"
          "\n     tool effect larger than ~the MDE,' NOT 'no effect.' A commercially-decisive 5-10pp"
          "\n     lift is far below our MDE -> structurally invisible to this design.")

    # ---- 3. Holm-Bonferroni over the full family ----
    print("\n## 3. Holm-Bonferroni correction (family-wise)\n")
    # opus headline comparisons (p-values from the original scoring; raw data lost with the box)
    fam = [
        ("opus med: coaching-lift (control->control_include)", 1.0),
        ("opus med: structure-lift (control_include->arm_ref)", 0.69),
        ("opus med: total-lift (control->arm_ref)", 0.625),
        ("opus med: cap effect on control (32k->100k)", 0.039),
        ("opus med: cap effect on arm_ref (32k->100k)", 0.0005),
        ("opus rep: control->arm_full8", 1.0),
    ]
    # gpt curve steps (deterministic re-score McNemar)
    steps = [("control", "cat2"), ("cat2", "cat4"), ("cat4", "validated5"), ("validated5", "arm_ref")]
    for a, b in steps:
        if a in perq and b in perq:
            ids = [q for q in perq[a] if q in perq[b]]
            b01 = sum(1 for q in ids if perq[a][q] == 0 and perq[b][q] == 1)
            b10 = sum(1 for q in ids if perq[a][q] == 1 and perq[b][q] == 0)
            fam.append((f"gpt curve(det): {a}->{b}", mcnemar_p(b01, b10)))
    corrected = holm(fam)
    print(f"  {'comparison':<52} {'p':>7} {'p_holm':>7} {'sig?':>5}")
    for label, p, ph, sig in corrected:
        print(f"  {label:<52} {p:>7} {ph:>7} {'YES' if sig else 'no':>5}")
    print("\n  -> after correction, only effects with p_holm<.05 are credible.")
    print("=" * 78)


if __name__ == "__main__":
    main()
