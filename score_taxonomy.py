#!/usr/bin/env python3
"""Score ablation cells with an HONEST failure taxonomy (Codex fold-in).

For each cell it reports, per question, not just right/wrong but WHY a wrong answer happened:
  - context_overflow : the 32k/raised input gate tripped (the cap artifact under test)
  - rate_limit       : 429 / transient infra (a confound, not a capability signal)
  - harness_error    : a crash/exception (infra, not capability)
  - no_answer        : agent hit max steps / returned nothing (a REAL retrieval failure — kept in scope)
  - answered         : the agent produced an answer (judged for correctness)

Two headline numbers, both reported (never silently drop failures):
  - raw_accuracy        : correct / ALL questions (failures count as wrong) — the product-level number
  - answerable_accuracy : correct / (questions minus context_overflow+rate_limit+harness_error)
                          — the capability number, with cap/infra artifacts removed SYMMETRICALLY from
                          every arm. A lift that only shows in raw but not answerable = cap-dodging,
                          not reasoning (the Finding A vs Finding B test).

Usage: EVAL_JUDGE_MODEL=gpt-5-mini python score_taxonomy.py runs/tier1
"""
import os
import sys
import csv
import json
import glob
import math
import random

JUDGE = os.environ.get("EVAL_JUDGE_MODEL", "gpt-5-mini")
INFRA_EXCLUDED = {"context_overflow", "rate_limit", "harness_error"}


def wilson_hw(p, n, z=1.96):
    """95% Wilson score-interval half-width — honest CIs from n binary trials (no fake seed reruns).
    At n=40 this is ~7-9pp, which is WHY a borderline +11pp lift needs this reported, not hidden."""
    if not n:
        return 0.0
    denom = 1 + z * z / n
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return round(margin / denom, 3)

# evaluation_metrics.py does argparse at import (--input required); neutralize our argv so we can
# reuse the benchmark's EXACT judge, then restore.
_saved_argv = sys.argv
sys.argv = [sys.argv[0], "--input", "_unused"]
try:
    from evaluation_metrics import check_answer_correctness_with_llm
except Exception as e:  # pragma: no cover
    print(f"WARN: could not import judge ({e}); taxonomy only, no correctness", file=sys.stderr)
    check_answer_correctness_with_llm = None
finally:
    sys.argv = _saved_argv


def classify(rec):
    err = (rec.get("error") or "")
    ans = (rec.get("agent_answer") or "")
    blob = (err + " " + ans).lower()
    if "input tokens exceeded" in blob or ("context" in blob and "exceed" in blob) or "context_length" in blob:
        return "context_overflow"
    if "429" in blob or "rate limit" in blob or "ratelimit" in blob or "too many requests" in blob:
        return "rate_limit"
    if ans.strip().startswith("Error:") or (err and not ans.strip()):
        return "harness_error"
    if not ans.strip() or "max steps reached" in ans.lower():
        return "no_answer"
    return "answered"


def judge(rec):
    if check_answer_correctness_with_llm is None:
        return None
    try:
        return int(check_answer_correctness_with_llm(
            rec.get("agent_answer") or "", rec.get("true_answer") or "",
            rec.get("question") or "", model=JUDGE))
    except Exception as e:
        print(f"  judge error qid={rec.get('question_id')}: {e}", file=sys.stderr)
        return None


def score_cell(path):
    recs = json.load(open(path))
    causes = {}
    correct_total = 0
    answerable = 0
    correct_answerable = 0
    judge_errors = 0
    n_to_judge = 0
    perq = {}  # question_id -> 0/1 raw correctness (infra failures count as 0) for PAIRED stats
    for r in recs:
        c = classify(r)
        causes[c] = causes.get(c, 0) + 1
        in_answerable = c not in INFRA_EXCLUDED
        if c in ("answered", "no_answer"):
            n_to_judge += 1
            j = judge(r)
            r["_judged"] = j
            if j is None:
                # FAIL-CLOSED: a judge error is NOT a wrong answer. Exclude it from every denominator
                # (and from paired stats) so a broken judge can't masquerade as a valid 0% table.
                judge_errors += 1
                continue
            v = 1 if j == 1 else 0
        else:
            v = 0  # infra / overflow failure -> genuinely counts as wrong in raw_acc
        qid = r.get("question_id")
        if qid is not None:
            perq[qid] = v
        if v == 1:
            correct_total += 1
            if in_answerable:
                correct_answerable += 1
        if in_answerable:
            answerable += 1
    if n_to_judge and judge_errors == n_to_judge:
        raise SystemExit(
            f"[FATAL] judge produced NO labels for {os.path.basename(path)} "
            f"({judge_errors}/{n_to_judge} judge calls failed) — check EVAL_JUDGE_MODEL (='{JUDGE}'). "
            f"Refusing to emit a misleading all-wrong table.")
    if judge_errors:
        print(f"  WARN {os.path.basename(path)}: {judge_errors}/{n_to_judge} judge calls failed -> those "
              f"questions are EXCLUDED from accuracy denominators (not scored 0).", file=sys.stderr)
    # PERSIST per-question judge labels (P0 fix): the prior version wrote ONLY aggregate _scores.csv, so
    # losing the run box destroyed the correctness evidence. Write a per-cell sidecar every time.
    judged_path = (path[:-5] if path.endswith(".json") else path) + ".judged.json"
    with open(judged_path, "w") as jf:
        json.dump([{"question_id": r.get("question_id"), "cause": classify(r), "judged": r.get("_judged"),
                    "agent_answer": r.get("agent_answer"), "true_answer": r.get("true_answer")}
                   for r in recs], jf, indent=2, default=str)
    n = len(recs)
    scored_n = n - judge_errors  # judge-errored questions are unscoreable -> out of the raw denominator
    raw_acc = round(correct_total / scored_n, 3) if scored_n else 0.0
    ans_acc = round(correct_answerable / answerable, 3) if answerable else 0.0
    row = {
        "cell": os.path.splitext(os.path.basename(path))[0],
        "n": n,
        "judge_errors": judge_errors,
        "answered": causes.get("answered", 0),
        "no_answer": causes.get("no_answer", 0),
        "overflow": causes.get("context_overflow", 0),
        "rate_limit": causes.get("rate_limit", 0),
        "harness_error": causes.get("harness_error", 0),
        "raw_acc": raw_acc,
        "raw_ci": wilson_hw(raw_acc, scored_n),
        "answerable_n": answerable,
        "answerable_acc": ans_acc,
        "ans_ci": wilson_hw(ans_acc, answerable),
    }
    return row, perq


def mcnemar_bootstrap(perq_a, perq_b, n_boot=5000, seed=0):
    """Paired comparison of two arms on the SAME questions (arms share question_ids).
    McNemar exact (binomial) on discordant pairs + a paired-bootstrap 95% CI on the accuracy delta
    (b - a). This is the right test for the design — independent Wilson intervals waste the pairing."""
    import math
    ids = [q for q in perq_a if q in perq_b]
    if not ids:
        return None
    a = [perq_a[q] for q in ids]
    b = [perq_b[q] for q in ids]
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # a wrong, b right
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)  # a right, b wrong
    n_disc = b01 + b10
    if n_disc == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        p = min(1.0, 2.0 * sum(math.comb(n_disc, i) for i in range(0, k + 1)) / (2 ** n_disc))
    delta = (sum(b) - sum(a)) / len(ids)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(ids)) for _ in range(len(ids))]
        deltas.append((sum(b[i] for i in idx) - sum(a[i] for i in idx)) / len(ids))
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    return {"n_paired": len(ids), "a_acc": round(sum(a) / len(ids), 3), "b_acc": round(sum(b) / len(ids), 3),
            "delta": round(delta, 3), "ci95": [round(lo, 3), round(hi, 3)],
            "b01_a_wrong_b_right": b01, "b10_a_right_b_wrong": b10, "mcnemar_p": round(p, 4)}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "runs/tier1"
    paths = sorted(p for p in glob.glob(os.path.join(d, "*.json"))
                   if not os.path.basename(p).startswith("_")
                   and not p.endswith(".judged.json"))  # don't re-score our own per-question label sidecars
    if not paths:
        print(f"no cell files in {d}")
        return
    rows = []
    perq_by_key = {}      # (arm, sample, cap) -> {question_id: 0/1}
    answered_by_key = {}  # (arm, sample, cap) -> count of gradeable answers
    for p in paths:
        print(f"scoring {os.path.basename(p)} ...", flush=True)
        row, perq = score_cell(p)
        rows.append(row)
        parts = row["cell"].split(".")  # surface.arm.model.sample.cXXk
        if len(parts) >= 4:
            perq_by_key[(parts[1], parts[-2], parts[-1])] = perq
            answered_by_key[(parts[1], parts[-2], parts[-1])] = row["answered"] + row["no_answer"]
    cols = ["cell", "n", "judge_errors", "answered", "no_answer", "overflow", "rate_limit",
            "harness_error", "raw_acc", "raw_ci", "answerable_n", "answerable_acc", "ans_ci"]
    out_csv = os.path.join(d, "_scores.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # pretty print (acc shown with its 95% Wilson half-width)
    width = max(len(r["cell"]) for r in rows) + 2
    print("\n" + "cell".ljust(width) + "  n  ansd  no_ans  ovfl  429  err   raw_acc±ci      ans_acc±ci")
    for r in rows:
        print(f"{r['cell'].ljust(width)} {r['n']:>2}  {r['answered']:>4}  {r['no_answer']:>5}  "
              f"{r['overflow']:>4}  {r['rate_limit']:>3}  {r['harness_error']:>3}  "
              f"{r['raw_acc']:>5}±{r['raw_ci']:<5}  {r['answerable_acc']:>5}±{r['ans_ci']:<5}")
    print(f"\nsaved {out_csv}")

    # ---- PAIRED comparisons (the headline: decomposes coaching-lift vs structure-lift) ----
    # comps are KEYS (arm,sample,cap), resolved in the loop, so we can also screen for degenerate cells.
    comps = [
        # DOSE-RESPONSE CURVE (rep): nested staircase — each step ADDS tools; watch for a peak-then-decline
        ("curve 1->2 (rep): control(1g) -> 2 tools", ("control", "rep", "c100k"), ("cat2", "rep", "c100k")),
        ("curve 2->4 (rep): 2 -> 4 tools", ("cat2", "rep", "c100k"), ("cat4", "rep", "c100k")),
        ("curve 4->5 (rep): 4 -> 5 tools", ("cat4", "rep", "c100k"), ("validated5", "rep", "c100k")),
        ("curve 5->6 (rep): 5 -> +resolve_refs(6)", ("validated5", "rep", "c100k"), ("arm_ref", "rep", "c100k")),
        ("curve 6->8 (rep): 6 -> 8 tools  [DIMINISHING-RETURNS TEST]", ("arm_ref", "rep", "c100k"), ("arm_full8", "rep", "c100k")),
        ("curve TOTAL (rep): control -> full 8-tool", ("control", "rep", "c100k"), ("arm_full8", "rep", "c100k")),
        ("frugal (rep): control -> c0(_elements)", ("control", "rep", "c100k"), ("c0", "rep", "c100k")),
        # MEDICATION 3-arm decomposition
        ("coaching-lift  (med): control -> control_include", ("control", "med", "c100k"), ("control_include", "med", "c100k")),
        ("structure-lift (med): control_include -> arm_ref", ("control_include", "med", "c100k"), ("arm_ref", "med", "c100k")),
        ("TOTAL lift     (med): control -> arm_ref", ("control", "med", "c100k"), ("arm_ref", "med", "c100k")),
        ("cap effect on control (med): 32k -> 100k", ("control", "med", "c32k"), ("control", "med", "c100k")),
        ("cap effect on arm_ref (med): 32k -> 100k", ("arm_ref", "med", "c32k"), ("arm_ref", "med", "c100k")),
    ]
    print("\n=== PAIRED comparisons (McNemar + paired-bootstrap 95% CI; * = p<.05) ===")
    paired_out = []
    for label, ka, kb in comps:
        a, b = perq_by_key.get(ka), perq_by_key.get(kb)
        if a is None or b is None:
            print(f"  {label}: (cell missing)")
            continue
        # A cell that produced ZERO gradeable answers (all overflow/rate-limit/error) never actually ran;
        # pairing against it manufactures a fake significant 'drop' (e.g. arm_full8's quota failure). Skip it.
        if answered_by_key.get(ka, 0) == 0 or answered_by_key.get(kb, 0) == 0:
            print(f"  {label}: (skipped — a cell produced 0 gradeable answers / all infra-failure, not a result)")
            continue
        r = mcnemar_bootstrap(a, b)
        if not r:
            print(f"  {label}: (no shared questions)")
            continue
        paired_out.append({"comparison": label, **r})
        sig = "*" if r["mcnemar_p"] < 0.05 else " "
        print(f"{sig} {label}: {r['a_acc']:.2f} -> {r['b_acc']:.2f}  d={r['delta']:+.3f} "
              f"CI95[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]  p={r['mcnemar_p']:.4f} "
              f"(n={r['n_paired']}, disc b>a={r['b01_a_wrong_b_right']} a>b={r['b10_a_right_b_wrong']})")
    if paired_out:
        with open(os.path.join(d, "_paired.json"), "w") as f:
            json.dump(paired_out, f, indent=2)
        print(f"\nsaved {os.path.join(d, '_paired.json')}")


if __name__ == "__main__":
    main()
