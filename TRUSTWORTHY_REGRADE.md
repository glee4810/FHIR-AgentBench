# Trustworthy re-grade — why the headline numbers changed (three times), and the grading that fixed them

*This is the audit trail for the final, defensible numbers in [FINDINGS.md](FINDINGS.md) and
[CODE_EXPERIMENT.md](CODE_EXPERIMENT.md). It documents two real defects we caught in our own evaluation —
a harness bug and an unreliable LLM judge — plus a grading bug we caught in our **own fix**, and the
deterministic + multi-judge-panel grading we ended on. Read this before citing any accuracy number.*

---

## TL;DR

The number moved three times under adversarial review; each move corrected a real defect.

| pass | grading | matched-budget verdict | defect corrected |
|---|---|---|---|
| 1 | gpt-5-mini, no error pre-filter | "+11pp, code is the lever" | harness bug: overflow/error answers fed to the judge instead of auto-scored 0; pooled mixes overflow |
| 2 | gpt-5-mini, canonical pre-filter | "−8.6pp, code HURTS" (p=0.02) | **judge artifact**: gpt-5-mini marks code's terse/precise numerics wrong |
| 3a | deterministic-numeric + Claude panel | "+1.4pp, null" | better, but the deterministic layer **mis-graded 115 boolean Yes/No golds as numeric** |
| 3b | **deterministic(clean numerics+failures) + Claude panel, boolean fixed** | **−3.6pp, not significant (p=0.18)** | the trustworthy answer |

The final, trustworthy, judge-family-independent result:

| stratum | n | resource | code | Δ (95% CI) | McNemar |
|---|---|---|---|---|---|
| **matched budget** (both arms answered) | 140 | 71.4% | 67.9% | **−3.6pp** (−7.7…+0.6) | p=0.18 → not significant |
| **resource-real** (predefined by resource success) | 147 | 70.8% | 64.6% | **−6.1pp** (−10.8…−1.4) | p=0.022 → significant |
| **large records** (resource overflows 32k) | 262 | 0% | 65.6% | — | by construction |
| pooled (mixes the strata) | 409 | 25.4% | 65.3% | +39.9pp (+34.6…+45.1) | p≈0 |

**The +11pp was never a reasoning/compute win.** On questions the no-code agent can answer, the code
interpreter gives no significant benefit and a slightly *negative* point estimate. Its entire positive value
is mechanical: the no-code agent overflows the 32k context on **262/409 (64%)** of questions, and the code
agent avoids that by handing the payload to a sandbox rather than serializing it into the prompt.

---

## Defect 1 — the harness bug (pass 1)

Our scoring scripts called the LLM judge *directly*, skipping the benchmark's own error pre-filter
(`evaluation_metrics.py:133-137` scores any error / `Input tokens exceeded` / empty answer **wrong** before
the judge sees it). Overflow garbage was reaching the judge, which sometimes credited it — inflating the
no-code arm and muddying the comparison. Fixed: failures are deterministically 0 for both arms.

## Defect 2 — the benchmark's default judge (gpt-5-mini) is unreliable (pass 2)

After the pre-filter, a *canonical gpt-5-mini* pass said the code arm significantly **hurt** at matched
budget (−8.6pp, p=0.02), claiming code "broke 18" questions. This was a **judge artifact**. We proved it two
ways:

**(a) Against non-LLM ground truth.** On the 97 numeric-gold questions we can grade deterministically (the
gold is a known number; match within tolerance — no LLM needed), we scored each judge's accuracy:

| judge | accuracy vs ground truth (111 arm-answers) | false negatives (correct→"wrong") | false positives (wrong→"correct") |
|---|---|---|---|
| **gpt-5-mini** (benchmark default) | **61.3%** | **43** | 0 |
| codex / GPT panel (3-vote) | **99.1%** | 1 | 0 |
| Claude panel (3-vote) | **98.2%** | 1 | 1 |

gpt-5-mini is **61% accurate** against known truth, with a purely **one-directional** failure: it wrongly
rejects correct, precise answers (43 false negatives) and essentially never accepts a gross miss (0 false
positives). The [magnitude analysis](runs/full409/_magnitude_analysis.json) shows those 43 rejected answers
had a **median relative error of 0.0** — they were exactly right. That bias punishes the code arm (whose
numeric answers are terse and exact), which is precisely how it manufactured a phantom "code hurts."

**(b) Against a different model family.** A 3-vote Claude panel and a 3-vote codex/GPT panel — independently
re-judging the non-numeric questions — agree with **each other on 408/420 = 97.1%** of labels, while each
agrees with gpt-5-mini only ~65-67%. The outlier is gpt-5-mini, confirmed from two directions.

Overall, gpt-5-mini disagreed with the trustworthy labels on **34.3% (181/527)** of real-answer judgments.

## Defect 3 — our own fix had a grading bug (pass 3a → 3b)

The first deterministic layer classified any gold containing a digit as "numeric" and graded it by tolerance.
But the benchmark encodes **Yes/No as `[[1]]`/`[[0]]`** (`evaluation_metrics.py:176-178`: "evaluate based on
meaning, not syntax"). So 115 boolean questions (114 bare `[[1]]`/`[[0]]` + 1 quoted `[['1']]`) were sent to
the numeric grader, and free-text "Yes"/"No"
answers (no number to extract) were auto-scored 0 — **99 of 230 boolean arm-labels were wrong**, and because
`final_grade.py` prefers deterministic labels, the panel could never correct them. Caught by the
grading-integrity adversarial review.

**Fix (`build_labels.py`):** a label is deterministic *only* when genuinely unambiguous — (i) any failure → 0,
(ii) a gold that is numeric **and not 0/1** with a tolerance-matching answer. Boolean `0/1` golds (ambiguous:
Yes/No vs a count of 1 — only the question disambiguates) and all categorical/list golds go to the
**Claude + codex judge panels**. Of the 115 boolean golds, 3 had both arms fail (deterministic 0) and 112 had
a real answer routed to the panel (111 newly judged; 1 was already in the original panel set). Re-judging them
moved matched budget from a +1.4pp null to the trustworthy −3.6pp (still not significant) and raised both
arms' absolute accuracy.

## The grading we ended on

Final label for each (question, arm): **deterministic where genuinely unambiguous, else 3-Claude-judge panel
majority.**

- **Deterministic (`runs/full409/det_labels.json`, 402 labels):** failures → 0 (both arms); clean numeric
  golds (not 0/1) → tolerance match against the known gold. This is the part that needs no LLM and is what
  exposed gpt-5-mini.
- **Claude panel (`runs/full409/panel_votes*.json`):** the 188 categorical/"other" questions + the 112
  boolean questions with a real answer, each graded by **3 independent Claude judges, majority vote**,
  ignoring abstentions. 0 unresolved ties.
- **Independent codex/GPT panel (`runs/full409/codex_votes*`):** the same questions, 3 codex passes,
  majority — used to show the result is judge-family-independent, not to set the final labels.

`final_grade.py` merges these and re-runs the stratified McNemar with 95% CIs.
`codex_judge_compare.py` reproduces the entire stratified result using the **codex/GPT** labels instead of
Claude's: matched budget −3.6pp (p=0.18), large records code 64.5%, pooled +39.1% — the same conclusion
whichever trustworthy judge you believe.

## The strata, and their honest limits

- **matched budget (n=140):** both arms produced a real answer. This **conditions on a post-treatment
  outcome** (both surviving), so it is a deliberately conservative read of code's *reasoning* benefit. Verdict:
  no significant difference (−3.6pp, 95% CI −7.7…+0.6, p=0.18). This is **absence of evidence of a code
  benefit, not proof of exact equivalence** — only 9 discordant pairs drive it; we are not powered to certify
  a zero effect.
- **resource-real (n=147):** predefined by resource success only (the cleaner control the stats review asked
  for, not conditioned on code's outcome). Code is significantly worse (−6.1pp, p=0.022) — but **decomposing
  the 11 questions code broke: 4 are code erroring/overflowing (a reliability gap) and 7 are wrong real answers
  (buggy generated code).** So roughly a third of the "significant" deficit is the code arm being less
  *reliable*, not less *accurate when it answers*. **And the significance is fragile:** it survives a
  patient-cluster bootstrap, but leave-one-patient-out (dropping the one patient with 2 discordant pairs,
  subject 10005909) lifts p to 0.065 — a small-N fragility. Treat the matched-budget null (p=0.18) as the
  load-bearing result and resource-real as a directional, fragile corroboration.
- **large records (n=262):** resource overflows the 32k cap (0% by construction); code produced a real answer
  on 240 and got 65.6% of all 262. The one robust, large effect — and it is **architectural** (where the data
  lives), not compute.

**Caveats we do not paper over:**
- **The strata are defined post-hoc** (by which arm overflowed/answered), not by pre-tokenized record size; a
  cleaner design would predefine size strata before either arm runs.
- **Questions are not independent** — the 409 cluster across ~90 patients (up to 30 questions each). McNemar
  treats pairs as independent; a patient-clustered bootstrap would widen the CIs (it does not threaten the
  matched-budget null, which is already non-significant, but it tempers the precision of the significant
  strata).
- **Panel agreement is reproducibility, not ground truth** — except on the numeric subset, where we *do* have
  non-LLM truth and the panels score 98-99%. For the categorical/boolean questions, blinded human
  adjudication on a sample remains the gold standard; the committed `runs/full409/human_review.csv` (every
  question, both answers, all four judges' labels) exists so a human can do exactly that.
- **The "architecture not compute" mechanism is not fully identified.** The code arm differs from the no-code
  arm in three ways at once — the interpreter, a code-tailored prompt, and pointer-based payload routing. We
  show the pooled win is overflow-driven and the matched-budget reasoning effect is null/negative; we have
  *not* isolated interpreter-compute from prompt and routing (that needs a same-payload, no-execution control).

## Files

- `build_labels.py` — rebuilds the deterministic layer (the boolean-bug fix) and emits panel batches.
- `final_grade.py` — deterministic + Claude panel → stratified McNemar + CIs + gpt-5-mini disagreement.
- `judge_leaderboard.py` — judge accuracy vs non-LLM ground truth on the numeric subset (the 61% vs 99% table).
- `magnitude_analysis.py` — agent numeric-error magnitudes + gpt-5-mini's one-directional precision bias.
- `codex_panel*.sh` + `codex_judge_compare.py` — the independent codex/GPT panels and the family-independence check.
- `runs/full409/_trustworthy_summary.json`, `_judge_leaderboard.json`, `_magnitude_analysis.json`,
  `_codex_triangulation.json` — the machine-readable results.
- `runs/full409/human_review.{json,csv}` — all 409 questions, both arms' answers, every judge's label, for human audit.
