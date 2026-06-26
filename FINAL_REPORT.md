# Final report — what actually moves an LLM agent's accuracy on FHIR clinical QA

*The complete, integrated result of this fork's experiments, including the new three-arm A0/A0′/A5 control.
Substrate: [FHIR-AgentBench](https://arxiv.org/abs/2509.19319) (MIMIC-IV-on-FHIR demo, single-patient
retrieval QA), 409-question paired held-out split, GPT-5.5. Grading: deterministic numeric checks on the
numeric subset + judge panels for non-numeric answers. **Red-teamed 2026-06-25** (8-skeptic workflow + 2 codex
reviewers, all recomputing from the local raw-answer dumps); the numerical table below reproduces locally, and
the labeling/interpretation fixes they required are incorporated. Status: final.*

---

## The one-paragraph answer

For an LLM agent over FHIR in these runs, **the largest observed accuracy effect is whether the data the
question needs can reach the model's context without overflowing it — and when both arms produce real answers
we detect no reasoning benefit from the tested tool/configuration changes.** Raw FHIR overflows the 32k window
on **64% of single-patient questions** (one ICU patient's record far exceeds the cap — one observed retrieval was ~2M
tokens), and those questions fail outright. The fix is getting the *right, bounded* slice into context. A
sandboxed code interpreter does this and recovers the lost accuracy almost entirely; a **blunt, query-blind
projection** of raw resources (strip + a single recency cap) recovers only about a third — but its failures
are mostly fixable *in context*: it overflows on accumulation, and it drops exactly the data many questions
ask for (the *earliest* value, while a recency cap keeps the most recent). The lever is therefore
**query-aware data selection** (fetch/keep what the question needs), which a code sandbox provides but is **not
exclusive to it**. The local decomposition gives an untested query-aware in-context projection a heuristic
ceiling of ~67.6%, numerically close to the sandbox's 65.6%, but that arm has not been run. The popular
framings — "more/typed tools help," "a code interpreter reasons better," "give the agent a bigger context" —
are wrong or beside the point for this setup.

## The headline result (trustworthy grading; p-values are exact paired McNemar; CIs are patient-cluster bootstrap)

| arm | overflow stratum (n=262) | resource-real stratum (n=147) | pooled (n=409) |
|---|---|---|---|
| **A0 — raw FHIR in context** | 0.0% | 70.7% | 25.4% |
| **A0′ — projection only** (strip + recency-cap 50) | 22.1% | 70.1% | 39.4% |
| **A5 — code interpreter (sandbox)** | 65.6% | 64.6% | 65.3% |

**Paired tests on the resource-real stratum (n=147):**
- **A0′ vs A0 (projection vs raw): −0.7pp, McNemar p=1.00, cluster CI [−4.9, +3.2] — a clean null.** Projection
  gives no accuracy benefit when the record already fits.
- **A5 vs A0 (code vs raw): −6.1pp, p=0.0225, cluster CI [−11.1, −1.4] — code is significantly *worse*** (a
  reliability tax from buggy generated code; 11 breaks vs 2 fixes; borderline after Holm correction over 3
  pairs, and rests on 13 discordant pairs).
- So **no arm reasons *better* than raw** when the data fits; projection is tied, code is slightly worse. This
  is **not** an "all arms equal" claim. (The cleanest matched-budget null — questions where *both* arms
  produced a real answer, n=140 — is also a null: A0 71.4% vs A5 67.9%, −3.6pp, p=0.18; see
  [TRUSTWORTHY_REGRADE.md](TRUSTWORTHY_REGRADE.md).)

**Overflow stratum (n=262 — questions where A0 overflowed):** raw 0% → projection 22.1% → code 65.6%.
Projection-alone recovers **34%** of the code arm's overflow-stratum accuracy *under this blunt cap*. It is the
genuine effect, and it is overflow-avoidance.

> **Stratum honesty:** "overflow" and "resource-real" are defined by whether A0 overflowed — an A0-observed
> proxy for record size under this retrieval path, and a *post-hoc* split. The resource-real stratum is therefore
> A0-success-selected (small records, median ~8 resources), so "projection ≈ raw when data fits" is partly a
> selection artifact.

## Why A0′ recovers only a third (the decomposition that corrects the interpretation)

A0′'s 262 overflow questions break down as **58 correct / 97 still-overflow (37%) / 107 fit-but-wrong**. The
recency-cap-50 is an *adversarial, query-blind* projection, and most of its failures are config artifacts, not
reasoning:
- **82 of the 107 wrong answers say "cannot find / truncated"** — the cap dropped the needed data.
- **40 of the 107 fit-but-wrong questions ask for the *first/earliest* value**, which a *recency* cap
  structurally discards (it keeps the most recent). A keep-first-and-last or question-date-filter could recover
  that class with no sandbox.
- **The 97 residual overflows are multi-turn *accumulation*, not single-payload size** (0/97 have a single
  projected block >32k; max 24.8k). 53% re-requested the *same* resource type — agent-loop waste, orthogonal
  to projection quality.
- **The code arm proves the data existed:** it recovers 55 of the 82 cap-drops and 64 of the 97 overflows.

A hypothetical **query-aware in-context projection** (fetch the question's resource type + date range, keep
first-and-last, dedup re-requests, no blunt cap) is heuristically upper-bounded by the local decomposition at
**(58+55+64)/262 = 67.6%**, numerically close to the sandbox's **65.6%**. That is a ceiling, not a measured
paired arm: it uses A5 successes as evidence that the data existed, not proof that an in-context projection
would retrieve, fit, and answer those rows. So the sandbox's observed advantage is **query-aware selection, not
out-of-context compute per se**, but whether an in-context query-aware layer can match it remains the decisive
next experiment. The "34% / naive projection insufficient" number is correct *for the blunt recency-cap-50
configuration*; it is a floor, not a ceiling for projection as a class.

## How we got here (three corrections, each from adversarial review)

1. **The celebrated "+11pp code interpreter win" was a context-overflow + harness-bug artifact** (our scorer
   skipped the benchmark's error pre-filter; the pooled mix is dominated by the overflow stratum).
2. **The benchmark's default judge (gpt-5-mini) is unreliable** — 61% accurate vs non-LLM numeric ground truth,
   one-directional precision bias (it manufactured a phantom "code HURTS −8.6pp"). Replaced with deterministic
   numeric grading + a 3-Claude-judge panel, cross-checked by an independent codex/GPT panel (97% mutual
   agreement; both panels 98–99% vs ground truth).
3. **A boolean-grading bug in our own first fix** (115 Yes/No golds graded as numeric) was caught and routed to
   the panel.

## What this licenses us to claim (and what it does not)

**License (scoped to this single-model, single-substrate setup):**
- The dominant observed accuracy lever is whether the needed payload reaches context without overflowing.
- When both arms produce real answers, we **detect no reasoning benefit** from typed tools, payload shaping,
  reasoning effort, projection, or a code interpreter.
- The lever is **query-aware data selection**; a sandbox provides it, and a smart in-context projection plausibly
  could too (heuristic ceiling near the sandbox); **blunt query-blind projection is insufficient** (recovers ~⅓ under
  cap-50).
- The benchmark's default judge is unreliable; a multi-vote panel mitigates the judge-family dependence.

**Do NOT claim:** "typed tools / projections make agents reason better" (no detected matched-budget benefit);
"projection alone solves it" (only 34% under blunt cap); "a sandbox is *required*" (a query-aware in-context
projection has not been run); exact three-arm equivalence / zero reasoning effect; that the result holds across
projection caps (one cap tested, no sweep); strict substrate parity (see caveats); that interpreter compute
was isolated from its prompt/routing; that panel labels are human ground truth; that this beats Medplum's MCP
(A0 ≈ raw-dump is a weak baseline — beating it only proves "don't dump raw FHIR").

## Honest caveats

- **Substrate parity (next to the table, not buried):** A0/A5 ran on the original (now torn-down) EC2 Medplum;
  A0′ on a freshly-loaded Medplum on a Mac mini. Verified parity from the answer files — identical question_id
  sets, and the *load-bearing* check, `agent_fhir_resources` UUID overlap, has median Jaccard 1.0 / mean 0.97
  (the PUT-preserves-ids signature; one large held-out patient returns the same Observation ID set in both A0
  and A0′). This supports parity but is **not** a strict same-instance control (the arms also differ in
  multi-turn retrieval *path*). 402/409 A0′ retrievals are non-empty (the 7 empty are gender questions, empty
  in A0 too).
- **A0′ is one configuration (recency-cap = 50)** and is grading-asymmetric: A0′ non-numeric labels are from a
  **single-family (codex-only) 3-vote panel** — the original's Claude+codex cross-family check does **not**
  transfer to A0′; codex is the stricter family, so A0′ numbers are a conservative lower bound.
- **Payload reduction** is large but config-dependent: ~60–67% typical vs A0's raw context, up to ~99% on the
  largest single retrievals.
- **Strata are post-hoc** (defined by A0 overflow); the matched comparison conditions on A0 success.
  Questions cluster across ~90 patients (cluster bootstrap used above).
- **Single model (GPT-5.5), single seed, single substrate; self-built/self-graded** (mitigated by deterministic
  grading, multi-family panels on the original arms, and a reproducibility-scoped harness — not by external/blinded human
  adjudication).

## Reproducibility

The branch commits the final reports, frozen summaries, A0′ panel votes, and `_strata.json`. Exact answer-level
recomputation of the A0/A5/A0′ table still requires the local raw answer dumps
(`runs/full409/multi_turn_{resource,code_resource}.json` and
`runs/a0prime/multi_turn_projected_resource.json`), which are large and gitignored. `a0prime_verdict.py`
recomputes the table when those dumps are present, but a fresh checkout currently verifies the committed
summaries rather than regenerating them from raw answers. Agent reruns need the Medplum substrate
(`medplum-eval-bundle/`) + a funded key.

## Contributions

1. The correct decomposition: the code "win" is a context-budget artifact, not compute — the same confound
   faked the tool-catalog "win."
2. The judge-reliability finding: the benchmark's default judge is 61% accurate; a multi-vote panel mitigates it.
3. The A0′ control: overflow-avoidance is the lever, *how* you avoid it matters, the likely lever is
   **query-aware selection** (not out-of-context compute exclusively), and blunt query-blind projection recovers
   only a third.
4. A reusable, trustworthy-graded harness on a self-hosted FHIR substrate.

## The decisive next experiment

A single **query-aware in-context projection** arm (fetch the question's resource type + date range, keep
first-and-last, dedup re-requests, no blunt cap) would confirm or collapse the "~67.6% ceiling near the sandbox"
hypothesis — and settle whether a sandbox is necessary here or merely one way to do query-aware selection.
