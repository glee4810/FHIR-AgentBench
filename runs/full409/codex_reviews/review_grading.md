**Findings**

1. **Deterministic grading is not trustworthy for boolean questions.**
   `evaluation_metrics.py` says `[[1]]`/`[[0]]` are yes/no answers, but the deterministic layer appears to treat them as numeric. I found 75 obvious “Yes” answers to `[[1]]` and 22 obvious “No” answers to `[[0]]` marked deterministic/final `0`. Because [final_grade.py](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/final_grade.py:36) always prefers `det_labels.json`, the panel can never correct these. This is the biggest grading-integrity defect.

2. **“Deterministic where present, else panel” hides disagreement.**
   [final_grade.py](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/final_grade.py:42) makes deterministic labels authoritative, with no audit against Claude/codex/human labels. The codex panel actually has 376 labels for the 188 panel questions, but the Claude comparison only uses 254 shared non-deterministic labels; 122 codex labels are excluded because deterministic labels preempt them. So the cross-check does not validate the deterministic layer.

3. **The 29.3% gpt-5-mini disagreement rate is inflated.**
   [final_grade.py](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/final_grade.py:79) compares `judged_cache.json` directly to final labels. It does not first apply the canonical error/overflow pre-filter. I count 47 of the 240 disagreements coming from failure rows, including 43 resource overflows. Canonicalizing gpt-5-mini before comparison drops the rate from `240/818 = 29.3%` to `193/818 = 23.6%`. The judge may still be bad, but the headline number is contaminated by the known harness bug.

4. **The 95.3% Claude-vs-codex agreement is narrower than advertised.**
   [codex_judge_compare.py](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/codex_judge_compare.py:61) calls these “188 non-numeric pairs,” but it actually computes agreement over 254 shared arm-labels. The codex panel produced 376 labels; only the subset where Claude emitted a usable non-deterministic label is counted. It is not padded by `[FAILED]` auto-fails in that intersection, but it is still a selected subset and does not validate the deterministic labels.

5. **Both panels can share the same systematic bias.**
   The codex prompt gives the same gold answer, same question, same answer text, same arm labels, and same grading framing. Agreement between two LLM panels does not detect bad golds, ambiguous yes/no encoding, wrong deterministic assumptions, prompt-induced leniency, or benchmark-reference errors. “Different model family” is not independence from the grading specification.

6. **The Claude “3 independent judges” claim is weak.**
   `panel_votes.json` shows repeated same-task votes; 240/245 three-vote labels are unanimous. That may just mean easy cases, but it also may mean deterministic same-prompt behavior. There is no committed Claude-panel generation script/prompt comparable to `codex_prompt.txt`, so reproducibility and true independence are under-documented.

7. **The “entirely overflow artifact” wording overclaims.**
   From `_trustworthy_summary.json`, pooled delta is `+137` correct answers. Overflow contributes `+139`, matched budget contributes `+2`, and the remaining 7 edge cases contribute `-4`. “Dominated by overflow” is fair; “entirely” is rhetorically too strong.

8. **Matched-budget null is not proof of no reasoning gain.**
   The matched stratum has only 10 discordant pairs (`fixed 6`, `broke 4`). McNemar `p=0.75` means no detected difference, not equivalence. They need an equivalence margin/CI if they want “NO gain” rather than “not significant.”

Bottom line: the qualitative overflow story is plausible, and gpt-5-mini likely is unreliable. But the claimed “trustworthy” labels are not trustworthy until the deterministic boolean/numeric layer is fixed and audited against independent labels. The current scripts can both hide real grading disagreement and inflate the anti-gpt-5-mini headline.