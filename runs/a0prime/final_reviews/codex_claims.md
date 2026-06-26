**Verdict**

Mostly coherent, but not clean enough for hostile review.

1. Numeric contradictions / weak spots:
- Headline A0/A0′/A5 table in [FINAL_REPORT.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINAL_REPORT.md:26) matches `runs/a0prime/_verdict_summary.json` and `a0prime_verdict.py`.
- Problem: `matched-answerable n=147` is not the older `matched budget n=140` in [TRUSTWORTHY_REGRADE.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/TRUSTWORTHY_REGRADE.md:25). It is the `resource-real/predefined` stratum. Do not call it “matched-budget” without explanation.
- Problem: summary JSON says the three-way experiment was on “Medplum on Mac mini”; [FINAL_REPORT.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINAL_REPORT.md:82) says A0/A5 were EC2 and A0′ was mini. That is a direct metadata contradiction.
- Problem: “99% payload reduction” is not supported by these docs. A trace char-count check gave ~63% reduction on A0′ full retained resources vs projected context for overflow rows, ~67% vs A0 raw trace. If 99% is token-based or subset-based, state the denominator.
- Problem: “millions of tokens” is not supported in the reviewed artifacts. Use “well beyond the 32k cap” unless you have a measured token max.

2. Main conclusion:
- Licensed: overflow/data-shape is the dominant observed lever in this setup. A0 overflow 0.0%, A0′ 22.1%, A5 65.6%; non-overflow/resource-real is ~70.7/70.1/64.6.
- Overclaimed: “not reasoning” as a universal causal statement. The honest version is “we detect no reasoning benefit at matched context budget.”
- Licensed but scoped: “out-of-context >> in-context projection” for this A0′ cap=50 projection vs this A5 sandbox.
- Licensed but scoped: “naive projection recovers only a third” only as `22.1 / 65.6 = 34%` on the overflow stratum, not as a general projection ceiling.

3. A0′ cap=50 caveat:
- The caveat exists, but the headline still treats `22.1%`/`34%` too generically.
- Also remove “structure holds across caps” unless you have a cap sweep. Current artifacts show one cap.

4. Substrate parity:
- Genuine threat, under-disclosed. It does not kill the result, but it prevents “strict three-arm causal control” language.
- Put the EC2-vs-mini caveat next to the table, not only at the bottom. The summary JSON should be fixed too.

5. “Do NOT claim” additions:
- Do not claim exact equivalence / zero reasoning effect.
- Do not claim best possible projection was tested.
- Do not claim results hold across projection caps without a cap sweep.
- Do not claim strict substrate parity.
- Do not claim interpreter compute was isolated from prompt/routing.
- Do not claim panel labels are human ground truth for non-numeric questions.

**Riskiest Rewrites**

[FINAL_REPORT.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINAL_REPORT.md:13):
“accuracy is gated by whether the relevant record fits in the model's context window, not by reasoning.”
Replace: “In these runs, the largest observed accuracy effect is whether the needed FHIR payload can be used without overflowing context; at matched context budget, we detect no reasoning benefit from the tested tool/configuration changes.”

[FINAL_REPORT.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINAL_REPORT.md:18):
“a sandboxed code interpreter … recovers the lost accuracy almost entirely”
Replace: “the sandboxed code arm restores overflow-stratum accuracy to roughly the matched-answerable range, though it bundles interpreter, prompt, and pointer-routing changes.”

[FINAL_REPORT.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINAL_REPORT.md:19):
“even a 99% payload reduction still overflows…”
Replace: “even this stripped, recency-capped projection still overflows 97/262 large-record questions; any payload-reduction percentage should be reported only with its exact denominator.”

[FINAL_REPORT.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINAL_REPORT.md:32):
“No arm reasons better”
Replace: “No tested arm shows a statistically established accuracy advantage when A0 can already retrieve a real answer.”

[FINAL_REPORT.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINAL_REPORT.md:61):
“shows the lever is out-of-context processing”
Replace: “shows that this out-of-context sandbox beats this naive in-context projection on the overflow stratum; it does not prove no stronger in-context projection could close more of the gap.”

[FINAL_REPORT.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINAL_REPORT.md:66):
“a smart out-of-context layer … is required”
Replace: “the tested evidence favors an out-of-context or answer-returning layer over blunt shrink-and-stuff projection; stronger projection/query designs remain untested.”

[FINAL_REPORT.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINAL_REPORT.md:78):
“the structure … holds across caps”
Replace: “we have not established a cap curve here; cap=50 is one point estimate, and other caps may trade off residual overflow against data drop differently.”
