**Hostile Review**

The numbers mostly support a narrower claim: **the pooled code win is dominated by resource-arm context failures, and the matched-answer subset shows no detected code advantage.** The docs overstate that into a mechanistic conclusion.

**Biggest flaws:**

1. **“Architecture not compute” is a false dichotomy.**  
   The code arm changes `execute_python_code`, prompt, and data plumbing. You have not isolated those factors. The docs admit this in [CODE_EXPERIMENT.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/CODE_EXPERIMENT.md:11): code adds interpreter, code-tailored prompt, and pointer-based payload routing. So you can say “not shown to be compute,” not “not compute.”

2. **Matched-budget null is selection-conditioned.**  
   The n=140 subset is explicitly “both arms produced a real answer.” That likely selects easier/smaller questions. It cannot justify dismissing code’s practical value on the 262-question hard/large stratum. Your own data says code produced real answers on **240/262** overflow cases and got **53.0%** correct.

3. **The docs underclaim the useful code benefit.**  
   Calling the code gain “trivial” or saying it “does nothing significant” is too strong. End-to-end, code goes **17.1% → 50.6%**, +33.5pp. If the production problem includes large patients, that is a real system benefit, even if the mechanism is data access/plumbing.

4. **“Entirely” is too absolute.**  
   The pooled net is +137 correct answers. The overflow stratum contributes about +139 code-correct/resource-zero cases, while matched budget contributes +2 and the remaining edge cases subtract about 4. “Dominated by overflow” is defensible. “Entirely” is rhetorically brittle.

5. **Judge critique is plausible but still overconfident.**  
   Claude vs codex/GPT agreement is strong: **95.3% on 254 shared non-numeric labels**. But the headline “gpt-5-mini disagreed on 29.3% of all pairs” compares against labels partly produced by new heuristics/panels, not human gold. Say “against our trustworthy labels,” not “ground truth.”

6. **Payload projection equivalence is speculative.**  
   [FINDINGS.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINDINGS.md:102) says payload projection “also achieves” the value. But the code experiment did not prove projection recovers the 262 overflow cases at similar accuracy. [CODE_EXPERIMENT.md](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/CODE_EXPERIMENT.md:45) is better: “would be expected.”

**Riskiest Sentences**

- [FINDINGS.md:20](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINDINGS.md:20): “Every ‘win’ in this project decomposed to the same thing: the context budget.”  
  Too sweeping across underpowered/single-seed experiments.

- [FINDINGS.md:31](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINDINGS.md:31): “Its large pooled lift is entirely the 262 questions…”  
  Use “dominated by,” not “entirely.”

- [FINDINGS.md:60](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINDINGS.md:60): “That data-plumbing difference — not compute — is the entire pooled gap.”  
  Not experimentally isolated.

- [CODE_EXPERIMENT.md:40](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/CODE_EXPERIMENT.md:40): “It is not that Python does temporal-ordering/aggregation the model can’t.”  
  You did not test this directly on the overflow stratum.

- [CODE_EXPERIMENT.md:62](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/CODE_EXPERIMENT.md:62): “‘A code interpreter improves FHIR-agent accuracy’ is true only in the trivial, mechanical sense…”  
  Bad framing. Answering 240 questions the baseline cannot answer is not trivial.

- [TRUSTWORTHY_REGRADE.md:114](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/TRUSTWORTHY_REGRADE.md:114): “Whichever trustworthy judge you believe, the answer is the same…”  
  Fair for the final numerical conclusion, too broad for mechanistic causality.

**Stale/Contradictory Artifact**

[medplum-eval/full409_summary.json](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/medplum-eval/full409_summary.json:4) still preserves the old +11pp gpt-5-mini story: resource 27.6%, code 38.6%, “code_fixed_compute_heavy,” significant p=1.6e-05. It is not clearly marked superseded inside the JSON except a weak note. This is a citation hazard.

**Verdict**

The corrected framing should be: **“At matched-answer budget, this run does not detect a reasoning advantage from code. The large end-to-end code gain is mainly caused by avoiding resource-arm context overflow. We have not yet disentangled interpreter compute, prompt, and payload routing.”** Anything stronger is overclaiming.