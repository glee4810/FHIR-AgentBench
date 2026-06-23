**Verdict:** the decomposition is plausible, but the paper-level claims overstate what the stats identify.

- **Matched budget is a post-treatment filter.** [final_grade.py:71](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/final_grade.py:71) selects only questions where both arms produced real answers. That conditions on outcomes affected by the treatment/architecture. It excludes 7 cases where resource answered but code failed; 4 of those were resource-correct/code-wrong. If the estimand is “resource can fit,” the cleaner stratum is resource-real/non-overflow regardless of code failure: resource 70/147 vs code 68/147, delta **-1.4pp**, fixed 6/broke 8, p=0.79. Same null, opposite sign.

- **“p=0.75 => NULL” is too strong.** McNemar is computed correctly for the selected pairs ([final_grade.py:59](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/final_grade.py:59)), but only 10 discordant pairs drive it. The matched 95% CI is roughly **-3.4pp to +5.4pp**. That is “no statistically significant evidence,” not evidence of no effect. To claim no meaningful gain, they need an equivalence margin and power/MDE analysis.

- **Matched budget does not isolate “reasoning.”** The arms differ by code tool, code-tailored prompt, and payload plumbing ([CODE_EXPERIMENT.md:11](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/CODE_EXPERIMENT.md:11)). Both-answering does not equal equal information path or equal budget. It just means both survived.

- **The “architecture not compute” causal claim is not identified.** The overflow stratum proves the no-code implementation with a 32k serialized prompt fails. It does not prove Python compute is irrelevant on large records. Needed control: same payload pointer/projection/100k context but no Python compute, ideally same prompt except execution affordance.

- **The overflow stratum is also post-treatment.** “Large records” is defined as resource arm overflow ([final_grade.py:73](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/final_grade.py:73), [TRUSTWORTHY_REGRADE.md:122](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/TRUSTWORTHY_REGRADE.md:122)). Better: predefine strata by tokenized record size before either arm runs.

- **“Entirely overflow” is rhetorically sloppy.** Exact decomposition: overflow contributes **+139 correct**; non-overflow/resource-real contributes **-2**. So the pooled +137 is more than explained by overflow and slightly offset elsewhere. Fine as shorthand, bad as causal language.

- **Pooled +33pp is valid for the deployed architecture comparison, not for reasoning.** They mostly say this ([FINDINGS.md:55](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/FINDINGS.md:55)), but any headline using pooled accuracy must say: “code-agent architecture beats resource-agent architecture under 32k cap,” not “code interpreter improves accuracy.”

- **Question independence is dubious.** The 409 questions cluster by patient: 90 patients, max 30 questions/patient. McNemar treats question pairs as independent. They need patient-cluster bootstrap/permutation or hierarchical modeling, especially for pooled/overflow claims.

- **Judge reliability is overclaimed.** The 29.3% disagreement in [final_grade.py:79](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/final_grade.py:79) compares old cached gpt labels directly; if the benchmark prefilter is applied first, disagreement is about **23.6%**, not 29.3%. Still bad, but the cited number conflates judge failure with the earlier harness bug.

- **Panel agreement is not truth.** Claude-vs-codex 95.3% agreement ([codex_judge_compare.py:61](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/codex_judge_compare.py:61)) shows reproducibility under similar rubric, not validity. Need blinded human adjudication on a sample, kappa/precision by answer type, and sensitivity of conclusions to disputed labels.

- **`rescore_canonical.py` has a stale denominator bug.** Its large-record stratum requires resource overflow **and code real** ([rescore_canonical.py:90](/Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench/rescore_canonical.py:90)), inflating code accuracy by dropping code failures. `final_grade.py` fixes this, but the audit trail contains an inconsistent scorer.

**What would change the conclusion:** include the 7 code-failure/resource-real cases, report CIs/equivalence tests, stratify by pre-run token size, cluster by patient, and run a no-code overflow-avoidance control. Until then, the defensible claim is narrower: **under this 32k serialized-resource architecture, code-agent plumbing avoids many overflows; among surviving small cases there is no detected advantage, but the study is not powered or designed to prove zero reasoning benefit.**