Found surviving issues:

- `data-room/bonfire-benchmark-baseline.md:11`: `GPT-5.5-only), gpt-5-mini (LLM judge). All claims paired (McNemar exact).`  
  Contradicts corrected code result: code claims are not judged by default `gpt-5-mini`; they use deterministic + Claude panel, with gpt-5-mini explicitly unreliable.

- `data-room/bonfire-benchmark-baseline.md:42-43`: `The pitch is about *what the agent computes over* / and *what it's allowed to compute*, not how many verbs it has.`  
  Surviving “compute over the data” framing. Should be “what bounded/projected record slice reaches context,” not compute.

- `data-room/bonfire-benchmark-baseline.md:71-72`: `make the record **compute-ready upstream** (bounded projections / read-models / freshness)`  
  Risky surviving compute framing. “Context-ready” / “projection-ready” would match the corrected finding better.

- `data-room/bonfire-benchmark-baseline.md:81-83`: `Code-over-raw-FHIR works for *single-patient* temporal/aggregate questions because one patient's data (mostly) fits and can be sorted/summed in Python.`  
  Contradicts the corrected mechanism. The measured value is not sorting/summing in Python; it is avoiding prompt overflow. Also “one patient’s data mostly fits” conflicts with `262/409 (64%)` no-code overflows.

- `plans/medplum-mcp-eval-results.md:70-71`: `the valid claim is the **baseline-vs-treatment lift**, which is what this eval was built to measure.`  
  Despite the top banner, this still reads as a live +11pp/tool-lift claim. It should be explicitly marked historical/superseded or removed.

- `plans/medplum-mcp-pr-scope.md:174`: `A **sandboxed code-execution tool over a single patient's already-fetched resources** is the natural, high-leverage agent capability`  
  Not an explicit accuracy claim, but “high-leverage” keeps code positioned as the lever. Safer: “useful overflow-handling/ergonomics capability.”

No direct numeric contradiction with `FHIR-AgentBench/runs/full409/_trustworthy_summary.json` in the core corrected numbers I checked: matched-budget, resource-real, large-record, and pooled values are consistent up to rounding.