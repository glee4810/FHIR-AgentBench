Recomputed with `/Users/aanishsachdev/miniconda3/bin/python3`.

**Table:** reproduces exactly by counts/rounding.

| arm | overflow | matched | pooled |
|---|---:|---:|---:|
| A0 | 0/262 = 0.000% | 104/147 = 70.748% | 104/409 = 25.428% |
| A0' | 58/262 = 22.137% | 103/147 = 70.068% | 161/409 = 39.364% |
| A5 | 172/262 = 65.649% | 95/147 = 64.626% | 267/409 = 65.281% |

**Matched paired tests:** second minus first.

| pair | diff | McNemar discordants | exact p | patient-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|
| A0' - A0 | -0.68 pp | 4 vs 5 | 1.0000 | [-4.93, +3.15] pp |
| A5 - A0 | -6.12 pp | 2 vs 11 | 0.0225 | [-11.11, -1.40] pp |
| A5 - A0' | -5.44 pp | 4 vs 12 | 0.0768 | [-11.04, 0.00] pp |

So: A0 vs A0' matched null reproduces. Three-arm “statistically tied” does **not** cleanly reproduce: A5 is worse than A0 on matched under unadjusted paired McNemar and cluster bootstrap. With Holm over 3 pairwise tests, A0-vs-A5 becomes borderline/not-significant, so do not oversell code-harm either.

**A0' overflow:** reproduces: `97/262` original-overflow-stratum rows still hit `Input tokens exceeded` = 37.0%. Also `97/409` overall.

**Substrate parity:** patient IDs pass: `0/409` A0-vs-A0' `patient_fhir_id` mismatches; qid sets all `409`.

**Does not reproduce:** global “A0' retrievals nonempty” is false. I get `402/409` nonempty, `7/409` empty. All 7 empty rows are matched-stratum gender questions. Overflow-stratum A0' retrievals are nonempty: `262/262`.

**Statistical overreach:** “No arm reasons better / statistically tied” is too strong. The evidence supports “A0' gives no matched-budget lift over A0”; it does not support a clean three-way equality claim. Also “entire effect is overflow/data-shape, not reasoning” is too causal for post-hoc A0-overflow strata plus different A5 tooling/prompt/substrate mechanics. “Projection alone recovers 34%” is accurate for this cap-50 A0' configuration, not for projection in general.
