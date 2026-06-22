# Retrieval quality report

No model calls were made. This report compares saved `agent_fhir_resources` against `true_fhir_ids`.

## Arm summary

| arm | recall | precision | gold rows covered | retrieved IDs | false positives | recall / 1k prompt |
|---|---:|---:|---:|---:|---:|---:|
| arm_full8 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0.0 |
| arm_ref | 0.4 | 0.035 | 0.913 | 3244 | 3130 | 0.006 |
| cat2 | 0.526 | 0.043 | 0.957 | 3453 | 3303 | 0.006 |
| cat4 | 0.526 | 0.045 | 0.957 | 3364 | 3214 | 0.007 |
| control | 0.281 | 0.076 | 0.826 | 1048 | 968 | 0.006 |
| validated5 | 0.512 | 0.04 | 0.826 | 3612 | 3466 | 0.009 |

## Control comparison verdicts

- `clean_recall_win`: 2
- `pure_bloat`: 59
- `recall_bought_with_bloat`: 8
- `retrieval_regression`: 28
- `same_or_cheaper`: 53

## Strongest recall wins vs control

| arm | question_id | gold types | recall delta | extra TP | extra FP | bloat / extra TP | question |
|---|---|---|---:|---:|---:|---:|---|
| cat2 | 7bb4dd3032dd60767f552577 | Medication,MedicationRequest | 0.873 | 69 | 11 | 0.2 | Has patient 10006580 had any medication since 13 months ago? |
| cat4 | 7bb4dd3032dd60767f552577 | Medication,MedicationRequest | 0.873 | 69 | 11 | 0.2 | Has patient 10006580 had any medication since 13 months ago? |
| validated5 | 7bb4dd3032dd60767f552577 | Medication,MedicationRequest | 0.873 | 69 | 11 | 0.2 | Has patient 10006580 had any medication since 13 months ago? |
| arm_ref | 0eba370644940aed20760324 | Medication,MedicationRequest | 0.5 | 1 | 29 | 29.0 | What was the name of the drug that patient 10006580 was first prescribed via a sc route in their first hospital visit? |
| cat2 | 0eba370644940aed20760324 | Medication,MedicationRequest | 0.5 | 1 | 40 | 40.0 | What was the name of the drug that patient 10006580 was first prescribed via a sc route in their first hospital visit? |
| arm_ref | 666f86f3566bfabb76e587a7 | Medication,MedicationRequest | 0.5 | 1 | 80 | 80.0 | When was patient 10038992 last prescribed a drug via the neb route since 07/2185? |
| cat4 | 666f86f3566bfabb76e587a7 | Medication,MedicationRequest | 0.5 | 1 | 80 | 80.0 | When was patient 10038992 last prescribed a drug via the neb route since 07/2185? |
| cat2 | 666f86f3566bfabb76e587a7 | Medication,MedicationRequest | 0.5 | 1 | 178 | 178.0 | When was patient 10038992 last prescribed a drug via the neb route since 07/2185? |

## Clean recall wins vs control

| arm | question_id | gold types | recall delta | extra TP | extra FP | question |
|---|---|---|---:|---:|---:|---|
| cat4 | 0eba370644940aed20760324 | Medication,MedicationRequest | 0.5 | 1 | 0 | What was the name of the drug that patient 10006580 was first prescribed via a sc route in their first hospital visit? |
| arm_ref | 7bb4dd3032dd60767f552577 | Medication,MedicationRequest | 0.417 | 33 | 9 | Has patient 10006580 had any medication since 13 months ago? |

## Worst bloat cases

| arm | verdict | question_id | extra retrieved | extra FP | recall delta | question |
|---|---|---|---:|---:|---:|---|
| cat4 | pure_bloat | e198cf387b6b22216a593d12 | 965 | 965 | 0.0 | Please show the total amount of output patient 10004720 had since 02/08/2183. |
| validated5 | pure_bloat | 20c3ddd36c65c44890c3f3e5 | 939 | 939 | 0.0 | What was the name of the specimen test that patient 10021666 was given for the first time since 03/2172? |
| arm_ref | pure_bloat | 56cc41f276a6e10711d396f4 | 482 | 482 | 0.0 | When was patient 10004720's first microbiology test done? |
| validated5 | pure_bloat | e198cf387b6b22216a593d12 | 469 | 469 | 0.0 | Please show the total amount of output patient 10004720 had since 02/08/2183. |
| cat2 | pure_bloat | 45b34eeae5d3c8d44b703194 | 340 | 340 | 0.0 | How much docusate sodium (liquid) dose has been prescribed to patient 10007818 on the first hospital encounter? |
| cat2 | pure_bloat | 73710b96b938a24a7a5c86d5 | 295 | 295 | 0.0 | Tell me the medication that patient 10014729 was prescribed for the first time? |
| cat4 | pure_bloat | 45b34eeae5d3c8d44b703194 | 292 | 292 | 0.0 | How much docusate sodium (liquid) dose has been prescribed to patient 10007818 on the first hospital encounter? |
| arm_ref | pure_bloat | 45b34eeae5d3c8d44b703194 | 290 | 290 | 0.0 | How much docusate sodium (liquid) dose has been prescribed to patient 10007818 on the first hospital encounter? |

## Interpretation

The useful question is not whether more tools improve final answers. The useful question is whether a strategy gets more gold FHIR IDs per token without dragging thousands of irrelevant resources into context.
