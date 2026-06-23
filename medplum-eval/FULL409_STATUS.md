# Full 409-question test-set run — status (judge quota-blocked, recoverable)

*2026-06-23. The full-test-set extension of the n=200 code-vs-resource experiment (`../CODE_EXPERIMENT.md`).*

## TL;DR

The committed result remains **n=200** (+6.5pp, McNemar **p=0.0106**, significant). The full 409-question
run **completed its agent passes** but the LLM judge ran out of OpenAI quota mid-scoring, so only 200 of the
409 questions are validly scored. **Do not cite `runs/full409/_summary.json` as a 409 result** — its numbers
are the n=200 numbers (n_paired=200) mislabeled `n_total=409`. The 409 score is one cheap command away once
OpenAI billing is restored.

## What ran

- Both arms (`multi_turn_resource` vs `multi_turn_code_resource`), GPT-5.5, against MIMIC-IV-on-FHIR demo
  loaded into self-hosted Medplum, on `../final_dataset/full_test409.csv` (all 409 test questions).
- **Agent passes finished:** resource **409/409** answered, code **390/409** answered.
- **Agent cost ≈ $34.38** (resource $11.63 + code $22.75).

## What's blocked

The judge (`gpt-5-mini-2025-08-07`) hit `RateLimitError: You exceeded your current quota` partway through
scoring. The scorer caches each label, and the 200 labels carried over from the n=200 run were valid, so
those scored; **all ~209 new questions × 2 arms = 418 labels erred to `None`**. Hence `n_paired=200`.

This is an **OpenAI account billing/quota** problem, not a code or data problem. The same judge function
scored the original 200 fine.

## Recovery (one command, <$1)

The agent answers are **durably backed up** in `full409_answers.json` (153 KB — qid + agent_answer +
true_answer for both arms, 409×2). This survives `runs/` being cleared. To finish the score once OpenAI
billing is restored (or with a funded key):

```bash
cd ..            # FHIR-AgentBench/
set -a; . ~/.codex/local-secrets/openai.env; set +a    # funded OPENAI_API_KEY
/Users/aanishsachdev/miniconda3/bin/python3 rejudge_409.py
```

`rejudge_409.py` reads the backup, **reuses the 400 valid cached labels**, re-judges only the 418 that erred,
and writes the real `runs/full409/_summary.json` + prints the n=409 numbers. It must stay `gpt-5-mini` (the
cached 400 are gpt-5-mini; switching judges would break the paired comparison).

## Expected outcome

With 409 paired questions the result will land **very close to n=200** — the conclusion does not change
(code interpreter wins, significant); the 409 just tightens the estimate and gives the full-test-set number.

## Files

- `full409_answers.json` — durable agent-answer backup (the re-judge input)
- `../rejudge_409.py` — the recovery scorer (judge-only, resumable)
- `../run_409.sh` — the run launcher (resume hardened: skip-check counts answered rows, not total)
- `../progress409.sh` — progress checker (`bash progress409.sh`)
- `runs/full409/_summary.json` — **stale** (n=200 mislabeled); overwritten by `rejudge_409.py`
