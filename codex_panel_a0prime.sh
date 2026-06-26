#!/bin/bash
cd /Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench
OUT=runs/a0prime/codex_votes; PROMPT="$(cat runs/a0prime/codex_prompt.txt)"
run_one(){ p="$1"; b="$2"; f="$OUT/p${p}_b${b}.json"; [ -s "$f" ] && { echo skip; return; }
  codex exec -s read-only --skip-git-repo-check --ephemeral --output-schema runs/a0prime/codex_schema.json \
    --output-last-message "$f" "$PROMPT
(grading pass ${p})" < "runs/a0prime/judge_batches/batch_${b}.json" >/dev/null 2>&1; echo "done p$p b$b"; }
export -f run_one; export OUT PROMPT
nb=$(ls runs/a0prime/judge_batches/batch_*.json | wc -l | tr -d ' ')
for p in 1 2 3; do for b in $(seq 0 $((nb-1))); do printf "%s %02d\n" "$p" "$b"; done; done | xargs -P 6 -n 2 bash -c 'run_one "$0" "$1"'
echo "A0PRIME CODEX PANEL COMPLETE"
