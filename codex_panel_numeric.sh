#!/bin/bash
cd /Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench
OUT=runs/full409/codex_votes_numeric; mkdir -p "$OUT"
PROMPT="$(cat runs/full409/codex_prompt.txt)"
run_one(){ p="$1"; b="$2"; f="$OUT/p${p}_b${b}.json"; [ -s "$f" ] && { echo skip; return; }
  codex exec -s read-only --skip-git-repo-check --ephemeral --output-schema runs/full409/codex_schema.json \
   --output-last-message "$f" "$PROMPT
(Independent grading pass ${p}. These golds are NUMERIC; a numeric answer matching the gold within rounding is correct, ignore units/formatting.)" \
   < "runs/full409/numeric_judge_batches/batch_${b}.json" >/dev/null 2>&1; echo "done p$p b$b"; }
export -f run_one; export OUT PROMPT
nb=$(ls runs/full409/numeric_judge_batches/batch_*.json | wc -l | tr -d ' ')
for p in 1 2 3; do for b in $(seq 0 $((nb-1))); do printf "%s %02d\n" "$p" "$b"; done; done | xargs -P 6 -n 2 bash -c 'run_one "$0" "$1"'
echo "NUMERIC CODEX PANEL COMPLETE"
