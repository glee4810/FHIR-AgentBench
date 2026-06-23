#!/bin/bash
# Independent codex judge panel: 3 passes x 13 batches over the SAME 188 non-numeric
# questions the Claude panel judged. A non-Claude (GPT/codex) cross-check that the
# trustworthy re-grade is judge-model-independent. Concurrency-capped.
cd /Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench
OUT=runs/full409/codex_votes
mkdir -p "$OUT"
PROMPT="$(cat runs/full409/codex_prompt.txt)"
run_one() {
  p="$1"; b="$2"
  f="$OUT/p${p}_b${b}.json"
  [ -s "$f" ] && { echo "skip p$p b$b (exists)"; return; }
  codex exec -s read-only --skip-git-repo-check --ephemeral \
    --output-schema runs/full409/codex_schema.json \
    --output-last-message "$f" \
    "$PROMPT
(Independent grading pass ${p}. Grade from scratch.)" \
    < "runs/full409/codex_batches/batch_${b}.json" >/dev/null 2>&1
  echo "done p$p b$b -> $(wc -c < "$f" 2>/dev/null) bytes"
}
export -f run_one
export OUT PROMPT
# build job list (pass batch)
for p in 1 2 3; do for b in 00 01 02 03 04 05 06 07 08 09 10 11 12; do echo "$p $b"; done; done \
  | xargs -P 6 -n 2 bash -c 'run_one "$0" "$1"'
echo "ALL CODEX JUDGE JOBS COMPLETE"
