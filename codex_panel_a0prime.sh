#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
OUT=runs/a0prime/codex_votes; PROMPT="$(cat runs/a0prime/codex_prompt.txt)"
mkdir -p "$OUT"
run_one(){ p="$1"; b="$2"; batch="runs/a0prime/judge_batches/batch_${b}.json"; f="$OUT/p${p}_b${b}.json"; [ -s "$f" ] && { echo skip; return; }
  tmp="$(mktemp "$OUT/.p${p}_b${b}.XXXXXX")"
  codex exec -s read-only --skip-git-repo-check --ephemeral --output-schema runs/a0prime/codex_schema.json \
    --output-last-message "$tmp" "$PROMPT
(grading pass ${p})" < "$batch" >/dev/null
  python3 - "$batch" "$tmp" <<'PY'
import json, sys
batch = json.load(open(sys.argv[1]))
out = json.load(open(sys.argv[2]))
expected = [row["qid"] for row in batch]
grades = out.get("grades")
if not isinstance(grades, list):
    raise SystemExit("missing grades array")
seen = [row.get("qid") for row in grades]
if seen != expected:
    raise SystemExit(f"qid mismatch: expected {len(expected)} exact ordered qids")
for row in grades:
    if row.get("label") not in (0, 1):
        raise SystemExit(f"bad label for {row.get('qid')}")
PY
  mv "$tmp" "$f"; echo "done p$p b$b"; }
export -f run_one; export OUT PROMPT
batches=(runs/a0prime/judge_batches/batch_*.json)
if [ ! -e "${batches[0]}" ]; then
  echo "missing runs/a0prime/judge_batches/*.json; restore the local/private judge batches before rerunning the panel" >&2
  exit 1
fi
nb=${#batches[@]}
for p in 1 2 3; do for b in $(seq 0 $((nb-1))); do printf "%s %02d\n" "$p" "$b"; done; done | xargs -P 6 -n 2 bash -c 'run_one "$0" "$1"'
echo "A0PRIME CODEX PANEL COMPLETE"
