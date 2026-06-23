#!/bin/bash
# Full 409-question test-set run (resumes from the seeded n=200 -> ~209 new per arm). Detach-safe:
# uses the EXPLICIT interpreter (plain python3 = homebrew, lacks deps) so it works under nohup.
# Launch: set -a; . ~/.codex/local-secrets/openai.env; set +a
#         nohup bash run_409.sh > runs/full409/run.log 2>&1 < /dev/null & disown
set -e
cd /Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench
export MEDPLUM_BASE_URL=http://localhost:8103
PY=/Users/aanishsachdev/miniconda3/bin/python3
M=gpt-5.5-2026-04-23
mkdir -p runs/full409
N=$($PY -c "import pandas as pd; print(len(pd.read_csv('final_dataset/full_test409.csv')))")

for strat in multi_turn_resource multi_turn_code_resource; do
  out="runs/full409/${strat}.json"
  # Count ANSWERED rows (non-null agent_answer), NOT total rows — run_agent pre-sizes the frame to all
  # 409 rows, so len(json) is always 409 and would wrongly skip a half-done arm on restart. This mirrors
  # run_agent.py's own resume logic (results_df.loc[agent_answer.isnull()]), so a relaunch resumes cleanly.
  done_n=$($PY -c "import json,os
d=json.load(open('$out')) if os.path.exists('$out') else []
print(sum(1 for r in d if isinstance(r.get('agent_answer'),str) and r.get('agent_answer').strip()))" 2>/dev/null || echo 0)
  if [ "$done_n" -ge "$N" ]; then
    echo "ARM $strat complete ($done_n/$N), skipping"; continue
  fi
  echo "ARM $strat ($done_n/$N done, running)"
  $PY run_agent.py --model "$M" --agent_strategy "$strat" \
    --input final_dataset/full_test409.csv --output "$out" --num_processes 1 2>&1 \
    | grep -vE "CACHE|Tool execution|Agent detected" | tail -4
done

echo "=== scoring 409 ==="
RUN_DIR=runs/full409 $PY score_full.py
echo "409 RUN COMPLETE"
