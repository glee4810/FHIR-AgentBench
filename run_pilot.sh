#!/bin/bash
# Pilot: benchmark's STANDARD resource agent (no code) vs code agent (+execute_python_code), GPT-5.5,
# 6 representative test questions, against the Medplum-loaded MIMIC. Measures accuracy + $/question so
# we can project the full n~100 cost before committing the $80. Run AFTER the MIMIC load finishes.
set -e
cd /Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench
export MEDPLUM_BASE_URL=http://localhost:8103
set -a; . ~/.codex/local-secrets/openai.env 2>/dev/null; set +a
M=gpt-5.5-2026-04-23
mkdir -p runs/pilot
rm -rf .tool_cache       # drop cached errors / empty-server entries
rm -f runs/pilot/*.json  # fresh full run (run_agent resumes from existing output otherwise)

for strat in multi_turn_resource multi_turn_code_resource; do
  echo "===================== ARM: $strat ====================="
  python3 run_agent.py --model "$M" --agent_strategy "$strat" \
    --input final_dataset/pilot_test6.csv --output "runs/pilot/${strat}.json" \
    --num_processes 1 2>&1 | tail -8
done

echo "===================== COST + ACCURACY + $80 PROJECTION ====================="
EVAL_JUDGE_MODEL=gpt-5-mini-2025-08-07 python3 - <<'PY'
import json, os, sys
sys.argv=[sys.argv[0],"--input","_unused"]
from evaluation_metrics import check_answer_correctness_with_llm
def judge(a,t):
    try: return int(check_answer_correctness_with_llm(a or "", t or "", "", model="gpt-5-mini-2025-08-07"))
    except Exception: return None
rows=[]
for strat in ("multi_turn_resource","multi_turn_code_resource"):
    p=f"runs/pilot/{strat}.json"
    recs=json.load(open(p))
    cost=sum((r.get("usage") or {}).get("cost",0) or 0 for r in recs)
    correct=sum(1 for r in recs if judge(r.get("agent_answer"), r.get("true_answer"))==1)
    n=len(recs); cpq=cost/n if n else 0
    rows.append((strat,n,correct,cost,cpq))
    print(f"  {strat:28} n={n} correct={correct}/{n}  cost=${cost:.3f}  ${cpq:.3f}/q")
pair=sum(r[4] for r in rows)  # cost per question-pair (both arms)
print(f"\n  cost per question-PAIR (both arms): ${pair:.3f}")
print(f"  => $80 buys n ~ {int(80/pair) if pair else 0} questions/arm (both arms run)")
print(f"  => projected cost at n=100/arm: ${pair*100:.0f}")
PY
