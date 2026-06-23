#!/bin/bash
# Progress of the code-arm re-run (135 cleared questions). Just: bash rerun_progress.sh
cd /Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench
PY=/Users/aanishsachdev/miniconda3/bin/python3
rem=$($PY -c "
import json
try:
    d=json.load(open('runs/full409/multi_turn_code_resource.json'))
    print(sum(1 for r in d if r.get('agent_answer') is None))
except Exception:
    print('?')   # file mid-write; just re-run
" 2>/dev/null)
if [ "$rem" = "?" ]; then echo "re-run: (file mid-write, run again)"; else echo "re-run: $((135-rem))/135 done  ($rem left)"; fi
parent=$(pgrep -f "run_agent.py" 2>/dev/null | head -1)
if [ -n "$parent" ]; then
  w=$(ps -axo ppid,command | awk -v p="$parent" '$1==p && /multiprocessing.spawn/' | wc -l | tr -d ' ')
  echo "status: RUNNING (parent pid $parent + $w workers)"
else
  echo "status: STOPPED (done, or killed)"
fi
echo -n "RAM free: "; memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{print $2}' || echo "?"
echo "(if RAM gets painful: pkill -f 'run_agent.py' to stop; it's resumable)"
