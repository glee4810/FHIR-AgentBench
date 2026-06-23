#!/bin/bash
# Check progress of the 409-question full-test run. Just: bash progress409.sh
cd /Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench
PY=/Users/aanishsachdev/miniconda3/bin/python3
echo "time: $(date +%H:%M:%S)"
if pgrep -f "run_agent.py.*full409" >/dev/null; then
  echo "agent: RUNNING (pid $(pgrep -f 'run_agent.py.*full409' | tr '\n' ' '))"
else
  echo "agent: not running"
fi
for s in multi_turn_resource multi_turn_code_resource; do
  $PY -c "
import json,os
f='runs/full409/$s.json'
d=json.load(open(f)) if os.path.exists(f) else []
a=sum(1 for r in d if isinstance(r.get('agent_answer'),str) and r.get('agent_answer').strip())
bar=int(a/409*20)
print(f'  {\"$s\".replace(\"multi_turn_\",\"\"):16} [{chr(9608)*bar}{chr(9617)*(20-bar)}] {a}/409')
" 2>/dev/null || echo "  $s: (mid-write, retry)"
done
if [ -f runs/full409/SCORE_DONE ]; then
  echo "SCORING: DONE -> result:"
  $PY -c "import json;print(json.dumps(json.load(open('runs/full409/_summary.json')),indent=2))" 2>/dev/null
else
  echo "scoring: not started (runs after the code arm finishes)"
fi
