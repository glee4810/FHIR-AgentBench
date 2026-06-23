#!/bin/bash
# Full run: benchmark's STANDARD resource agent (no code) vs code agent (+execute_python_code), GPT-5.5,
# n=200 held-out test questions, against the Medplum-loaded MIMIC. The real lever experiment.
# Resumable: run_agent skips questions already in the output JSON, so re-run to continue an interrupted run.
set -e
cd /Users/aanishsachdev/Desktop/cstack/FHIR-AgentBench
export MEDPLUM_BASE_URL=http://localhost:8103
set -a; . ~/.codex/local-secrets/openai.env 2>/dev/null; set +a
M=gpt-5.5-2026-04-23
mkdir -p runs/full
N=$(python3 -c "import pandas as pd; print(len(pd.read_csv('final_dataset/full_test200.csv')))")  # actual rows (CSV fields contain newlines, so wc -l overcounts)

for strat in multi_turn_resource multi_turn_code_resource; do
  out="runs/full/${strat}.json"
  done_n=$(python3 -c "import json,os;print(len(json.load(open('$out'))) if os.path.exists('$out') else 0)" 2>/dev/null || echo 0)
  if [ "$done_n" -ge "$N" ]; then
    echo "===================== ARM: $strat (already complete: $done_n/$N, skipping) ====================="
    continue
  fi
  echo "===================== ARM: $strat ($done_n/$N done, running) ====================="
  python3 run_agent.py --model "$M" --agent_strategy "$strat" \
    --input final_dataset/full_test200.csv --output "$out" \
    --num_processes 1 2>&1 | grep -vE "CACHE (SAVE|HIT)|Tool execution|Agent detected" | tail -6
done

echo "===================== RESULTS: code-vs-resource (paired) ====================="
EVAL_JUDGE_MODEL=gpt-5-mini-2025-08-07 python3 - <<'PY'
import json, os, sys, math, random
sys.argv=[sys.argv[0],"--input","_unused"]
from evaluation_metrics import check_answer_correctness_with_llm
random.seed(0)
def judge(a,t):
    try: return int(check_answer_correctness_with_llm(a or "", t or "", "", model="gpt-5-mini-2025-08-07"))
    except Exception: return None
res={r['question_id']:r for r in json.load(open('runs/full/multi_turn_resource.json'))}
cod={r['question_id']:r for r in json.load(open('runs/full/multi_turn_code_resource.json'))}
ids=[q for q in res if q in cod]
rj={q:judge(res[q].get('agent_answer'),res[q].get('true_answer')) for q in ids}
cj={q:judge(cod[q].get('agent_answer'),cod[q].get('true_answer')) for q in ids}
paired=[q for q in ids if rj[q] in (0,1) and cj[q] in (0,1)]
ra=sum(rj[q] for q in paired); ca=sum(cj[q] for q in paired); n=len(paired)
b01=sum(1 for q in paired if rj[q]==0 and cj[q]==1)  # code fixed
b10=sum(1 for q in paired if rj[q]==1 and cj[q]==0)  # code broke
k=min(b01,b10); nd=b01+b10
p=min(1.0, 2.0*sum(math.comb(nd,i) for i in range(k+1))/(2**nd)) if nd else 1.0
rc=sum((res[q].get('usage') or {}).get('cost',0) or 0 for q in ids)
cc=sum((cod[q].get('usage') or {}).get('cost',0) or 0 for q in ids)
print(f"  n paired = {n}")
print(f"  resource (no code): {ra}/{n} = {ra/n:.3f}")
print(f"  code (+interp):     {ca}/{n} = {ca/n:.3f}   delta = {(ca-ra)/n:+.3f}")
print(f"  McNemar: code FIXED {b01}, code BROKE {b10}  -> exact p = {p:.4f}  {'(SIGNIFICANT)' if p<0.05 else '(n.s.)'}")
print(f"  cost: resource ${rc:.2f}  code ${cc:.2f}  total ${rc+cc:.2f}")
json.dump({"n":n,"resource_acc":ra/n,"code_acc":ca/n,"code_fixed":b01,"code_broke":b10,"mcnemar_p":p,
           "cost_resource":rc,"cost_code":cc}, open('runs/full/_summary.json','w'), indent=2)
PY
