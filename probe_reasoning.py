#!/usr/bin/env python3
"""Cheap, server-free probe: at FIXED retrieval, does higher reasoning effort fix GPT-5.5's answers?

Replays the exact FHIR context each agent already retrieved (from the committed trace), flattens it into
one prompt, and re-asks GPT-5.5 to answer at reasoning_effort = {medium, high}. Judges each with
gpt-5-mini. No Medplum server, no agent tool-loop -> pennies. Isolates the reasoning-effort effect from
retrieval (retrieval is held identical across both conditions).

Usage: set OPENAI_API_KEY, then: python probe_reasoning.py [arm] [n]
"""
import os, sys, json, ast
import litellm

litellm.suppress_debug_info = True
sys.argv_backup = sys.argv
sys.argv = [sys.argv[0], "--input", "_unused"]
from evaluation_metrics import check_answer_correctness_with_llm  # benchmark's exact judge
sys.argv = sys.argv_backup

AGENT = os.environ.get("PROBE_AGENT_MODEL", "gpt-5.5-2026-04-23")
JUDGE = os.environ.get("EVAL_JUDGE_MODEL", "gpt-5-mini-2025-08-07")
HARD_CAP = float(os.environ.get("PROBE_CAP", "6.0"))
CTX_CHAR_CAP = int(os.environ.get("PROBE_CTX_CHARS", "80000"))  # ~20k tokens; truncate giant contexts

spent = [0.0]


def call(model, messages, effort=None):
    kw = dict(model=model, messages=messages)
    if effort:
        kw["reasoning_effort"] = effort
    out = litellm.completion(**kw)
    c = 0.0
    hp = getattr(out, "_hidden_params", {}) or {}
    if hp.get("response_cost"):
        c = hp["response_cost"]
    spent[0] += c
    return out.choices[0].message.content or "", c


def flat_context(rec):
    """Concatenate the FHIR resources the agent retrieved (tool-turn contents) into one text block."""
    parts = []
    for t in rec.get("trace") or []:
        if isinstance(t, dict) and t.get("role") == "tool" and t.get("content"):
            parts.append(str(t["content"]))
    ctx = "\n\n".join(parts)
    return ctx[:CTX_CHAR_CAP], len(ctx) > CTX_CHAR_CAP


def judge(ans, true):
    try:
        return int(check_answer_correctness_with_llm(ans or "", true or "", "", model=JUDGE))
    except Exception as e:
        print("  judge err:", str(e)[:120], file=sys.stderr)
        return None


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "control"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    p = f"medplum-eval/results/mcp.{arm}.gpt-5.5-2026-04-23.rep.c100k.json"
    recs = json.load(open(p))[:n]
    jp = p[:-5] + ".judged.json"
    orig = {r["question_id"]: r.get("judged") for r in json.load(open(jp))} if os.path.exists(jp) else {}

    rows = []
    for i, rec in enumerate(recs, 1):
        if spent[0] > HARD_CAP:
            print(f"[BUDGET STOP] ${spent[0]:.2f} > ${HARD_CAP} after {i-1} questions", flush=True)
            break
        q = rec.get("question") or ""
        ta = rec.get("true_answer")
        ctx, trunc = flat_context(rec)
        sysmsg = ("You are a clinical data assistant. Answer the question using ONLY the FHIR resources "
                  "provided. Be concise and give the specific answer.")
        user = f"Question: {q}\n\nRetrieved FHIR resources:\n{ctx}"
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]
        row = {"qid": rec.get("question_id"), "q": q[:60], "trunc": trunc, "orig_judged": orig.get(rec.get("question_id"))}
        for eff in ("medium", "high"):
            try:
                ans, c = call(AGENT, msgs, effort=eff)
                row[f"ans_{eff}"] = ans[:200]
                row[f"judged_{eff}"] = judge(ans, ta)
                row[f"cost_{eff}"] = round(c, 4)
            except Exception as e:
                row[f"judged_{eff}"] = None
                row[f"err_{eff}"] = str(e)[:140]
        rows.append(row)
        print(f"  [{i}/{len(recs)}] med={row.get('judged_medium')} high={row.get('judged_high')} "
              f"orig={row['orig_judged']} ${spent[0]:.2f}  {row['q']}", flush=True)

    def acc(key):
        vals = [r[key] for r in rows if r.get(key) in (0, 1)]
        return (sum(vals) / len(vals), len(vals)) if vals else (float('nan'), 0)
    am, nm = acc("judged_medium")
    ah, nh = acc("judged_high")
    # paired flips on questions graded in BOTH
    both = [(r["judged_medium"], r["judged_high"]) for r in rows if r.get("judged_medium") in (0, 1) and r.get("judged_high") in (0, 1)]
    hi_fixed = sum(1 for m, h in both if m == 0 and h == 1)
    hi_broke = sum(1 for m, h in both if m == 1 and h == 0)
    print("\n" + "=" * 64)
    print(f"REASONING-EFFORT PROBE ({arm}, gpt-5.5, fixed retrieval)")
    print("=" * 64)
    print(f"  accuracy @ medium effort: {am:.3f} (n={nm})")
    print(f"  accuracy @ high   effort: {ah:.3f} (n={nh})")
    print(f"  paired (n={len(both)}): high FIXED {hi_fixed} that medium got wrong; high BROKE {hi_broke}")
    print(f"  cost: medium ${sum(r.get('cost_medium',0) for r in rows):.3f}  high ${sum(r.get('cost_high',0) for r in rows):.3f}  total ${spent[0]:.2f}")
    print(f"  (committed agent baseline @ default effort for this arm: see _scores.csv)")
    json.dump(rows, open(f"medplum-eval/probe_reasoning_{arm}.json", "w"), indent=2)
    print(f"  wrote medplum-eval/probe_reasoning_{arm}.json")


if __name__ == "__main__":
    main()
