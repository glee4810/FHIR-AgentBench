#!/usr/bin/env python3
"""Ablation matrix runner with a HARD $-budget cap.

One server, many arms. For each cell (surface x arm x model x seed x input_cap) this:
  1. (re)starts treatment_mcp_server.py advertising only that arm's TOOL_SUBSET,
  2. runs N sampled FHIR-AgentBench questions through the agent (MCP or $ai surface),
  3. charges every question's token usage to a shared BudgetLedger and HARD-STOPS at the cap,
  4. writes a per-cell results JSON scored by score_taxonomy.py (by-cause + answerable-set + CIs).

Cells are ordered CORE-FIRST so a budget stop still yields the headline result.
`--pilot K` runs K questions on the cheapest cell, projects the full-matrix cost, and exits — so a
real run is a decision, not a surprise. Scale up for more budget with --n / --cap / --seeds.

  # project cost first:
  EVAL_GPT_MODEL=gpt-5 python run_matrix.py --pilot 3 --n 40
  # then run capped:
  python run_matrix.py --n 40 --cap 100 --out-dir runs/tier1
"""
import os
import sys
import csv
import json
import time
import random
import socket
import threading
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# Point the agents at the LOCAL treatment server BEFORE importing them (module reads env at import).
os.environ.setdefault("MEDPLUM_MCP_URL", "http://127.0.0.1:8765/mcp")
TREAT_PORT = int(os.environ.get("TREATMENT_PORT", "8765"))

from eval_budget import BudgetLedger, BudgetExceeded, project  # noqa: E402
from utils.core_utils import curate_input_dataset, parse_outputs  # noqa: E402
import pandas as pd  # noqa: E402

OPUS = os.environ.get("EVAL_OPUS_MODEL", "claude-opus-4-8")
GPT = os.environ.get("EVAL_GPT_MODEL", "gpt-5")
RAISED_CAP = int(os.environ.get("EVAL_RAISED_CAP", "100000"))
STOCK_CAP = int(os.environ.get("EVAL_STOCK_CAP", "32000"))


def _cell(surface, arm, model, seed=0, cap=RAISED_CAP, sample="rep", tag=""):
    return {"surface": surface, "arm": arm, "model": model, "seed": seed, "cap": cap, "sample": sample,
            "tag": tag or f"{surface}.{arm}.{model.split('/')[-1]}.{sample}.c{cap//1000}k"}


def tier1_cells():
    """The full matrix behind REPORT.md. Two runs, selected by EVAL_RUN (default 'gpt'):

      EVAL_RUN=gpt   RUN-2: the GPT-5.5 nested dose-response staircase on the representative slice
                     (control -> 2 -> 4 -> 5 -> 6 -> 8 tools) + the orthogonal frugal-generic.
      EVAL_RUN=opus  RUN-1: the Opus medication-slice 3-arm decomposition + the 2x2 cap-factorial
                     (the headline cap effect) + a representative anchor.

    Run each into a SEPARATE --out-dir and score them separately: score_taxonomy.py keys cells by
    arm/sample/cap (NOT by model), so opus and gpt cells with the same arm/sample/cap must not share an
    out-dir. Cells are CORE-FIRST so a budget stop still yields the headline result. run_cell() skips a
    cell whose output JSON already exists, so use a FRESH --out-dir to re-run from scratch."""
    which = os.environ.get("EVAL_RUN", "gpt").lower()
    C = []
    if which == "opus":
        # RUN-1 (Opus): cap-factorial @32k FIRST (it produces the one robust effect), then the @100k
        # 3-arm decomposition (which also supplies the @100k half of the cap-factorial), then rep anchor.
        C.append(_cell("mcp", "control", OPUS, cap=STOCK_CAP, sample="med"))
        C.append(_cell("mcp", "arm_ref", OPUS, cap=STOCK_CAP, sample="med"))
        for arm in ["control", "control_include", "arm_ref"]:
            C.append(_cell("mcp", arm, OPUS, cap=RAISED_CAP, sample="med"))
        C.append(_cell("mcp", "control", OPUS, cap=RAISED_CAP, sample="rep"))
        C.append(_cell("mcp", "arm_full8", OPUS, cap=RAISED_CAP, sample="rep"))
    else:  # gpt (default; ~10x cheaper than opus)
        for arm in ["control", "cat2", "cat4", "validated5", "arm_ref", "arm_full8"]:
            C.append(_cell("mcp", arm, GPT, cap=RAISED_CAP, sample="rep"))
        C.append(_cell("mcp", "c0", GPT, cap=RAISED_CAP, sample="rep"))  # frugal-generic (_elements)
    # Paired McNemar + bootstrap come from score_taxonomy.py (arms share question_ids per sample).
    return C


def port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_server(arm):
    env = os.environ.copy()
    env["TOOL_SUBSET"] = arm
    log = open(f"/tmp/treatsrv_{arm}.log", "w")
    p = subprocess.Popen([sys.executable, "treatment_mcp_server.py"], env=env,
                         stdout=log, stderr=subprocess.STDOUT)
    for _ in range(80):
        if p.poll() is not None:
            raise RuntimeError(f"treatment server (arm={arm}) exited early; see /tmp/treatsrv_{arm}.log")
        if port_open("127.0.0.1", TREAT_PORT):
            time.sleep(1.5)  # let FastMCP finish binding the MCP route
            return p
        time.sleep(0.5)
    p.terminate()
    raise RuntimeError(f"treatment server (arm={arm}) did not open :{TREAT_PORT}")


def stop_server(p):
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
    # return only once the port is actually BINDABLE (no SO_REUSEADDR — matches what the next
    # server faces), so a TIME_WAIT lingerer can't make the next arm fail "address already in use".
    for _ in range(40):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", TREAT_PORT))
            s.close()
            time.sleep(0.3)
            return
        except OSError:
            time.sleep(0.5)


def make_agent(surface, model):
    if surface == "ai":
        from agent.ai_agent import AIAgent
        return AIAgent(model=model, verbose=False)
    from agent.mcp_agent import MCPAgent
    return MCPAgent(model=model, verbose=False)


def load_questions(input_csv, n, sample_seed, sample="rep"):
    df = pd.read_csv(input_csv)
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()  # held-out TEST split only (comparable to published figures)
    df = df[df["patient_fhir_id"].notnull() & df["true_answer"].notnull()].copy()
    if sample == "med" and "true_fhir_ids" in df.columns:
        # the MedicationRequest->Medication reference slice (the headline; ~17% of demand)
        df = df[df["true_fhir_ids"].astype(str).str.contains("Medication")].copy()
    rng = random.Random(sample_seed)
    idx = list(df.index)
    rng.shuffle(idx)
    df = df.loc[idx[:n]].reset_index(drop=True)
    df["question_with_context"] = curate_input_dataset(df, add_patient_fhir_id=True)
    return df


def run_cell(cell, df, ledger, out_dir):
    surface, arm, model, seed, cap = cell["surface"], cell["arm"], cell["model"], cell["seed"], cell["cap"]
    os.environ["MCP_INPUT_CAP"] = str(cap)
    os.environ["AI_MODEL"] = model if surface == "ai" else os.environ.get("AI_MODEL", GPT)
    out_path = os.path.join(out_dir, cell["tag"] + ".json")
    if os.path.exists(out_path):
        print(f"[skip] {cell['tag']} (exists)", flush=True)
        return "skipped"
    workers = int(os.environ.get("EVAL_WORKERS", "6"))  # I/O-bound; thread-safe token/budget/asyncio
    srv = start_server(arm)
    records = []
    stopped = [None]
    lock = threading.Lock()

    def do_q(row):
        agent = make_agent(surface, model)  # fresh per question -> clean per-question usage
        try:
            raw = agent.run(row["question_with_context"])  # single attempt: no retry-reuse double-count
            out = parse_outputs(raw)
        except Exception as e:
            out = {"agent_answer": f"Error: {e}", "agent_fhir_resources": None,
                   "trace": [], "usage": None, "error": str(e)}
        return row, out

    try:
        rows = [row for _, row in df.iterrows()]
        rows_it = iter(rows)
        # Bounded submission: keep at most `workers` questions in flight and stop submitting NEW ones
        # the moment the budget trips. So a cap crossing halts the cell after the in-flight batch
        # (<= EVAL_WORKERS questions) finishes, instead of letting the whole cell run to completion.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            pending = set()
            for _ in range(workers):
                nxt = next(rows_it, None)
                if nxt is None:
                    break
                pending.add(ex.submit(do_q, nxt))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    row, out = fut.result()
                    usage = out.get("usage") or {}
                    rec = {
                        "question_id": row.get("question_id"),
                        "question": row.get("question"),
                        "true_answer": row.get("true_answer"),
                        "true_fhir_ids": row.get("true_fhir_ids"),
                        "patient_fhir_id": row.get("patient_fhir_id"),
                        "agent_answer": out.get("agent_answer"),
                        "agent_fhir_resources": out.get("agent_fhir_resources"),
                        "trace": out.get("trace"),  # tool-call trace -> the before/after artifact
                        "error": out.get("error"),
                        "usage": usage,
                        "_cell": cell["tag"],
                    }
                    with lock:
                        records.append(rec)
                        try:
                            ledger.charge_usage(model, usage)
                        except BudgetExceeded as e:
                            stopped[0] = f"budget: {e}"  # stop submitting new questions immediately
                        if len(records) % 5 == 0:
                            _save(out_path, records)
                            print(f"  [{cell['tag']}] {len(records)}/{len(rows)} spent=${ledger.spent:.2f}", flush=True)
                # only backfill new questions while under budget
                if stopped[0] is None:
                    for _ in range(len(done)):
                        nxt = next(rows_it, None)
                        if nxt is None:
                            break
                        pending.add(ex.submit(do_q, nxt))
    finally:
        stop_server(srv)
    _save(out_path, records)
    print(f"[cell done] {cell['tag']} n={len(records)} spent=${ledger.spent:.2f}"
          + (f" STOPPED({stopped[0]})" if stopped[0] else ""), flush=True)
    return stopped[0] or "ok"


def _save(path, records):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(records, f, default=str, indent=2)
    os.rename(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="final_dataset/questions_answers_sql_fhir.csv")
    ap.add_argument("--n", type=int, default=25, help="representative-anchor questions per cell")
    ap.add_argument("--n-med", type=int, default=40, help="medication-slice questions per cell (the headline)")
    ap.add_argument("--cap", type=float, default=100.0, help="hard $ budget for the whole matrix")
    ap.add_argument("--sample-seed", type=int, default=20260621)
    ap.add_argument("--out-dir", default="runs/tier1")
    ap.add_argument("--pilot", type=int, default=0, help="run K questions on the cheapest cell, project, exit")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples = {
        "rep": load_questions(args.input, args.n, args.sample_seed, sample="rep"),
        "med": load_questions(args.input, args.n_med, args.sample_seed, sample="med"),
    }
    print(f"[sample] rep={len(samples['rep'])} med={len(samples['med'])} (held-out test split, seed={args.sample_seed})", flush=True)

    if args.pilot:
        ledger = BudgetLedger(cap_usd=9999)
        cheap = _cell("mcp", "control", GPT, cap=RAISED_CAP, sample="rep", tag="PILOT")
        run_cell(cheap, samples["rep"].head(args.pilot), ledger, args.out_dir)
        cpq = ledger.spent / max(1, args.pilot)
        n_cells = len(tier1_cells())
        project(cpq, args.n_med, n_cells, label="tier1 (rough; opus cells cost ~4x this GPT pilot)")
        print(f"[pilot] ${cpq:.4f}/q on {GPT}; budget cap would bind at ${args.cap}", flush=True)
        return

    ledger = BudgetLedger(cap_usd=args.cap)
    cells = tier1_cells()
    print(f"[matrix] {len(cells)} cells (med={args.n_med}, rep={args.n}), hard cap ${args.cap}", flush=True)
    summary = []
    for cell in cells:
        status = run_cell(cell, samples[cell["sample"]], ledger, args.out_dir)
        summary.append({"cell": cell["tag"], "status": status, "spent": round(ledger.spent, 2)})
        if isinstance(status, str) and status.startswith("budget"):
            print(f"[BUDGET STOP] at {cell['tag']}: {status}", flush=True)
            break
    _save(os.path.join(args.out_dir, "_summary.json"), summary)
    print("[done]", json.dumps(ledger.report(), indent=2), flush=True)


if __name__ == "__main__":
    main()
