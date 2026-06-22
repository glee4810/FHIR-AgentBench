#!/usr/bin/env python3
"""No-token retrieval quality scorer for saved FHIR-AgentBench runs.

This scores the part of the experiment that does not need an LLM judge:
did the agent retrieve the gold FHIR resource IDs, how much extra did it pull,
and how large was the saved trace/token footprint?

Usage:
  python retrieval_quality.py medplum-eval/results
  python retrieval_quality.py medplum-eval/results --out-dir medplum-eval/retrieval-metrics
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


INFRA_NEEDLES = ("rate limit", "ratelimit", "quota", "429", "input tokens exceeded", "context")


def parse_resource_map(value: Any) -> dict[str, set[str]]:
    """Normalize {'Observation': ['id']} from strings or dicts into sets."""
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return {}
    if not isinstance(value, dict):
        return {}
    out: dict[str, set[str]] = {}
    for resource_type, ids in value.items():
        if ids is None:
            continue
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, (list, tuple, set)):
            continue
        clean = {str(i) for i in ids if i is not None and str(i)}
        if clean:
            out[str(resource_type)] = clean
    return out


def flatten(resource_map: dict[str, set[str]]) -> set[tuple[str, str]]:
    return {(resource_type, resource_id) for resource_type, ids in resource_map.items() for resource_id in ids}


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round((pct / 100) * (len(vals) - 1))))
    return vals[idx]


def classify_failure(record: dict[str, Any]) -> str:
    blob = f"{record.get('error') or ''} {record.get('agent_answer') or ''}".lower()
    if not blob.strip():
        return "answered"
    if any(needle in blob for needle in INFRA_NEEDLES):
        return "infra"
    if "error:" in blob or blob.startswith("error"):
        return "harness"
    return "answered"


def trace_content_chars(record: dict[str, Any]) -> int:
    total = 0
    for turn in record.get("trace") or []:
        if isinstance(turn, dict):
            total += len(str(turn.get("content") or ""))
    return total


def trace_tool_calls(record: dict[str, Any]) -> int:
    calls = 0
    for turn in record.get("trace") or []:
        if isinstance(turn, dict) and turn.get("role") == "tool":
            calls += 1
    return calls


def record_metrics(arm: str, record: dict[str, Any]) -> dict[str, Any]:
    gold_map = parse_resource_map(record.get("true_fhir_ids"))
    retrieved_map = parse_resource_map(record.get("agent_fhir_resources"))
    gold = flatten(gold_map)
    retrieved = flatten(retrieved_map)
    tp_set = gold & retrieved
    fp_set = retrieved - gold
    fn_set = gold - retrieved
    usage = record.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0
    total_tokens = int(usage.get("total_tokens") or 0) if isinstance(usage, dict) else 0
    precision = len(tp_set) / len(retrieved) if retrieved else 0.0
    recall = len(tp_set) / len(gold) if gold else 0.0
    return {
        "arm": arm,
        "question_id": record.get("question_id"),
        "failure": classify_failure(record),
        "gold_ids": len(gold),
        "retrieved_ids": len(retrieved),
        "tp": len(tp_set),
        "fp": len(fp_set),
        "fn": len(fn_set),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "gold_covered": int(bool(gold) and gold <= retrieved),
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "tool_calls": trace_tool_calls(record),
        "trace_chars": trace_content_chars(record),
        "gold_types": ",".join(sorted(gold_map.keys())),
        "retrieved_types": ",".join(sorted(retrieved_map.keys())),
        "question": record.get("question"),
    }


@dataclass
class ArmStats:
    arm: str
    n: int = 0
    infra_failures: int = 0
    harness_failures: int = 0
    gold_total: int = 0
    retrieved_total: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    exact_gold_set: int = 0
    no_gold_rows: int = 0
    prompt_tokens: list[int] = field(default_factory=list)
    total_tokens: list[int] = field(default_factory=list)
    trace_chars: list[int] = field(default_factory=list)
    tool_calls: list[int] = field(default_factory=list)
    by_type_gold: Counter[str] = field(default_factory=Counter)
    by_type_tp: Counter[str] = field(default_factory=Counter)

    def add(self, record: dict[str, Any]) -> None:
        self.n += 1
        failure = classify_failure(record)
        if failure == "infra":
            self.infra_failures += 1
        elif failure == "harness":
            self.harness_failures += 1

        gold_map = parse_resource_map(record.get("true_fhir_ids"))
        retrieved_map = parse_resource_map(record.get("agent_fhir_resources"))
        gold = flatten(gold_map)
        retrieved = flatten(retrieved_map)

        if not gold:
            self.no_gold_rows += 1

        tp_set = gold & retrieved
        fp_set = retrieved - gold
        fn_set = gold - retrieved

        self.gold_total += len(gold)
        self.retrieved_total += len(retrieved)
        self.true_positive += len(tp_set)
        self.false_positive += len(fp_set)
        self.false_negative += len(fn_set)
        if gold and gold <= retrieved:
            self.exact_gold_set += 1

        for resource_type, _ in gold:
            self.by_type_gold[resource_type] += 1
        for resource_type, _ in tp_set:
            self.by_type_tp[resource_type] += 1

        usage = record.get("usage") or {}
        if isinstance(usage, dict):
            self.prompt_tokens.append(int(usage.get("prompt_tokens") or 0))
            self.total_tokens.append(int(usage.get("total_tokens") or 0))
        self.trace_chars.append(trace_content_chars(record))
        self.tool_calls.append(trace_tool_calls(record))

    def row(self) -> dict[str, Any]:
        precision = self.true_positive / self.retrieved_total if self.retrieved_total else 0.0
        recall = self.true_positive / self.gold_total if self.gold_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows_with_gold = max(1, self.n - self.no_gold_rows)
        avg_prompt = statistics.mean(self.prompt_tokens) if self.prompt_tokens else 0.0
        return {
            "arm": self.arm,
            "n": self.n,
            "infra_failures": self.infra_failures,
            "harness_failures": self.harness_failures,
            "gold_ids": self.gold_total,
            "retrieved_ids": self.retrieved_total,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "micro_precision": round(precision, 3),
            "micro_recall": round(recall, 3),
            "micro_f1": round(f1, 3),
            "gold_covered_rows": round(self.exact_gold_set / rows_with_gold, 3),
            "avg_prompt_tokens": round(avg_prompt, 1),
            "p95_prompt_tokens": percentile(self.prompt_tokens, 95),
            "avg_tool_calls": round(statistics.mean(self.tool_calls), 2) if self.tool_calls else 0,
            "avg_trace_chars": round(statistics.mean(self.trace_chars), 1) if self.trace_chars else 0,
            "p95_trace_chars": percentile(self.trace_chars, 95),
            "recall_per_1k_prompt": round((recall / avg_prompt) * 1000, 3) if avg_prompt else 0.0,
        }

    def type_rows(self) -> list[dict[str, Any]]:
        rows = []
        for resource_type, gold_count in sorted(self.by_type_gold.items()):
            tp = self.by_type_tp.get(resource_type, 0)
            rows.append({
                "arm": self.arm,
                "resource_type": resource_type,
                "gold_ids": gold_count,
                "tp": tp,
                "recall": round(tp / gold_count, 3) if gold_count else 0.0,
            })
        return rows


def arm_name(path: Path) -> str:
    # mcp.control.gpt-5.5-2026-04-23.rep.c100k.json -> control
    parts = path.name.split(".")
    return parts[1] if len(parts) > 2 else path.stem


def compare_to_control(question_rows: list[dict[str, Any]], baseline: str = "control") -> list[dict[str, Any]]:
    by_q: dict[str, dict[str, dict[str, Any]]] = {}
    for row in question_rows:
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        by_q.setdefault(qid, {})[row["arm"]] = row

    comparisons: list[dict[str, Any]] = []
    for qid, arms in sorted(by_q.items()):
        base = arms.get(baseline)
        if not base:
            continue
        for arm, row in sorted(arms.items()):
            if arm == baseline:
                continue
            extra_tp = row["tp"] - base["tp"]
            extra_fp = row["fp"] - base["fp"]
            extra_retrieved = row["retrieved_ids"] - base["retrieved_ids"]
            recall_delta = row["recall"] - base["recall"]
            precision_delta = row["precision"] - base["precision"]
            if recall_delta > 0 and extra_tp > 0 and extra_fp <= 10:
                verdict = "clean_recall_win"
            elif recall_delta > 0 and extra_tp > 0:
                verdict = "recall_bought_with_bloat"
            elif recall_delta == 0 and extra_retrieved > 0:
                verdict = "pure_bloat"
            elif recall_delta < 0:
                verdict = "retrieval_regression"
            else:
                verdict = "same_or_cheaper"
            comparisons.append({
                "question_id": qid,
                "arm": arm,
                "verdict": verdict,
                "gold_ids": row["gold_ids"],
                "control_recall": base["recall"],
                "arm_recall": row["recall"],
                "recall_delta": round(recall_delta, 3),
                "control_precision": base["precision"],
                "arm_precision": row["precision"],
                "precision_delta": round(precision_delta, 3),
                "extra_tp": extra_tp,
                "extra_fp": extra_fp,
                "extra_retrieved": extra_retrieved,
                "prompt_delta": row["prompt_tokens"] - base["prompt_tokens"],
                "bloat_per_extra_tp": round(extra_fp / extra_tp, 1) if extra_tp > 0 else "",
                "gold_types": row["gold_types"],
                "question": row["question"],
            })
    return comparisons


def load_stats(results_dir: Path) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    by_arm: dict[str, ArmStats] = {}
    question_rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("_") or path.name.endswith(".judged.json"):
            continue  # skip aggregates + per-question judge-label sidecars
        arm = arm_name(path)
        stats = by_arm.setdefault(arm, ArmStats(arm=arm))
        with path.open() as f:
            for record in json.load(f):
                stats.add(record)
                question_rows.append(record_metrics(arm, record))
    rows = [stats.row() for _, stats in sorted(by_arm.items())]
    type_rows = [row for _, stats in sorted(by_arm.items()) for row in stats.type_rows()]
    comparisons = compare_to_control(question_rows)
    return rows, type_rows, question_rows, comparisons


def print_markdown(rows: list[dict[str, Any]]) -> None:
    cols = [
        "arm", "n", "micro_recall", "micro_precision", "gold_covered_rows",
        "retrieved_ids", "fp", "infra_failures", "avg_prompt_tokens",
        "p95_prompt_tokens", "recall_per_1k_prompt",
    ]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[c]) for c in cols) + " |")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    def top(verdict: str, limit: int = 8) -> list[dict[str, Any]]:
        rows = [r for r in comparisons if r["verdict"] == verdict]
        return sorted(rows, key=lambda r: (r["recall_delta"], -r["extra_fp"]), reverse=True)[:limit]

    def worst_bloat(limit: int = 8) -> list[dict[str, Any]]:
        rows = [r for r in comparisons if r["verdict"] in {"pure_bloat", "recall_bought_with_bloat"}]
        return sorted(rows, key=lambda r: (r["extra_fp"], r["extra_retrieved"]), reverse=True)[:limit]

    verdict_counts = Counter(r["verdict"] for r in comparisons)
    lines = [
        "# Retrieval quality report",
        "",
        "No model calls were made. This report compares saved `agent_fhir_resources` against `true_fhir_ids`.",
        "",
        "## Arm summary",
        "",
        "| arm | recall | precision | gold rows covered | retrieved IDs | false positives | recall / 1k prompt |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['arm']} | {row['micro_recall']} | {row['micro_precision']} | "
            f"{row['gold_covered_rows']} | {row['retrieved_ids']} | {row['fp']} | "
            f"{row['recall_per_1k_prompt']} |"
        )

    lines.extend([
        "",
        "## Control comparison verdicts",
        "",
    ])
    for verdict, count in sorted(verdict_counts.items()):
        lines.append(f"- `{verdict}`: {count}")

    lines.extend([
        "",
        "## Strongest recall wins vs control",
        "",
        "| arm | question_id | gold types | recall delta | extra TP | extra FP | bloat / extra TP | question |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ])
    for row in top("recall_bought_with_bloat"):
        q = str(row["question"]).replace("|", "\\|")
        lines.append(
            f"| {row['arm']} | {row['question_id']} | {row['gold_types']} | {row['recall_delta']} | "
            f"{row['extra_tp']} | {row['extra_fp']} | {row['bloat_per_extra_tp']} | {q} |"
        )

    clean = top("clean_recall_win")
    if clean:
        lines.extend([
            "",
            "## Clean recall wins vs control",
            "",
            "| arm | question_id | gold types | recall delta | extra TP | extra FP | question |",
            "|---|---|---|---:|---:|---:|---|",
        ])
        for row in clean:
            q = str(row["question"]).replace("|", "\\|")
            lines.append(
                f"| {row['arm']} | {row['question_id']} | {row['gold_types']} | {row['recall_delta']} | "
                f"{row['extra_tp']} | {row['extra_fp']} | {q} |"
            )

    lines.extend([
        "",
        "## Worst bloat cases",
        "",
        "| arm | verdict | question_id | extra retrieved | extra FP | recall delta | question |",
        "|---|---|---|---:|---:|---:|---|",
    ])
    for row in worst_bloat():
        q = str(row["question"]).replace("|", "\\|")
        lines.append(
            f"| {row['arm']} | {row['verdict']} | {row['question_id']} | {row['extra_retrieved']} | "
            f"{row['extra_fp']} | {row['recall_delta']} | {q} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The useful question is not whether more tools improve final answers. The useful question is whether a strategy gets more gold FHIR IDs per token without dragging thousands of irrelevant resources into context.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="medplum-eval/results")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    rows, type_rows, question_rows, comparisons = load_stats(Path(args.results_dir))
    print_markdown(rows)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        write_csv(out_dir / "retrieval_summary.csv", rows)
        write_csv(out_dir / "retrieval_by_type.csv", type_rows)
        write_csv(out_dir / "retrieval_per_question.csv", question_rows)
        write_csv(out_dir / "retrieval_vs_control.csv", comparisons)
        write_report(out_dir / "retrieval_report.md", rows, comparisons)
        with (out_dir / "retrieval_summary.json").open("w") as f:
            json.dump({
                "summary": rows,
                "by_type": type_rows,
                "per_question": question_rows,
                "vs_control": comparisons,
            }, f, indent=2)
        print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
