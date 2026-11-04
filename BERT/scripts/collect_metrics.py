#!/usr/bin/env python3
"""Collect loss, unchanged-repository UAS, and paper-equation UAS metrics."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


UAS_FILENAME = "repo_exact_depparse_uas.csv"
PAPER_UAS_FILENAME = "paper_equation_uas.csv"


def step_of(path):
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def write_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def latest_trainer_state(run_dir):
    root_state = run_dir / "trainer_state.json"
    candidates = [root_state, *run_dir.glob("checkpoint-*/trainer_state.json")]
    candidates = [path for path in candidates if path.exists()]
    # A root state can predate newer checkpoints after an interrupted resume.
    return max(candidates, key=lambda path: (
        int(json.loads(path.read_text()).get("global_step", 0)), path == root_state,
    )) if candidates else None


def collect_losses(run_dir):
    by_step = {}
    initial = run_dir / "initial_eval.json"
    if initial.exists():
        values = json.loads(initial.read_text())
        by_step[0] = {
            "step": 0,
            "train_loss_logged": math.nan,
            "test_loss": values.get("eval_loss", math.nan),
            "learning_rate": 0.0,
            "epoch": 0.0,
        }
    state_path = latest_trainer_state(run_dir)
    if state_path:
        state = json.loads(state_path.read_text())
        for event in state.get("log_history", []):
            if "step" not in event:
                continue
            step = int(event["step"])
            row = by_step.setdefault(
                step,
                {
                    "step": step,
                    "train_loss_logged": math.nan,
                    "test_loss": math.nan,
                    "learning_rate": math.nan,
                    "epoch": math.nan,
                },
            )
            if "loss" in event:
                row["train_loss_logged"] = event["loss"]
            if "eval_loss" in event:
                row["test_loss"] = event["eval_loss"]
            if "learning_rate" in event:
                row["learning_rate"] = event["learning_rate"]
            if "epoch" in event:
                row["epoch"] = event["epoch"]
    # Standalone callback records survive interruptions before trainer_state save.
    for path in run_dir.glob("checkpoint_metrics/step-*.json"):
        values = json.loads(path.read_text())
        step = int(values["step"])
        row = by_step.setdefault(step, {"step": step, "train_loss_logged": math.nan,
            "test_loss": math.nan, "learning_rate": math.nan, "epoch": math.nan})
        row["test_loss"] = values["validation_loss"]
        if values.get("epoch") is not None:
            row["epoch"] = values["epoch"]
    for row in by_step.values():
        row["validation_loss"] = row["test_loss"]  # historical test_loss is validation
    return [by_step[step] for step in sorted(by_step)]


def parse_upstream_row(line):
    fields = [value.strip() for value in line.rstrip("\n").split(",")]
    if len(fields) != 7:
        raise ValueError(f"Unexpected upstream UAS row: {line!r}")
    relation, count, uas, offset, baseline, layer_head, direction = fields
    layer, head = layer_head.split("-", maxsplit=1)
    return {
        "relation": relation,
        "n_edges": int(count),
        "uas": float(uas) / 100.0,
        "uas_percent": float(uas),
        "offset": int(offset.split("=", maxsplit=1)[1]),
        "offset_baseline_uas": float(baseline) / 100.0,
        "offset_baseline_percent": float(baseline),
        "layer": int(layer),
        "head": int(head),
        "direction": direction,
    }


def collect_uas(run_dir):
    global_rows = []
    relation_rows = []
    checkpoints = sorted(
        [path for path in run_dir.glob("checkpoint-*") if step_of(path) is not None],
        key=step_of,
    )
    for checkpoint in checkpoints:
        path = checkpoint / UAS_FILENAME
        if not path.exists():
            continue
        step = step_of(checkpoint)
        parsed = [
            parse_upstream_row(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        for row in parsed:
            output = {"step": step, **row, "raw_file": str(path)}
            if row["relation"] == "all":
                global_rows.append(output)
            else:
                relation_rows.append(output)
    return global_rows, relation_rows


def collect_paper_uas(run_dir):
    aggregate_rows = []
    relation_rows = []
    checkpoints = sorted(
        [path for path in run_dir.glob("checkpoint-*") if step_of(path) is not None],
        key=step_of,
    )
    integer_fields = {"n_edges", "n_correct"}
    float_fields = {"uas", "uas_percent"}
    optional_integer_fields = {"layer", "head"}
    for checkpoint in checkpoints:
        path = checkpoint / PAPER_UAS_FILENAME
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            parsed = list(csv.DictReader(handle))
        for row in parsed:
            for field in integer_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            for field in optional_integer_fields:
                row[field] = int(row[field]) if row[field] != "" else ""
            output = {"step": step_of(checkpoint), **row, "raw_file": str(path)}
            if row["row_type"] == "aggregate":
                aggregate_rows.append(output)
            elif row["row_type"] == "relation":
                relation_rows.append(output)
            else:
                raise ValueError(f"Unexpected paper UAS row type in {path}: {row}")
    return aggregate_rows, relation_rows


def save_best_checkpoint(run_dir, losses, global_uas, relation_uas, paper_global, paper_relations):
    """Select by validation alone, then attach UAS from that exact saved step."""
    eligible = []
    for row in losses:
        checkpoint = run_dir / f"checkpoint-{row['step']}"
        saved = (checkpoint / "config.json").exists() and any(
            (checkpoint / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"))
        if saved and math.isfinite(float(row["validation_loss"])):
            eligible.append(row)
    result = {"selection": "minimum finite validation loss over saved checkpoints; earlier step breaks ties",
              "loss_aggregation": "unchanged Hugging Face Trainer eval_loss",
              "test_loss_definition": "legacy alias of validation_loss, not a separate test set",
              "train_loss_logged_definition": "recent Trainer training log average; not full-dataset loss",
              "selected_step": None}
    if eligible:
        best = min(eligible, key=lambda row: (row["validation_loss"], row["step"]))
        step = best["step"]
        matching = [row for row in global_uas if row["step"] == step]
        train_logs = [row for row in losses if row["step"] <= step
                      and math.isfinite(float(row["train_loss_logged"]))]
        train_loss = max(train_logs, key=lambda row: row["step"])["train_loss_logged"] if train_logs else None
        result.update(selected_step=step, checkpoint_dir=str(run_dir / f"checkpoint-{step}"),
            validation_loss=best["validation_loss"], test_loss=best["test_loss"],
            train_loss_logged=train_loss,
            repo_uas_global=matching[0]["uas"] if matching else None,
            repo_uas_global_metadata=matching[0] if matching else None,
            repo_uas_relation=[row for row in relation_uas if row["step"] == step],
            paper_uas_aggregates=[row for row in paper_global if row["step"] == step],
            paper_uas_relation=[row for row in paper_relations if row["step"] == step])
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return None if isinstance(value, float) and not math.isfinite(value) else value
    path = run_dir / "metrics" / "best_checkpoint.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    losses = collect_losses(run_dir)
    global_uas, relation_uas = collect_uas(run_dir)
    paper_global_uas, paper_relation_uas = collect_paper_uas(run_dir)
    write_csv(
        metrics_dir / "mlm_loss_trajectory.csv",
        ["step", "train_loss_logged", "validation_loss", "test_loss", "learning_rate", "epoch"],
        losses,
    )
    uas_fields = [
        "step",
        "relation",
        "n_edges",
        "uas",
        "uas_percent",
        "offset",
        "offset_baseline_uas",
        "offset_baseline_percent",
        "layer",
        "head",
        "direction",
        "raw_file",
    ]
    write_csv(metrics_dir / "repo_uas_global.csv", uas_fields, global_uas)
    write_csv(metrics_dir / "repo_uas_relation.csv", uas_fields, relation_uas)
    paper_fields = [
        "step",
        "row_type",
        "scope",
        "relation",
        "n_edges",
        "n_correct",
        "uas",
        "uas_percent",
        "layer",
        "head",
        "prediction_rule",
        "raw_file",
    ]
    write_csv(
        metrics_dir / "paper_equation_uas_global.csv",
        paper_fields,
        paper_global_uas,
    )
    write_csv(
        metrics_dir / "paper_equation_uas_relation.csv",
        paper_fields,
        paper_relation_uas,
    )

    save_best_checkpoint(run_dir, losses, global_uas, relation_uas, paper_global_uas, paper_relation_uas)

    primary_paper_rows = [
        row for row in paper_global_uas if row["scope"] == "paper_all_relations"
    ]
    no_root_punct_rows = [
        row for row in paper_global_uas if row["scope"] == "paper_no_root_punct"
    ]

    summary = {
        "run_dir": str(run_dir),
        "loss_points": len(losses),
        "validation_loss_definition": "held-out fixed-mask MLM loss; test_loss is its legacy alias",
        "train_loss_logged_definition": "recent Trainer log average, not full-dataset training loss",
        "uas_checkpoints": len(global_uas),
        "relation_rows": len(relation_uas),
        "latest_repo_uas": global_uas[-1]["uas"] if global_uas else None,
        "latest_repo_uas_step": global_uas[-1]["step"] if global_uas else None,
        "paper_uas_checkpoints": len(primary_paper_rows),
        "paper_relation_rows": len(paper_relation_uas),
        "latest_paper_equation_uas": (
            primary_paper_rows[-1]["uas"] if primary_paper_rows else None
        ),
        "latest_paper_equation_uas_step": (
            primary_paper_rows[-1]["step"] if primary_paper_rows else None
        ),
        "latest_paper_equation_uas_no_root_punct": (
            no_root_punct_rows[-1]["uas"] if no_root_punct_rows else None
        ),
        "final_exact_train_loss": None,
        "final_test_loss": None,
    }
    train_eval_path = run_dir / "train_eval_results.json"
    eval_path = run_dir / "eval_results.json"
    if train_eval_path.exists():
        summary["final_exact_train_loss"] = json.loads(train_eval_path.read_text()).get(
            "train_eval_loss"
        )
    if eval_path.exists():
        summary["final_test_loss"] = json.loads(eval_path.read_text()).get("eval_loss")
    (metrics_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
