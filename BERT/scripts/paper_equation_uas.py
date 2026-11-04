#!/usr/bin/env python3
"""Evaluate UAS using the metric stated in Appendix A, Equations 3--5.

This intentionally lives outside ``sudden-drops-in-the-loss-main``.  The
upstream repository evaluator remains byte-for-byte unchanged.  Both
evaluators consume the same word-level attention pickle, so their difference
is scoring rather than model inference or token-to-word alignment.

Equation 3 predicts a parent for word i with

    argmax_j max(alpha[i, j], alpha[j, i]).

Equation 4 selects the best layer/head separately for every dependency
relation.  Equation 5 pools the correct predictions from those independently
selected relation probes into one micro-averaged UAS.
"""

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_ATTENTION_FILENAME = "repo_exact_wsj_attention.pkl"
DEFAULT_OUTPUT_FILENAME = "paper_equation_uas.csv"
DEFAULT_MANIFEST_FILENAME = "paper_equation_uas_manifest.json"


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def as_numpy(value):
    """Convert NumPy arrays or CPU/GPU torch tensors without requiring torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def equation3_predictions(attentions):
    """Return parent predictions with shape [layers, heads, words]."""
    attentions = as_numpy(attentions)
    if attentions.ndim != 4:
        raise ValueError(
            "Expected attention shape [layers, heads, words, words], got "
            f"{attentions.shape}"
        )
    if attentions.shape[-1] != attentions.shape[-2]:
        raise ValueError(f"Attention maps are not square: {attentions.shape}")

    # Paper Eq. 3: combine the two directions for every unordered word pair.
    scores = np.maximum(attentions, np.swapaxes(attentions, -1, -2)).copy()

    # Match the repository probe's candidate set: no self, [CLS], or [SEP].
    diagonal = np.arange(scores.shape[-1])
    scores[..., diagonal, diagonal] = -np.inf
    scores = scores[..., 1:-1, 1:-1]
    if scores.shape[-1] == 0:
        return np.empty((*scores.shape[:2], 0), dtype=np.int64)
    return np.argmax(scores, axis=-1).astype(np.int64) + 1


def accumulate_relation_scores(data):
    """Accumulate correct counts [layer, head] for every scored relation."""
    correct_by_relation = {}
    relation_counts = Counter()
    shape = None

    for example_index, example in enumerate(data):
        predictions = equation3_predictions(example["attns"])
        current_shape = predictions.shape[:2]
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise ValueError(
                f"Inconsistent layer/head shape at example {example_index}: "
                f"expected {shape}, got {current_shape}"
            )

        # Deliberately mirror upstream evaluate_predictor's zip behavior for
        # examples truncated by the model context length.
        scored_words = min(
            predictions.shape[-1], len(example["heads"]), len(example["relns"])
        )
        for word_index in range(scored_words):
            relation = str(example["relns"][word_index])
            gold_parent = int(example["heads"][word_index])
            if relation not in correct_by_relation:
                correct_by_relation[relation] = np.zeros(shape, dtype=np.int64)
            correct_by_relation[relation] += (
                predictions[..., word_index] == gold_parent
            )
            relation_counts[relation] += 1

    if shape is None:
        raise ValueError("Attention pickle contains no examples")
    return correct_by_relation, relation_counts, shape


def best_relation_rows(correct_by_relation, relation_counts):
    rows = []
    for relation in sorted(relation_counts):
        correct = correct_by_relation[relation]
        flat_index = int(np.argmax(correct))
        layer, head = np.unravel_index(flat_index, correct.shape)
        n_correct = int(correct[layer, head])
        n_edges = int(relation_counts[relation])
        rows.append(
            {
                "row_type": "relation",
                "scope": "relation",
                "relation": relation,
                "n_edges": n_edges,
                "n_correct": n_correct,
                "uas": n_correct / n_edges,
                "uas_percent": 100.0 * n_correct / n_edges,
                "layer": int(layer),
                "head": int(head),
                "prediction_rule": "max_forward_reverse_then_argmax",
            }
        )
    return rows


def aggregate_row(scope, relation_rows, keep_relation):
    selected = [row for row in relation_rows if keep_relation(row)]
    n_edges = sum(row["n_edges"] for row in selected)
    n_correct = sum(row["n_correct"] for row in selected)
    uas = n_correct / n_edges if n_edges else float("nan")
    return {
        "row_type": "aggregate",
        "scope": scope,
        "relation": "",
        "n_edges": n_edges,
        "n_correct": n_correct,
        "uas": uas,
        "uas_percent": 100.0 * uas,
        "layer": "",
        "head": "",
        "prediction_rule": "relation_selected_micro_average",
    }


def evaluate(data):
    correct, counts, shape = accumulate_relation_scores(data)
    relation_rows = best_relation_rows(correct, counts)
    aggregate_rows = [
        # Literal reading of Eqs. 4--5: the set of all dependency relations.
        aggregate_row("paper_all_relations", relation_rows, lambda row: True),
        # Common UAS reporting convention, useful as a sensitivity check.
        aggregate_row(
            "paper_no_root_punct",
            relation_rows,
            lambda row: row["relation"] not in {"root", "punct"},
        ),
        # Isolates scoring/head-selection differences from the upstream file's
        # Table-1 relation filtering (root/punct excluded, count >= 100).
        aggregate_row(
            "paper_repo_relation_filter",
            relation_rows,
            lambda row: row["relation"] not in {"root", "punct"}
            and row["n_edges"] >= 100,
        ),
    ]
    return aggregate_rows, relation_rows, shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--attention-filename", default=DEFAULT_ATTENTION_FILENAME)
    parser.add_argument("--output-filename", default=DEFAULT_OUTPUT_FILENAME)
    parser.add_argument("--manifest-filename", default=DEFAULT_MANIFEST_FILENAME)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    attention_path = checkpoint / args.attention_filename
    output_path = checkpoint / args.output_filename
    manifest_path = checkpoint / args.manifest_filename
    if not attention_path.exists():
        raise FileNotFoundError(attention_path)

    data = load_pickle(attention_path)
    aggregate_rows, relation_rows, shape = evaluate(data)
    fields = [
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
    ]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate_rows + relation_rows)
    temporary.replace(output_path)

    manifest = {
        "attention_file": str(attention_path),
        "output_file": str(output_path),
        "num_examples": len(data),
        "num_layers": int(shape[0]),
        "num_heads": int(shape[1]),
        "primary_scope": "paper_all_relations",
        "definition": {
            "parent_prediction": "argmax_j max(alpha_ij, alpha_ji)",
            "head_selection": "independent best layer/head per dependency relation",
            "aggregation": "micro-average of relation-selected correct predictions",
            "special_tokens": "CLS, SEP, and self-attention excluded",
            "truncation": "same zip-style truncation as upstream evaluator",
            "poss_special_case": False,
        },
        "scopes": {
            "paper_all_relations": "literal Appendix A Equations 3-5",
            "paper_no_root_punct": "all relations except root and punct",
            "paper_repo_relation_filter": (
                "relations except root/punct with at least 100 scored edges"
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Paper-equation UAS: {output_path}")
    for row in aggregate_rows:
        print(
            f"  {row['scope']}: {row['uas_percent']:.2f}% "
            f"({row['n_correct']}/{row['n_edges']})"
        )


if __name__ == "__main__":
    main()
