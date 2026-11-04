#!/usr/bin/env python3
"""Sample 1,000 WSJ sentences as one validated dependency tree per JSONL row.

The Hugging Face dataset contains document-level rows in which dependency-head
indices restart at every sentence.  The authors' attention extractor, however,
treats every JSONL row as one sequence.  This script reconstructs sentence
boundaries before export so that word positions and gold head indices share the
same coordinate system during UAS evaluation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile


TERMINAL_TOKENS = {".", "?", "!", "...", "?!", "!?"}
CLOSING_TOKENS = {"''", "'", '"', "’", "”", ")", "]", "}"}


def dependency_tree_error(heads, relns):
    """Return None for a valid local dependency tree, otherwise an explanation."""
    n_words = len(heads)
    if len(relns) != n_words:
        return "head/relation lengths differ"

    roots_by_head = [i for i, head in enumerate(heads) if head == 0]
    roots_by_label = [i for i, reln in enumerate(relns) if reln == "root"]
    if len(roots_by_head) != 1:
        return f"expected one head=0 root, found {len(roots_by_head)}"
    if len(roots_by_label) != 1:
        return f"expected one relation='root', found {len(roots_by_label)}"
    if roots_by_head[0] != roots_by_label[0]:
        return "head=0 and relation='root' identify different words"

    for child, head in enumerate(heads, start=1):
        if not isinstance(head, int):
            return f"head at word {child} is not an integer: {head!r}"
        if head < 0 or head > n_words:
            return f"head {head} at word {child} is outside [0, {n_words}]"
        if head == child:
            return f"self-loop at word {child}"

    # Every node must reach the unique root rather than a cycle.
    for child in range(1, n_words + 1):
        current = child
        visited = set()
        while current != 0:
            if current in visited:
                return f"cycle reached from word {child}"
            visited.add(current)
            current = heads[current - 1]

    return None


def boundary_penalty(words, end, document_length):
    """Prefer conventional sentence-final punctuation when a split is ambiguous."""
    if end == document_length:
        return 0.0

    final_index = end - 1
    while final_index >= 0 and words[final_index] in CLOSING_TOKENS:
        final_index -= 1
    if final_index >= 0 and words[final_index] in TERMINAL_TOKENS:
        return 0.0
    if words[end - 1] in TERMINAL_TOKENS:
        return 0.01
    if words[end - 1] in {",", ";", ":"}:
        return 10.0
    return 100.0


def infer_sentence_spans(words, heads, relns):
    """Infer the minimum-penalty partition into valid local dependency trees."""
    if not (len(words) == len(heads) == len(relns)):
        raise ValueError(
            f"parallel fields have different lengths: words={len(words)}, "
            f"heads={len(heads)}, relns={len(relns)}"
        )
    if not words:
        raise ValueError("empty document")

    head_roots = [i for i, head in enumerate(heads) if head == 0]
    label_roots = [i for i, reln in enumerate(relns) if reln == "root"]
    if head_roots != label_roots:
        raise ValueError(
            "positions with head=0 do not equal positions labeled root: "
            f"{head_roots} vs {label_roots}"
        )
    if not head_roots:
        raise ValueError("document contains no dependency root")

    # Each boundary separating two roots must fall strictly after the left root
    # and no later than the right root.  Dynamic programming tries every such
    # boundary and retains only slices that are valid local dependency trees.
    boundary_options = [[0]]
    for left_root, right_root in zip(head_roots[:-1], head_roots[1:]):
        boundary_options.append(list(range(left_root + 1, right_root + 1)))
    boundary_options.append([len(words)])

    states = {0: (0.0, [])}
    for sentence_index, root_position in enumerate(head_roots):
        next_states = {}
        for start, (cost, spans) in states.items():
            for end in boundary_options[sentence_index + 1]:
                if not (start <= root_position < end):
                    continue
                error = dependency_tree_error(heads[start:end], relns[start:end])
                if error is not None:
                    continue
                candidate = (
                    cost + boundary_penalty(words, end, len(words)),
                    spans + [(start, end)],
                )
                previous = next_states.get(end)
                if previous is None or candidate[0] < previous[0]:
                    next_states[end] = candidate
        states = next_states
        if not states:
            raise ValueError(
                f"could not form a valid sentence containing root "
                f"{sentence_index + 1}/{len(head_roots)} at word {root_position + 1}"
            )

    if len(words) not in states:
        raise ValueError("could not partition the complete document")
    return states[len(words)]


def split_document(row):
    words = list(row["words"])
    heads = [int(head) for head in row["heads"]]
    relns = list(row["relns"])
    penalty, spans = infer_sentence_spans(words, heads, relns)

    sentences = []
    for start, end in spans:
        sentence = {
            "words": words[start:end],
            "heads": heads[start:end],
            "relns": relns[start:end],
        }
        error = dependency_tree_error(sentence["heads"], sentence["relns"])
        if error is not None:
            raise AssertionError(f"internal sentence validation failed: {error}")
        sentences.append(sentence)
    return sentences, penalty


def sample_sentence_indices(n_available, n_requested, seed):
    """Sample without replacement, retaining the source order in the output."""
    if not 1 <= n_requested <= n_available:
        raise ValueError(
            f"requested {n_requested} sentences, but source contains {n_available}; "
            "choose a positive sample size no larger than the source"
        )
    return sorted(random.Random(seed).sample(range(n_available), n_requested))


def publish_probe(output, payload, manifest):
    """Publish complete files atomically, without replacing an existing probe."""
    manifest_path = Path(str(output) + ".manifest.json")
    temporary_paths = []
    published_paths = []
    try:
        for destination, contents in (
            (output, payload),
            (manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8")),
        ):
            with tempfile.NamedTemporaryFile(
                dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                temporary_paths.append(temporary)
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            # Unlike replace(), link() refuses to overwrite a concurrently
            # created destination. Both paths are on the same filesystem.
            os.link(temporary, destination)
            published_paths.append(destination)
    except BaseException:
        for path in published_paths:
            path.unlink()
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dataset", default="angie-chen55/wsj-dep-parses-1k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-sentences", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_sentences < 1:
        parser.error("--num-sentences must be positive")

    output = Path(args.output).expanduser().resolve()
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite {output} or {manifest_path}; choose a new filename"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    # Keep this import inside main so reconstruction and sampling can be checked
    # independently of the Hugging Face environment.
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    all_sentences = []
    multi_sentence_documents = 0
    total_boundary_penalty = 0.0
    for document_index, row in enumerate(dataset):
        try:
            sentences, penalty = split_document(row)
        except Exception as exc:
            raise ValueError(
                f"failed to reconstruct dataset document {document_index}: {exc}"
            ) from exc
        multi_sentence_documents += len(sentences) > 1
        total_boundary_penalty += penalty
        all_sentences.extend(sentences)

    indices = sample_sentence_indices(
        len(all_sentences), args.num_sentences, args.sample_seed
    )
    serialized = [
        (json.dumps(sentence, ensure_ascii=False) + "\n").encode("utf-8")
        for sentence in all_sentences
    ]
    source_digest = hashlib.sha256()
    for line in serialized:
        source_digest.update(line)
    payload = b"".join(serialized[i] for i in indices)
    word_counts = [len(all_sentences[i]["words"]) for i in indices]
    n_tokens = sum(word_counts)
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "probe_unit": "sentence",
        "sentence_split": True,
        "sentence_split_method": "dependency-root partition with tree validation",
        "sampling_method": "random.Random(seed).sample without replacement; source order",
        "sampling_seed": args.sample_seed,
        "source_indices_zero_based": indices,
        "n_documents": len(dataset),
        "source_n_sentences": len(all_sentences),
        "source_n_tokens": sum(len(sentence["words"]) for sentence in all_sentences),
        "documents_with_multiple_sentences": multi_sentence_documents,
        "source_total_boundary_penalty": total_boundary_penalty,
        "source_fingerprint": getattr(dataset, "_fingerprint", None),
        "source_sha256": source_digest.hexdigest(),
        "source_hash_format": "reconstructed sentences serialized as UTF-8 JSONL",
        "n_sentences": len(indices),
        "n_tokens": n_tokens,
        "mean_sentence_words": n_tokens / len(indices),
        "maximum_sentence_words": max(word_counts),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "output": str(output),
    }
    publish_probe(output, payload, manifest)

    print(
        f"Wrote {len(indices)} sampled, validated sentences "
        f"from {len(all_sentences)} reconstructed sentences / {len(dataset)} documents "
        f"(seed {args.sample_seed})"
    )
    print(f"Probe: {output}")
    print(f"Manifest: {manifest_path}")
    print(f"SHA-256: {manifest['sha256']}")


if __name__ == "__main__":
    main()
