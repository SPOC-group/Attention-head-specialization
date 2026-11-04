#!/usr/bin/env python3
"""Create nested finite datasets with exactly one permanent MLM mask per sequence."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import datasets
import torch
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dataset", default="angie-chen55/bert_pretraining_data")
    p.add_argument("--source-config", default=None)
    p.add_argument("--source-split", default="train")
    p.add_argument("--tokenizer", default="bert-base-uncased")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-train-sequences", type=int, default=10000)
    p.add_argument("--validation-sequences", type=int, default=2048)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--mask-seed", type=int, default=1009)
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--cache-dir", default=None)
    return p.parse_args()


def fixed_mask(example, index, tokenizer, args, seed_offset):
    raw_input_ids = list(example["input_ids"])
    raw_attention = list(example.get("attention_mask", [1] * len(raw_input_ids)))
    raw_types = list(example.get("token_type_ids", [0] * len(raw_input_ids)))
    active_positions = [i for i, value in enumerate(raw_attention) if value]
    active_length = active_positions[-1] + 1 if active_positions else 0
    input_ids = raw_input_ids[:active_length][: args.max_seq_length]
    attention_mask = raw_attention[:active_length][: len(input_ids)]
    token_type_ids = raw_types[:active_length][: len(input_ids)]
    if not input_ids:
        raise ValueError("Encountered an empty tokenized sequence")
    # Cropping a 512-token pretraining example would otherwise discard its
    # terminal separator. Preserve standard BERT framing at length 128.
    if active_length > args.max_seq_length and tokenizer.sep_token_id is not None:
        input_ids[-1] = tokenizer.sep_token_id

    ids = torch.tensor(input_ids, dtype=torch.long)
    eligible = torch.tensor(
        [
            (not is_special) and bool(attend)
            for is_special, attend in zip(
                tokenizer.get_special_tokens_mask(
                    input_ids, already_has_special_tokens=True
                ),
                attention_mask,
            )
        ],
        dtype=torch.bool,
    )
    candidates = torch.nonzero(eligible, as_tuple=False).flatten()
    if len(candidates) == 0:
        raise ValueError("Sequence has no maskable tokens")

    generator = torch.Generator()
    generator.manual_seed(args.mask_seed + seed_offset + int(index))
    probabilities = torch.full(ids.shape, args.mlm_probability)
    masked = torch.bernoulli(probabilities, generator=generator).bool() & eligible
    if not torch.any(masked):
        chosen = candidates[
            torch.randint(len(candidates), (1,), generator=generator).item()
        ]
        masked[chosen] = True

    labels = ids.clone()
    labels[~masked] = -100

    # Standard BERT 80/10/10 corruption, sampled once and then frozen.
    replace_with_mask = (
        torch.bernoulli(torch.full(ids.shape, 0.8), generator=generator).bool()
        & masked
    )
    ids[replace_with_mask] = tokenizer.mask_token_id

    replace_random = (
        torch.bernoulli(torch.full(ids.shape, 0.5), generator=generator).bool()
        & masked
        & ~replace_with_mask
    )
    random_words = torch.randint(
        len(tokenizer), ids.shape, dtype=torch.long, generator=generator
    )
    ids[replace_random] = random_words[replace_random]

    return {
        "input_ids": ids.tolist(),
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels.tolist(),
        "example_id": int(index + seed_offset),
        "n_masked": int(masked.sum().item()),
    }


def split_hash(dataset):
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(torch.tensor(row["input_ids"], dtype=torch.int32).numpy().tobytes())
        digest.update(torch.tensor(row["labels"], dtype=torch.int32).numpy().tobytes())
    return digest.hexdigest()


def validate_fixed_split(dataset, max_seq_length, split_name):
    """Fail before publication if a frozen split is malformed."""
    if len(dataset) == 0:
        raise ValueError(f"{split_name} split is empty")
    for index, row in enumerate(dataset):
        lengths = {
            len(row[key])
            for key in ("input_ids", "attention_mask", "token_type_ids", "labels")
        }
        if len(lengths) != 1:
            raise ValueError(
                f"{split_name}[{index}] has inconsistent sequence lengths: {lengths}"
            )
        sequence_length = lengths.pop()
        if sequence_length == 0 or sequence_length > max_seq_length:
            raise ValueError(
                f"{split_name}[{index}] has invalid length {sequence_length}"
            )
        supervised = sum(label != -100 for label in row["labels"])
        if supervised != int(row["n_masked"]) or supervised < 1:
            raise ValueError(
                f"{split_name}[{index}] has inconsistent fixed MLM labels"
            )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Use a new path so a finite dataset is never "
            "silently replaced."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, cache_dir=args.cache_dir, use_fast=True
    )
    source = datasets.load_dataset(
        args.source_dataset,
        args.source_config,
        split=args.source_split,
        cache_dir=args.cache_dir,
    )
    required = args.max_train_sequences + args.validation_sequences
    if len(source) < required:
        raise ValueError(f"Source has {len(source)} rows, but {required} are required")
    if "input_ids" not in source.column_names:
        raise ValueError("The source dataset must already contain tokenized input_ids")

    # One deterministic permutation defines all nested training subsets.  The
    # validation block is disjoint from even the largest permitted subset.
    selected = source.shuffle(seed=args.subset_seed).select(range(required))
    train_source = selected.select(range(args.max_train_sequences))
    validation_source = selected.select(range(args.max_train_sequences, required))

    remove_train = train_source.column_names
    remove_validation = validation_source.column_names
    train_fixed = train_source.map(
        lambda row, idx: fixed_mask(row, idx, tokenizer, args, seed_offset=0),
        with_indices=True,
        remove_columns=remove_train,
        num_proc=args.num_proc,
        desc="Freezing training masks",
    )
    validation_fixed = validation_source.map(
        lambda row, idx: fixed_mask(
            row, idx, tokenizer, args, seed_offset=10**9
        ),
        with_indices=True,
        remove_columns=remove_validation,
        num_proc=args.num_proc,
        desc="Freezing validation masks",
    )

    fixed = datasets.DatasetDict(
        {"train": train_fixed, "validation": validation_fixed}
    )

    validate_fixed_split(train_fixed, args.max_seq_length, "train")
    validate_fixed_split(validation_fixed, args.max_seq_length, "validation")

    train_hash = split_hash(train_fixed)
    validation_hash = split_hash(validation_fixed)

    manifest = {
        "source_dataset": args.source_dataset,
        "source_config": args.source_config,
        "source_split": args.source_split,
        "source_fingerprint": getattr(source, "_fingerprint", None),
        "fixed_dataset_fingerprint": {
            split: getattr(fixed[split], "_fingerprint", None) for split in fixed
        },
        "tokenizer": args.tokenizer,
        "max_train_sequences": len(train_fixed),
        "validation_sequences": len(validation_fixed),
        "max_seq_length": args.max_seq_length,
        "mlm_probability": args.mlm_probability,
        "subset_seed": args.subset_seed,
        "mask_seed": args.mask_seed,
        "train_masked_tokens": int(sum(train_fixed["n_masked"])),
        "validation_masked_tokens": int(sum(validation_fixed["n_masked"])),
        "train_fixed_mask_sha256": train_hash,
        "validation_fixed_mask_sha256": validation_hash,
    }

    # Publish atomically. An interrupted save can leave a temporary directory,
    # but never a final DATA_DIR that looks complete and is not.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.with_name(f"{output_dir.name}.tmp-{os.getpid()}")
    if temporary_dir.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_dir}")
    fixed.save_to_disk(str(temporary_dir), num_proc=args.num_proc)
    (temporary_dir / "fixed_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary_dir.rename(output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved immutable fixed-mask dataset to {output_dir}")


if __name__ == "__main__":
    main()
