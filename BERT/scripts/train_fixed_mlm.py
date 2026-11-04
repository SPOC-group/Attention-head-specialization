#!/usr/bin/env python3
"""Train an ordinary Hugging Face BERT MLM on one immutable masked dataset.

The model class and MLM cross-entropy are the standard BertForMaskedLM used by
the paper.  The only experimental change is that input corruption and labels
were sampled once by prepare_fixed_dataset.py instead of by a dynamic collator.
"""

import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import datasets
import torch
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.optimization import get_linear_schedule_with_warmup


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-size", type=int, required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "bert_small_4x512.json"))
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--scheduler-total-steps", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--checkpoint-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-final-full-train-eval", action="store_true")
    return parser.parse_args()


class FixedMLMCollator:
    """Pad already-corrupted inputs without resampling their MLM labels."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [list(feature["labels"]) for feature in features]
        model_features = [
            {
                key: feature[key]
                for key in ("input_ids", "attention_mask", "token_type_ids")
                if key in feature
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        width = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [label + [-100] * (width - len(label)) for label in labels],
            dtype=torch.long,
        )
        return batch


class SaveValidationMetrics(TrainerCallback):
    """Persist the existing Trainer loss without changing its aggregation."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # train_eval uses another prefix and is deliberately not validation.
        if not state.is_world_process_zero or not metrics or "eval_loss" not in metrics:
            return control
        directory = Path(args.output_dir) / "checkpoint_metrics"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"step-{state.global_step}.json"
        temporary = path.with_suffix(".json.tmp")
        values = {"step": state.global_step, "validation_loss": metrics["eval_loss"],
                  "test_loss": metrics["eval_loss"], "epoch": state.epoch,
                  "loss_aggregation": "unchanged Hugging Face Trainer eval_loss"}
        temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        return control


class PaperScheduleTrainer(Trainer):
    """Use the upstream script's linear schedule with a fixed 1M-step horizon."""

    def __init__(self, *args, scheduler_total_steps=1000000, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler_total_steps = scheduler_total_steps

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.scheduler_total_steps <= 0:
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is None:
            chosen_optimizer = self.optimizer if optimizer is None else optimizer
            self.lr_scheduler = get_linear_schedule_with_warmup(
                chosen_optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=self.scheduler_total_steps,
            )
        return self.lr_scheduler


def checkpoint_step(path):
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def find_last_complete_checkpoint(output_dir):
    candidates = []
    if not output_dir.exists():
        return None
    for checkpoint in output_dir.glob("checkpoint-*"):
        step = checkpoint_step(checkpoint)
        if step in (None, 0):
            continue
        has_weights = any(
            (checkpoint / filename).exists()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
        if has_weights and all((checkpoint / filename).exists() for filename in required):
            candidates.append((step, checkpoint))
    return str(max(candidates)[1]) if candidates else None


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    dataset = datasets.load_from_disk(str(dataset_dir))
    if args.dataset_size > len(dataset["train"]):
        raise ValueError(
            f"Requested N={args.dataset_size}, but fixed master dataset contains "
            f"only {len(dataset['train'])} training sequences"
        )
    train_dataset = dataset["train"].select(range(args.dataset_size))
    eval_dataset = dataset["validation"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    config = BertConfig.from_json_file(str(config_path))
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    maximum_data_length = max(len(row) for row in dataset["train"]["input_ids"][: min(100, len(dataset["train"]))])
    if maximum_data_length > config.max_position_embeddings:
        raise ValueError(
            f"Dataset sequences reach length {maximum_data_length}, but the architecture "
            f"supports only {config.max_position_embeddings} positions"
        )
    model = BertForMaskedLM(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    run_name = args.run_name or f"{config_path.stem}_N{args.dataset_size}_seed{args.seed}"
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=not args.resume,
        do_train=True,
        do_eval=True,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.checkpoint_steps,
        save_strategy="steps",
        save_steps=args.checkpoint_steps,
        save_total_limit=None,
        prediction_loss_only=True,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        report_to=["wandb"] if args.wandb_project else [],
        run_name=run_name,
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
        logging_nan_inf_filter=False,
    )
    trainer = PaperScheduleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=FixedMLMCollator(tokenizer),
        scheduler_total_steps=args.scheduler_total_steps,
        callbacks=[SaveValidationMetrics()],
    )

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"{output_dir} is non-empty. Use a fresh path or --resume.")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint0 = output_dir / "checkpoint-0"
    last_checkpoint = find_last_complete_checkpoint(output_dir) if args.resume else None
    if last_checkpoint is None:
        trainer.save_model(str(checkpoint0))
        tokenizer.save_pretrained(str(checkpoint0))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = (
        args.train_batch_size * args.gradient_accumulation_steps * world_size
    )
    dataset_manifest_path = dataset_dir / "fixed_dataset_manifest.json"
    run_manifest = {
        "run_name": run_name,
        "dataset_dir": str(dataset_dir),
        "dataset_size": args.dataset_size,
        "validation_size": len(eval_dataset),
        "config_path": str(config_path),
        "config": config.to_dict(),
        "parameter_count": parameter_count,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "global_batch_size": global_batch,
        "max_steps": args.max_steps,
        "sequence_presentations": global_batch * args.max_steps,
        "presentations_per_training_sequence": global_batch * args.max_steps / args.dataset_size,
        "optimizer": "AdamW (Hugging Face Trainer)",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "scheduler_total_steps": args.scheduler_total_steps,
        "precision": args.precision,
        "fixed_dataset_manifest": json.loads(dataset_manifest_path.read_text())
        if dataset_manifest_path.exists()
        else None,
        "training_arguments": asdict(training_args),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True, default=str) + "\n"
    )

    if last_checkpoint is None:
        initial_eval = trainer.evaluate(metric_key_prefix="eval")
        (output_dir / "initial_eval.json").write_text(
            json.dumps(initial_eval, indent=2, sort_keys=True) + "\n"
        )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    final_checkpoint = output_dir / f"checkpoint-{trainer.state.global_step}"
    trainer.save_model(str(final_checkpoint))
    tokenizer.save_pretrained(str(final_checkpoint))
    trainer.save_model(str(output_dir / "final-model"))
    tokenizer.save_pretrained(str(output_dir / "final-model"))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if not args.skip_final_full_train_eval:
        train_eval = trainer.evaluate(
            eval_dataset=train_dataset, metric_key_prefix="train_eval"
        )
        trainer.log_metrics("train_eval", train_eval)
        trainer.save_metrics("train_eval", train_eval)
    final_eval = trainer.evaluate(metric_key_prefix="eval")
    trainer.log_metrics("eval", final_eval)
    trainer.save_metrics("eval", final_eval)
    trainer.save_state()
    (output_dir / "TRAINING_COMPLETE").write_text(
        f"{trainer.state.global_step}\n"
    )

    print(f"Run complete: {run_name}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
