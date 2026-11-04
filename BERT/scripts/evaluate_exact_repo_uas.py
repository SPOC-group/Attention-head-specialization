#!/usr/bin/env python3
"""Run both the unchanged repository probe and the paper-equation UAS probe.

No UAS prediction, scoring, filtering, or head-selection logic is implemented
here. The first result comes from the verbatim files in
``sudden-drops-in-the-loss-main``.  The second result is computed by
``paper_equation_uas.py`` from the same attention pickle using Appendix A,
Equations 3--5.  This wrapper only selects checkpoints and invokes the scripts.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


ATTENTION_FILENAME = "repo_exact_wsj_attention.pkl"
UAS_FILENAME = "repo_exact_depparse_uas.csv"
PAPER_UAS_FILENAME = "paper_equation_uas.csv"
PAPER_MANIFEST_FILENAME = "paper_equation_uas_manifest.json"


def step_of(path):
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def find_checkpoints(run_dir, requested):
    checkpoints = sorted(
        [
            path
            for path in run_dir.glob("checkpoint-*")
            if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name)
        ],
        key=step_of,
    )
    if requested:
        wanted = {int(value) for value in requested.split(",")}
        checkpoints = [path for path in checkpoints if step_of(path) in wanted]
        missing = wanted - {step_of(path) for path in checkpoints}
        if missing:
            raise FileNotFoundError(f"Missing checkpoint steps: {sorted(missing)}")
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories under {run_dir}")
    return checkpoints


def run(command, root):
    command = [str(value) for value in command]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wsj-json", required=True)
    parser.add_argument("--checkpoint-steps", default=None)
    parser.add_argument("--keep-attention-pickles", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    root = scripts.parent
    run_dir = Path(args.run_dir).expanduser().resolve()
    wsj_json = Path(args.wsj_json).expanduser().resolve()
    if not wsj_json.exists():
        raise FileNotFoundError(wsj_json)

    extractor = root / "sudden-drops-in-the-loss-main" / "extract_attention_pt.py"
    probe = root / "sudden-drops-in-the-loss-main" / "attention_depparse_probe.py"
    paper_probe = scripts / "paper_equation_uas.py"

    checkpoints = find_checkpoints(run_dir, args.checkpoint_steps)
    for index, checkpoint in enumerate(checkpoints, start=1):
        attention_path = checkpoint / ATTENTION_FILENAME
        uas_path = checkpoint / UAS_FILENAME
        paper_uas_path = checkpoint / PAPER_UAS_FILENAME
        paper_manifest_path = checkpoint / PAPER_MANIFEST_FILENAME
        print(f"[{index}/{len(checkpoints)}] repository + paper-equation UAS: {checkpoint.name}", flush=True)
        if uas_path.exists() and paper_uas_path.exists() and not args.force:
            print(f"  already complete: {uas_path} and {paper_uas_path}")
            continue
        if args.force:
            for path in (attention_path, uas_path, paper_uas_path, paper_manifest_path):
                if path.exists():
                    path.unlink()

        if not attention_path.exists():
            run(
                [
                    sys.executable,
                    extractor,
                    "--model-dir",
                    checkpoint,
                    "--preprocessed-data-file",
                    wsj_json,
                    "--output-filename",
                    ATTENTION_FILENAME,
                    "--word_level",
                    "--batch-size",
                    "16",
                ],
                root,
            )
        if not uas_path.exists():
            run(
                [
                    sys.executable,
                    probe,
                    "--bert-parent-dir",
                    checkpoint,
                    "--attn_map_filename",
                    ATTENTION_FILENAME,
                    "--output_filename",
                    UAS_FILENAME,
                ],
                root,
            )
        if not paper_uas_path.exists():
            run(
                [
                    sys.executable,
                    paper_probe,
                    "--checkpoint-dir",
                    checkpoint,
                    "--attention-filename",
                    ATTENTION_FILENAME,
                    "--output-filename",
                    PAPER_UAS_FILENAME,
                    "--manifest-filename",
                    PAPER_MANIFEST_FILENAME,
                ],
                root,
            )
        if not uas_path.exists():
            raise RuntimeError(f"The upstream probe did not create {uas_path}")
        if not paper_uas_path.exists():
            raise RuntimeError(f"The paper-equation probe did not create {paper_uas_path}")
        if attention_path.exists() and not args.keep_attention_pickles:
            attention_path.unlink()

    run([sys.executable, scripts / "collect_metrics.py", "--run-dir", run_dir], root)
    print(f"Repository and paper-equation evaluation complete: {run_dir}")


if __name__ == "__main__":
    main()
