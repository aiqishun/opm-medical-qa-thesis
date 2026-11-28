#!/usr/bin/env python3
"""Run one experimental condition on one split and freeze the raw outputs.

    python3 scripts/run_condition.py --condition C1 --split pilot [--limit 2]

Model settings come from ``configs/<condition>.json`` (or ``--config``) and can
be overridden on the command line. The runner reads only answer-free inputs;
scoring is done separately by ``scripts/score_run.py``.

Outputs go to ``experiments/runs/<run_id>/`` with
``run_id = <UTC timestamp>_<condition>_<split>_<git short hash>``:

    manifest.json              configuration and provenance (written at start)
    calls/<nnn>_<qid>_r<k>.json one record per model call (written when the call returns)
    predictions/<qid>.json     final prediction per question
    complete.json              end-of-run counts (absent if the run was interrupted)

Every file is created once and never modified. Passage text is never written;
only passage identifiers are stored.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import get_condition  # noqa: E402
from agents.base import ModelConfig  # noqa: E402
from experiment import inputs as data  # noqa: E402
from experiment.records import CODE_DIRS, git_info, safe_component, sha256_file, write_once_json  # noqa: E402
from llm.client import LLMClient, OpenAIResponsesClient  # noqa: E402

RUNS_DIR = ROOT / "experiments/runs"
CONFIGS_DIR = ROOT / "configs"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(data.ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--split", required=True, choices=sorted(data.SPLIT_FILES))
    parser.add_argument("--config", type=Path, help="JSON config (default: configs/<condition>.json)")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--limit", type=int, help="run only the first N questions of the split (file order)")
    parser.add_argument("--i-confirm-config-frozen", action="store_true",
                        help="required for --split test: the configuration is frozen and tagged in git")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="allow running with uncommitted changes to tracked files (development only)")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> ModelConfig:
    path = args.config or CONFIGS_DIR / f"{args.condition.lower()}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("condition", args.condition) != args.condition:
        raise SystemExit(f"error: {path} is for condition {config['condition']}, not {args.condition}")
    values = {
        "model": args.model or config.get("model"),
        "reasoning_effort": args.reasoning_effort or config.get("reasoning_effort"),
        "max_output_tokens": args.max_output_tokens or config.get("max_output_tokens"),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise SystemExit(f"error: missing model settings {missing} (config {path} or command line)")
    return ModelConfig(model=values["model"], reasoning_effort=values["reasoning_effort"],
                       max_output_tokens=int(values["max_output_tokens"]))


def main(argv: list[str] | None = None,
         client_factory: Callable[[], LLMClient] = OpenAIResponsesClient,
         root: Path = ROOT) -> int:
    args = parse_args(argv)

    if args.split == "test" and not args.i_confirm_config_frozen:
        print("error: --split test requires --i-confirm-config-frozen. Run the test split only after C1-C3 "
              "are implemented, the configuration is frozen and a git tag has been created.", file=sys.stderr)
        return 2

    git = git_info(root)
    if args.split == "test":
        if git["dirty"] is not False:
            print("error: the test split may not be run from a dirty working tree.", file=sys.stderr)
            return 2
        if not git["tag"]:
            print("error: --split test requires HEAD to be exactly at a git tag "
                  "(git describe --exact-match --tags HEAD). Tag the frozen configuration first.", file=sys.stderr)
            return 2
    if git["dirty"] and not args.allow_dirty:
        print("error: uncommitted changes to tracked files or untracked files under "
              f"{', '.join(CODE_DIRS)}: {(git['modified_tracked'] + git['untracked_code_files'])[:5]}. "
              "Commit first so the run is tied to a commit (or pass --allow-dirty for development).",
              file=sys.stderr)
        return 2

    condition_cls = get_condition(args.condition)
    config = resolve_config(args)

    question_path = data.question_file(args.split)
    retrieval_path = data.RETRIEVAL_DIR / f"{args.split}.jsonl"
    questions = data.load_questions(question_path)
    if args.limit is not None:
        questions = questions[:args.limit]
    retrieval = data.load_retrieval(retrieval_path)
    evidence = data.build_evidence(retrieval, questions, data.CORPUS_PASSAGES)

    client = client_factory()
    agent = condition_cls(config, client)

    started = datetime.now(timezone.utc)
    run_id = "_".join([started.strftime("%Y%m%dT%H%M%SZ"), args.condition, args.split, git["short"]])
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    import openai  # only for the version string

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "condition": args.condition,
        "split": args.split,
        "started_at": started.isoformat(),
        "model_config": {"model": config.model, "reasoning_effort": config.reasoning_effort,
                         "max_output_tokens": config.max_output_tokens, "sampling_parameters": "none (not sent)"},
        "config_file": rel(args.config or CONFIGS_DIR / f"{args.condition.lower()}.json"),
        "retry_policy": "at most 1 retry, only on network or rate-limit errors; SDK auto-retry disabled",
        **agent.provenance(),
        "inputs": {
            "questions": {"path": rel(question_path), "sha256": sha256_file(question_path)},
            "retrieval": {"path": rel(retrieval_path), "sha256": sha256_file(retrieval_path),
                          "manifest_sha256": sha256_file(data.RETRIEVAL_DIR / "manifest.json")},
            "corpus_manifest_sha256": sha256_file(data.CORPUS_PASSAGES.parent / "manifest.json"),
        },
        "limit": args.limit,
        "question_ids": [q.id for q in questions],
        "passage_ids": {q.id: [p.passage_id for p in evidence[q.id]] for q in questions},
        "git": git,
        "git_tag": git["tag"],
        "allow_dirty": args.allow_dirty,
        "software": {"python": platform.python_version(), "openai": openai.__version__},
    }
    write_once_json(run_dir / "manifest.json", manifest)

    call_seq = 0
    counts = {"questions": 0, "valid": 0, "invalid": 0, "abstain": 0, "calls": 0}

    def on_call(record: dict[str, Any]) -> None:
        nonlocal call_seq
        call_seq += 1
        record = {"run_id": run_id, "call_index": call_seq, **record}
        name = f"{call_seq:04d}_{safe_component(record['question_id'])}_r{record['round']}.json"
        write_once_json(run_dir / "calls" / name, record)

    for index, question in enumerate(questions, 1):
        result = agent.run_question(question, evidence[question.id], on_call)
        write_once_json(run_dir / "predictions" / f"{safe_component(question.id)}.json",
                        {"run_id": run_id, "condition": args.condition, **result.to_dict()})
        counts["questions"] += 1
        counts["calls"] += result.n_calls
        counts[result.validity] += 1
        counts["abstain"] += result.answer == "abstain"
        print(f"[{index}/{len(questions)}] {question.id[:15]}… answer={result.answer} {result.validity}"
              + (f" ({result.invalid_reason})" if result.invalid_reason else ""), flush=True)

    write_once_json(run_dir / "complete.json", {
        "run_id": run_id, "finished_at": datetime.now(timezone.utc).isoformat(), "counts": counts})
    print(f"run written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
