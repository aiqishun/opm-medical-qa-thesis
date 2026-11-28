#!/usr/bin/env python3
"""Send one minimal structured-output request to confirm the API accepts the configured parameters.

    python3 scripts/check_model_params.py [--config configs/c1.json]

Uses the same client, model, reasoning effort, max_output_tokens and JSON-schema
mode as the experiment runner, with a trivial prompt and no question data.
Prints the API's reply metadata (never the API key).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.client import LLMRequest, OpenAIResponsesClient  # noqa: E402

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c1.json")
    parser.add_argument("--schema", type=Path,
                        help="send this response schema instead of the trivial one, to confirm strict mode accepts it")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8")) if args.schema else SCHEMA
    system = ("Return a minimal JSON object that satisfies the response schema; use empty lists where allowed."
              if args.schema else "Reply with the JSON object {\"ok\": true}.")
    request = LLMRequest(model=config["model"], reasoning_effort=config["reasoning_effort"],
                         max_output_tokens=config["max_output_tokens"],
                         system_prompt=system, user_prompt="ping",
                         schema_name="schema_check" if args.schema else "param_check", schema=schema)
    response = OpenAIResponsesClient().complete(request)
    print(json.dumps({
        "parameters_sent": request.api_parameters(),
        "api_ok": response.ok,
        "error": response.error,
        "model_returned": response.response_model,
        "status": response.status,
        "raw_text": response.raw_text,
        "usage": {"input": response.input_tokens, "output": response.output_tokens,
                  "reasoning": response.reasoning_tokens},
        "latency_s": round(response.latency_s, 2),
    }, indent=2))
    return 0 if response.ok and response.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
