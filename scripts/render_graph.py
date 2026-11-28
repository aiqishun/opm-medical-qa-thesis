#!/usr/bin/env python3
"""Export the OPM graph of every question in a run as a Mermaid diagram (one .md per question).

    python3 scripts/render_graph.py experiments/runs/<run_id> [--out-dir outputs/graphs/<run_id>]

For manual inspection and as a starting point for OPCloud modelling. Only
identifiers, labels, link types, grounding labels, passage identifiers and
validator results are exported: no passage text, no question quotes and no
reference answers. The run directory is not modified.

Notation: objects are rectangles, processes are stadiums (OPM ellipses) and
states are rounded boxes joined to their owner by a dotted line. Findings
stated in the question have a bold border, and the answer element is
highlighted. Links on the support path are drawn thick.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/graphs"

ARROWS = {  # OPM link type -> Mermaid arrow (normal, on support path)
    "consumption": ("-->", "==>"), "result": ("-->", "==>"), "effect": ("<-->", "<==>"),
    "instrument": ("--o", "==o"), "input_state": ("-->", "==>"), "output_state": ("-->", "==>"),
    "invocation": ("-.->", "==>"), "aggregation": ("-->", "==>"), "exhibition": ("-->", "==>"),
    "generalization": ("-->", "==>"),
}
GROUNDING_MARK = {"question": "Q", "passages": "P", "none": "∅"}


def node_id(raw: str) -> str:
    return "n_" + re.sub(r"[^A-Za-z0-9_]", "_", raw)


def label(text: str) -> str:
    return text.replace('"', "#quot;").replace("\n", " ")


def to_mermaid(graph: dict[str, Any]) -> str:
    path = graph.get("support_path", [])
    path_relations = set(path[1::2])
    findings = {o["id"] for o in graph["objects"] if o.get("is_question_finding")}
    findings |= {s["id"] for s in graph["states"] if s.get("is_question_finding")}
    lines = ["flowchart LR"]
    for o in graph["objects"]:
        lines.append(f'    {node_id(o["id"])}["{label(o["label"])}<br/><small>{o["id"]} · {GROUNDING_MARK[o["grounding"]]}</small>"]')
    for p in graph["processes"]:
        lines.append(f'    {node_id(p["id"])}(["{label(p["label"])}<br/><small>{p["id"]} · {GROUNDING_MARK[p["grounding"]]}</small>"])')
    for s in graph["states"]:
        lines.append(f'    {node_id(s["id"])}("{label(s["label"])}<br/><small>{s["id"]} · {GROUNDING_MARK[s["grounding"]]}</small>")')
        lines.append(f'    {node_id(s["owner_object_id"])} -. state .- {node_id(s["id"])}')
    for r in graph["relations"]:
        arrow = ARROWS[r["type"]][1 if r["id"] in path_relations else 0]
        lines.append(f'    {node_id(r["source_id"])} {arrow}|"{r["type"]} ({r["id"]} · {GROUNDING_MARK[r["grounding"]]})"| '
                     f'{node_id(r["target_id"])}')
    lines.append("    classDef finding stroke-width:3px")
    lines.append("    classDef answer fill:#ffe9a8,stroke:#b8860b,stroke-width:3px")
    if findings:
        lines.append("    class " + ",".join(node_id(f) for f in sorted(findings)) + " finding")
    if graph.get("answer_element_id"):
        lines.append(f"    class {node_id(graph['answer_element_id'])} answer")
    return "\n".join(lines)


def element_table(graph: dict[str, Any]) -> str:
    rows = ["| id | kind | label / link | grounding | passage ids |", "|---|---|---|---|---|"]
    for kind, key in (("object", "objects"), ("process", "processes"), ("state", "states")):
        for e in graph[key]:
            extra = f" (state of {e['owner_object_id']})" if kind == "state" else ""
            extra += " · finding" if e.get("is_question_finding") else ""
            rows.append(f"| {e['id']} | {kind} | {e['label']}{extra} | {e['grounding']} | {', '.join(e['passage_ids'])} |")
    for r in graph["relations"]:
        rows.append(f"| {r['id']} | {r['type']} | {r['source_id']} → {r['target_id']} | {r['grounding']} | "
                    f"{', '.join(r['passage_ids'])} |")
    return "\n".join(rows)


def render_prediction(pred: dict[str, Any]) -> str | None:
    output = pred.get("output")
    if pred.get("validity") != "valid" or not output or "graph" not in output:
        return None
    graph, validation = output["graph"], pred.get("extra", {}).get("validation")
    parts = [f"# {pred['question_id']}", "",
             f"Run `{pred['run_id']}` · condition {pred['condition']} · selected answer **{output['answer']}** · "
             f"answer element `{graph.get('answer_element_id')}`", "",
             f"Support path: `{' → '.join(graph.get('support_path', [])) or '(empty)'}`", "",
             "```mermaid", to_mermaid(graph), "```", "", "## Elements and relations", "", element_table(graph), ""]
    if validation:
        parts += ["## Validator results", ""]
        for name, check in validation["checks"].items():
            status = "pass" if check["passed"] else "FAIL"
            detail = "; ".join(f"{i.get('id')}: {i['problem']}" for i in check["issues"])
            parts.append(f"- {name}: {status}" + (f" — {detail}" if detail else ""))
        parts.append("")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, help="default: outputs/graphs/<run_id>/")
    args = parser.parse_args(argv)
    out_dir = args.out_dir or DEFAULT_OUT / args.run_dir.name
    if args.run_dir.resolve() in [out_dir.resolve(), *out_dir.resolve().parents]:
        parser.error("do not write into the frozen run directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for path in sorted((args.run_dir / "predictions").glob("*.json")):
        text = render_prediction(json.loads(path.read_text(encoding="utf-8")))
        if text is None:
            skipped += 1
            continue
        (out_dir / f"{path.stem}.md").write_text(text, encoding="utf-8")
        written += 1
    print(f"wrote {written} graph(s) to {out_dir}" + (f"; skipped {skipped} without a valid graph" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
