"""Deterministic validators for evidence-linked OPM graphs (Definition 4.1, Tables 4.1 and 4.2).

Only the programmatic checks of Table 4.2 are implemented here; evidence
support, applicability and answer consistency need a reader (Verification
Agent / judge). Every issue names the element or relation concerned so that a
later repair step (C3) can target it.

Interpretation decisions (confirmed with the author, 2026-11-28):

* effect links are stored as process -> object (OPM's bidirectional effect link
  in one canonical direction);
* question findings may be objects or states;
* ``support_path`` alternates element, relation, element, ...; each relation must
  connect its two neighbours in either direction (undirected); state ownership
  is not a path step. The path is checked without regard to direction because
  OPM link types point in different directions (consumption is object ->
  process, result is process -> object, input_state is state -> process,
  output_state is process -> state, ...). A causal chain of reasoning from a
  finding to the answer may therefore run against some links. For example, a
  finding is often the *result* of the disease process that is the answer, so
  the path walks finding <-result- process. Direction stays meaningful through
  the endpoint-type check; the path check only asks whether the cited links
  connect the steps;
* ownership counts as a connection for the isolated-element check;
* when a process has both input and output states, the owners of its input
  states must equal the owners of its output states; a process with only input
  states or only output states passes;
* for an abstention, ``answer_element_id`` is null and ``support_path`` is empty,
  and the answer-path check is not applicable.

Checks beyond the six listed in the C2 task: ``unique_ids`` (ids are
unique across elements and relations) and ``grounding_rules`` (the
question/passages/none rules for graph items).
"""

from __future__ import annotations

from typing import Any

from evidence_graph.grounding import grounding_violations, quote_in_stem

ELEMENT_KINDS = {"objects": "object", "processes": "process", "states": "state"}

# relation type -> (allowed source kinds, allowed target kinds), Table 4.1
RELATION_ENDPOINTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "consumption": (frozenset({"object"}), frozenset({"process"})),
    "result": (frozenset({"process"}), frozenset({"object"})),
    "effect": (frozenset({"process"}), frozenset({"object"})),
    "instrument": (frozenset({"object"}), frozenset({"process"})),
    "input_state": (frozenset({"state"}), frozenset({"process"})),
    "output_state": (frozenset({"process"}), frozenset({"state"})),
    "invocation": (frozenset({"process"}), frozenset({"process"})),
    "aggregation": (frozenset({"object"}), frozenset({"object"})),
    "exhibition": (frozenset({"object"}), frozenset({"object", "process"})),
    "generalization": (frozenset({"object"}), frozenset({"object"})),
}

CHECKS = ("schema", "unique_ids", "endpoint_types", "state_ownership", "connectivity",
          "citation_validity", "question_quotes", "grounding_rules")


def _issue(item_id: str | None, problem: str, **detail: Any) -> dict[str, Any]:
    return {"id": item_id, "problem": problem, **detail}


def graph_items(graph: dict[str, Any]):
    """(kind, item) for every element and relation."""
    for key, kind in ELEMENT_KINDS.items():
        for item in graph.get(key, []):
            yield kind, item
    for item in graph.get("relations", []):
        yield "relation", item


def validate_graph(graph: dict[str, Any], *, answer: str, stem: str, allowed_passages: set[str]) -> dict[str, Any]:
    """Run all deterministic checks on a schema-valid graph."""
    issues: dict[str, list[dict[str, Any]]] = {name: [] for name in CHECKS}
    path_applicable = True

    # unique ids ---------------------------------------------------------------
    kind_of: dict[str, str] = {}
    for kind, item in graph_items(graph):
        if item["id"] in kind_of:
            issues["unique_ids"].append(_issue(item["id"], "duplicate_id", kinds=[kind_of[item["id"]], kind]))
        else:
            kind_of[item["id"]] = kind
    elements = {i: k for i, k in kind_of.items() if k != "relation"}
    relations = {r["id"]: r for r in graph["relations"]}
    states = {s["id"]: s for s in graph["states"]}

    # endpoint types -----------------------------------------------------------
    for rel in graph["relations"]:
        sources, targets = RELATION_ENDPOINTS[rel["type"]]
        for end, allowed in (("source_id", sources), ("target_id", targets)):
            node = rel[end]
            if node not in elements:
                issues["endpoint_types"].append(_issue(rel["id"], "unknown_endpoint", end=end, ref=node))
            elif elements[node] not in allowed:
                issues["endpoint_types"].append(_issue(rel["id"], "wrong_endpoint_kind", end=end, ref=node,
                                                       kind=elements[node], allowed=sorted(allowed)))
        if rel["source_id"] == rel["target_id"]:
            issues["endpoint_types"].append(_issue(rel["id"], "self_loop", ref=rel["source_id"]))

    # state ownership ------------------------------------------------------------
    for state in graph["states"]:
        owner = state["owner_object_id"]
        if owner not in elements:
            issues["state_ownership"].append(_issue(state["id"], "unknown_owner", ref=owner))
        elif elements[owner] != "object":
            issues["state_ownership"].append(_issue(state["id"], "owner_not_object", ref=owner, kind=elements[owner]))
    in_owners: dict[str, dict[str, list[str]]] = {}
    out_owners: dict[str, dict[str, list[str]]] = {}
    for rel in graph["relations"]:
        if rel["type"] == "input_state" and rel["source_id"] in states:
            in_owners.setdefault(rel["target_id"], {}).setdefault(states[rel["source_id"]]["owner_object_id"], []).append(rel["id"])
        if rel["type"] == "output_state" and rel["target_id"] in states:
            out_owners.setdefault(rel["source_id"], {}).setdefault(states[rel["target_id"]]["owner_object_id"], []).append(rel["id"])
    for process in sorted(set(in_owners) & set(out_owners)):
        if set(in_owners[process]) != set(out_owners[process]):
            issues["state_ownership"].append(_issue(
                process, "input_output_state_owner_mismatch",
                input_state_owners=sorted(in_owners[process]), output_state_owners=sorted(out_owners[process]),
                relations=sorted(r for d in (in_owners[process], out_owners[process]) for rs in d.values() for r in rs)))

    # connectivity: isolated elements ---------------------------------------------
    degree = {element: 0 for element in elements}
    for rel in graph["relations"]:
        for end in (rel["source_id"], rel["target_id"]):
            if end in degree:
                degree[end] += 1
    for state in graph["states"]:
        if state["owner_object_id"] in degree:
            degree[state["id"]] += 1
            degree[state["owner_object_id"]] += 1
    for element, d in degree.items():
        if d == 0:
            issues["connectivity"].append(_issue(element, "isolated_element", kind=elements[element]))

    # connectivity: answer element and support path ---------------------------------
    path, answer_id = graph["support_path"], graph["answer_element_id"]
    path_issues: list[dict[str, Any]] = []
    if answer == "abstain":
        if answer_id is not None or path:
            path_issues.append(_issue(answer_id, "abstain_with_answer_path"))
        else:
            path_applicable = False
    else:
        if answer_id is None:
            path_issues.append(_issue(None, "missing_answer_element"))
        elif answer_id not in elements:
            path_issues.append(_issue(answer_id, "answer_element_not_an_element"))
        path_issues.extend(_check_path(path, answer_id, elements, relations, graph))
    issues["connectivity"].extend(path_issues)

    # citations, quotes, grounding rules ----------------------------------------------
    for kind, item in graph_items(graph):
        for pid in item["passage_ids"]:
            if pid not in allowed_passages:
                issues["citation_validity"].append(_issue(item["id"], "passage_not_in_Eq", passage_id=pid))
        quote = item.get("question_quote")
        if item["grounding"] == "question" and quote and not quote_in_stem(quote, stem):
            issues["question_quotes"].append(_issue(item["id"], "quote_not_in_stem"))
        for problem in grounding_violations(item):
            issues["grounding_rules"].append(_issue(item["id"], problem))

    checks = {name: {"passed": not issues[name], "issues": issues[name]} for name in CHECKS}
    return {
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks.values()),
        "support_path_valid": (not path_issues) if path_applicable else None,
    }


def _check_path(path: list[str], answer_id: str | None, elements: dict[str, str],
                relations: dict[str, dict[str, Any]], graph: dict[str, Any]) -> list[dict[str, Any]]:
    findings = {o["id"] for o in graph["objects"] if o["is_question_finding"]}
    findings |= {s["id"] for s in graph["states"] if s["is_question_finding"]}
    if not path:
        return [_issue(None, "empty_support_path")]
    problems = []
    if len(path) % 2 == 0:
        problems.append(_issue(None, "support_path_even_length", length=len(path)))
    for position, item in enumerate(path):
        expected = "element" if position % 2 == 0 else "relation"
        known = item in elements if expected == "element" else item in relations
        if not known:
            problems.append(_issue(item, f"support_path_expected_{expected}", position=position))
    if problems:
        return problems
    for position in range(1, len(path) - 1, 2):
        rel = relations[path[position]]
        ends = {rel["source_id"], rel["target_id"]}
        if ends != {path[position - 1], path[position + 1]}:
            problems.append(_issue(rel["id"], "support_path_relation_does_not_connect_neighbours",
                                   position=position, neighbours=[path[position - 1], path[position + 1]]))
    if path[0] not in findings:
        problems.append(_issue(path[0], "support_path_does_not_start_at_question_finding"))
    if answer_id is not None and path[-1] != answer_id:
        problems.append(_issue(path[-1], "support_path_does_not_end_at_answer_element", answer_element_id=answer_id))
    return problems
