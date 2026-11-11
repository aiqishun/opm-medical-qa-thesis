# An OPM-Based Explainable Question Answering Framework for Cardiology

## 1. Abstract

We present a research prototype for explainable cardiology question answering
based on Object-Process Methodology (OPM). Given a free-text question, the
system matches the question to a cardiology topic, generates a short answer
and explanation, and exports an OPM-style graph containing objects, processes,
states, and typed links. The goal is not to validate clinical correctness, but
to study whether structured graph artifacts can make medical QA behavior more
auditable. We evaluate 543 generated graphs from a 546-row local cardiology QA
batch using a heuristic graph-quality audit. The graphs show strong structural
validity (`valid_graph_rate = 1.0000`) and strong simplified OPM constraint
satisfaction (`process_object_connectivity_rate = 0.9435`,
`state_object_attachment_rate = 0.9509`, `explanation_path_rate = 1.0000`).
However, source grounding and answer-graph consistency remain moderate
(`overall_grounding_rate = 0.3980`, `answer_concept_coverage = 0.4103`,
`graph_evidence_usage_rate = 0.3364`). A 20-sample manual sanity check
confirms this pattern: most graphs are at least partially relevant, but full
path explainability is limited.

## 2. Introduction

Medical QA systems increasingly produce fluent answers, but their reasoning
steps are often difficult to inspect. In clinical and educational settings,
users need to understand not only an answer but also the mechanism by which a
system reached it. This paper asks whether OPM-style structured artifacts can
make a cardiology QA prototype more transparent and easier to audit.

We implement a compact pipeline in which each matched question produces a
natural-language answer, a short explanation, a reasoning path, and an
OPM-style graph. The representation is intentionally simplified: it is a
readable prototype format rather than a standard-compliant OPM serialization.
Our evaluation focuses on graph quality, grounding, and answer-graph alignment
rather than clinical accuracy. We do not claim that the generated answers are
clinically correct or suitable for diagnosis, triage, or medical
decision-making.

Our contributions are:

- We present an OPM-inspired explainable QA pipeline for cardiology that
  outputs both text and structured graph artifacts.
- We define an automatic graph-quality evaluation protocol covering schema
  validity, simplified OPM constraint satisfaction, source grounding, and
  answer-graph consistency.
- We combine automatic metrics, a manual sanity check, and case studies to
  characterize when OPM graphs support explanation and when they fail.
- We provide a lightweight element-level audit of generated OPM components to
  examine where object-process-state reasoning remains reliable or fragile.

## 3. Related Work

Medical QA research has studied retrieval, language models, and
domain-specific supervision for answering biomedical and clinical questions
[REF]. These systems can produce useful text, but answer fluency does not by
itself establish transparent reasoning, source grounding, or clinical
validity.

Explainable AI work argues that high-stakes systems should expose
intermediate evidence, reasoning, or model behavior in forms that humans can
inspect [REF]. In medical QA, this motivates explanation artifacts that can be
evaluated separately from final answer text.

Knowledge graphs and formal modeling languages provide structured
representations for entities, relations, and mechanisms [REF]. OPM represents
systems through objects, processes, states, and links [REF]. We use an
OPM-inspired graph as an explicit QA explanation artifact and evaluate the
artifact separately from answer correctness.

## 4. Methodology

### 4.1 Overall Framework

Let the input be a free-text cardiology question \(q\). The system returns
either a fallback status or a tuple \((t, a, e, r, G)\), where \(t\) is the
matched topic, \(a\) is a short answer, \(e\) is a natural-language
explanation, \(r\) is an ordered reasoning path, and \(G\) is an exported
OPM-style graph. The output is intended for audit and explanation, not
clinical decision-making.

### 4.2 Cardiology QA Processing

The prototype uses a hand-built cardiology knowledge base with 21 topics.
Each topic stores matching cues, a short answer, an explanation, a 3-5 step
reasoning path, and OPM graph fields. Topic matching uses phrase hits and
shared content-token overlap. If no topic exceeds the matching threshold, the
system returns an explicit fallback rather than fabricating an answer.

### 4.3 Manual Evaluation Protocol

In addition to the 100-sample topic-level audit for each setting, we conducted
a lightweight OPM element-level audit on 20 manually annotated samples. This
additional audit compares generated objects, processes, states, and relations
with gold mechanism-level annotations. It is used as an exploratory analysis
of OPM element alignment rather than as a full clinical validation or
benchmark-level evaluation.

### 4.4 OPM Graph Construction

For each matched topic, the system constructs an OPM-style graph
\(G = (O, P, S, E)\). \(O\) is the set of object labels, \(P\) is the set of
process labels, \(S\) is the set of state labels, and \(E\) is the set of
typed links. Each link is a triple \((source, relationship, target)\), where
the source and target are graph labels or a terminal topic outcome. The
restricted relationship vocabulary includes `object participates in process`,
`process changes state`, and `state contributes to outcome`.

Topic matching, reasoning, and graph export are coupled through the selected
topic. Once a topic is selected, its answer and reasoning path provide the
textual explanation, while its OPM fields provide the structured graph. The
reasoning path is the ordered view of the mechanism; the graph is the typed
object-process-state view of the same explanation scaffold.

### 4.5 Explainable Answer Generation

The answer is accompanied by a natural-language explanation, the ordered
reasoning path, and the OPM graph. These artifacts make the selected topic and
intermediate mechanism inspectable, but they do not prove medical correctness
or guarantee that every answer token is grounded in graph evidence.

## 5. Experiments and Evaluation

### 5.1 Dataset

The graph-quality audit evaluates 543 graph files matched to 546 local
cardiology QA result rows; 543 rows include graph paths and 3 are fallbacks.
This is a heuristic graph-quality audit, not clinical validation, and should
not be interpreted as a full MedQA benchmark or medical performance result.

### 5.2 Evaluation Metrics

We evaluate four metric groups: schema validity, simplified OPM constraint
satisfaction, source grounding, and answer-graph consistency. Source
grounding uses normalized surface-form overlap between graph labels and the
available question, answer, explanation, or generated knowledge text, so
synonymy and paraphrase may be underestimated.

### 5.3 OPM Element-Level Extraction Audit

We additionally run a lightweight OPM element-level extraction audit on the
20 manually annotated samples. This audit compares predicted graph objects,
processes, states, and relations against draft gold annotations using exact
normalized matching plus a simple token-overlap match. It is intended to
measure graph extraction alignment, not clinical correctness.

| Group | Object F1 | Process F1 | State F1 | Relation F1 |
| --- | ---: | ---: | ---: | ---: |
| All samples | 0.3375 | 0.1418 | 0.0857 | 0.0247 |
| Graph-relevant | 0.4000 | 0.2857 | 0.1714 | 0.0484 |
| Clean cases | 0.6875 | 0.5714 | 0.2857 | 0.1304 |
| Topic mismatch | 0.2813 | 0.0000 | 0.0000 | 0.0000 |

The grouped results show that clean cases have substantially better object
and process alignment, while topic-mismatch cases collapse for process,
state, and relation alignment. Relation extraction remains the hardest part
of the task across all groups, suggesting that graph structure is easier to
produce than semantically accurate typed links. This result should be
interpreted as evidence of audit need rather than as evidence of mature
relation extraction performance.

### 5.4 Automatic OPM Graph Evaluation

Table 1 summarizes the main automatic metrics. The results show strong graph
well-formedness and OPM constraint satisfaction, with more moderate source
grounding and answer-graph consistency.

| Category | Metric | Mean |
| --- | --- | ---: |
| Schema validity | `valid_graph_rate` | 1.0000 |
| Schema validity | `isolated_node_ratio` | 0.0078 |
| OPM constraint satisfaction | `process_object_connectivity_rate` | 0.9435 |
| OPM constraint satisfaction | `state_object_attachment_rate` | 0.9509 |
| OPM constraint satisfaction | `explanation_path_rate` | 1.0000 |
| Source grounding | `overall_grounding_rate` | 0.3980 |
| Answer-graph consistency | `answer_concept_coverage` | 0.4103 |
| Answer-graph consistency | `graph_evidence_usage_rate` | 0.3364 |
| Answer-graph consistency | `path_support_rate` | 1.0000 |

### 5.5 Manual Sanity Check

We manually audited 20 samples from four buckets (`high_quality`,
`low_grounding`, `low_consistency`, and `diverse_topics`; five samples each).
The review shows that 19 of 20 graphs are at least partially relevant, but
only 4 of 20 are fully path-explainable. Partial explainability is the modal
outcome, and `topic_mismatch` is the most common failure mode.

| Field | Distribution |
| --- | --- |
| `graph_relevance` | yes 10, partial 9, no 1 |
| `path_explainability` | yes 4, partial 14, no 2 |
| `main_error_type` | topic_mismatch 8, none 4, over_generalization 4, weak_grounding 2, answer_graph_misalignment 2 |

### 5.6 Case Studies

The successful case `q0115` matches `aortic stenosis` and follows the chain
valve calcification -> aortic valve narrowing -> left ventricular outflow
obstruction -> pressure overload -> exertional dyspnea/chest pressure,
providing a useful mechanism scaffold despite not encoding the final numeric
continuity-equation calculation. In contrast, `q0003` asks about a congenital
heart defect in a neonatal infection scenario, but the matched topic is
`pulmonary embolism`; the resulting graph is internally coherent for
thromboembolism but unrelated to the expected congenital rubella and patent
ductus arteriosus mechanism.

### 5.7 Error Analysis

The main error pattern is alignment rather than graph syntax. The automatic
audit shows high schema validity and OPM constraint satisfaction, while
source grounding and answer-graph consistency remain moderate. The manual
audit shows the same pattern: many graphs are topically adjacent, but the
reasoning path often fails to support the exact tested mechanism or answer.
For example, `q0140` is partially aligned: the graph explains infective
endocarditis findings but not the downstream Streptococcus
gallolyticus/bovis-colorectal neoplasia association.

The `path_support_rate` of 1.0000 should be interpreted carefully. Under the
current heuristic, it indicates that at least some answer concepts overlap
with graph-path or reasoning-path concepts; it does not prove that the final
answer fully uses the graph. The lower answer concept coverage and graph
evidence usage rates provide a more conservative view of answer-graph
alignment.

## 6. Discussion

The results suggest that OPM-style graph artifacts can make a cardiology QA
prototype more auditable. The graph format exposes the matched topic,
mechanism steps, and object-process-state structure, which makes failures
easier to inspect. A graph may be structurally valid but too general, relevant
to a visible finding but not the requested answer, or mismatched to the
clinical mechanism.

The key implication is that the present bottleneck is semantic rather than
syntactic. Structural graph validity and simplified OPM constraint
satisfaction are strong, but source grounding and answer-level graph usage
remain moderate. The manual sanity check further shows that topic mismatch is
the dominant observed failure mode, indicating that better semantic routing
and answer-specific graph alignment are central next steps.

## 7. Limitations and Future Work

This work is a research prototype. The knowledge base is small and hand-built,
the matching strategy is simple, and the OPM representation is a simplified
custom format rather than a full OPM serialization. The automatic evaluation
is a heuristic graph-quality audit, not clinical validation. First, the
current evaluation focuses on topic relevance, error patterns, and an
exploratory element-level OPM audit, rather than comprehensively evaluating
full medical answer accuracy or complete OPM-path correctness. It does not
assess medical answer correctness, clinical completeness, or suitability for
deployment.

Future work should expand the cardiology knowledge base, improve topic
matching, add normalized medical concept linking, connect graph labels to
explicit source spans, and use semantic matching for synonyms and paraphrases.
Graph-grounded answer generation should also make the supporting OPM path
explicit, so that final answers can be checked against the graph rather than
merely accompanied by it.

## 8. Conclusion

We introduced an OPM-based explainable QA framework for cardiology and
evaluated its generated graph artifacts. The system produces structurally
valid OPM-style graphs with strong simplified constraint satisfaction, but
source grounding and answer-graph consistency remain incomplete. Manual
inspection and case studies show that accurate topic matching can yield useful
explanation scaffolds, whereas failed topic alignment limits case-level
explainability. These findings motivate future work on stronger semantic
grounding and graph-faithful answer generation.
