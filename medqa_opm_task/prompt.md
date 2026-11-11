# MedQA 心血管主题路由与 OPM 解释可靠性复现

## 任务目标

研究现有主题路由在何种医学语境下发生误判，检验语境感知筛选对心血管候选队列的影响，并审计 OPM 图是否反映题目的主要医学过程。

本任务是研究评估，不用于临床诊断、治疗或患者决策。

## 可用材料

- `data/input/dev_questions.jsonl`：开发集题目。
- `data/reference/dev_labels.jsonl`：公开开发集人工标签。
- `data/input/test_questions.jsonl`：测试集题目；测试标签不公开。
- `data/processed/cardiology_knowledge.json`：现有主题与 OPM 知识结构。
- `data/contracts/`：提交文件的数据契约。
- `legacy_project/`：现有匹配、推理、图生成与评估代码。
- `reference/`：数据来源、许可证和人工标注规范。
- `tools/validate_submission.py`：不读取隐藏答案的本地格式验证器。

开发集和测试集的 `id` 分别在各自文件内唯一；相同字符串 ID 可以在两个 split 中表示不同记录。跨文件关联必须使用 `(split, id)`，不得只用 `id` 覆盖记录。

不得访问 `answer/`、`grader/`、`rubrics.md`、`judge_prompt.md`、`traces/` 或其他隐藏评分材料。不得在线搜索题库答案，也不得编造不存在的题目、标注、图结构或实验结果。

## 必须完成的处理

1. 审计 JSONL、已有结构化字段和流程输出之间的题目映射，检查重复记录、缺失字段、跨流程 ID 冲突及无法回溯的记录。所有排除、合并和纠正决定必须保留原始位置与处理依据。
2. 根据主要诊断、机制或处置目标判断是否属于心血管研究范围。必须实际处理题干、选项、`metamap_phrases` 或功能等价的结构化文本结果，并区分主要考点、既往史、危险因素、否定信息、治疗语境和干扰项。不得仅以关键词是否出现完成分类。
3. 运行并检查现有主题匹配与 OPM 生成流程。逐题审计对象、过程、状态和关系是否来自当前题目，关系方向是否合理，解释路径是否连续。图能够导出不等于医学语义正确。
4. 在同批可比开发样本上比较原始流程与改进流程，分别报告队列判定、主题一致性和 OPM 解释质量，并把变化归因到数据映射、语境判断、主题匹配或图生成环节。不得预设改进方法优于基线。

连续处理链为：

`题目与记录映射 → 医学短语及语境处理 → 心血管判定 → OPM 结构检查 → 对照实验与错误归因`

## 提交文件

所有结果必须写入 `submission/`：

- `data_audit.json`
- `filter_results_dev.jsonl`
- `filter_results_test.jsonl`
- `semantic_annotations.jsonl`
- `opm_audit_results.jsonl`
- `metrics.json`
- `research_report.md`
- `run_all.sh`

`filter_results_dev.jsonl` 与 `filter_results_test.jsonl` 的每条记录必须符合 `data/contracts/filter_results.schema.json`：

```json
{"id":"q0026","is_cardiology":true,"reason":"主要考查心律失常的诊断"}
```

可以填写契约允许的可选字段，但不得增加契约未声明的字段。

`semantic_annotations.jsonl` 必须记录 split、题目 ID、医学短语、语境角色和原文字符位置。`opm_audit_results.jsonl` 必须记录 split、题目 ID、匹配主题、对象、过程、状态、关系、结构有效性、语义一致性、证据和 `opl_text`。

`opl_text` 必须满足本任务的数据契约，其状态声明形式为：

```text
Heart rate can be high, normal, low. State high is initial.
```

这是一条任务级格式约束；不要把通过该字符串检查表述为对完整 ISO OPM 规范的认证。

## 指标与报告边界

开发集必须报告 Precision、Recall、F1、样本数及计算分母。测试集没有公开参考标签，不得在提交报告中声称测试集 Precision、Recall、F1 或优于基线。

最终报告必须将内容分为：

- 由文件或程序直接验证的事实；
- 基于已列结果的解释；
- 尚未验证的限制。

只列出工具名称而不提供实际处理产物或运行记录，不视为完成相应处理。
