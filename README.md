# OPM Medical QA — Thesis

毕业论文源码仓库，基于既有 OPM 心血管问答研究项目建立。
本仓库独立维护，原项目：https://github.com/aiqishun/opm-medical-qa 。

这是毕业论文工作的起点。继承的论文、指标和运行记录属于既有研究材料，
不代表已经完成毕业论文新增实验。会议最终发表版对应的版本仍需核对。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/run_qa.py --question "What causes myocardial infarction?"
python -m pytest -q
```

核心问答演示使用 Python 标准库。可选模型调用使用 `OPENAI_API_KEY` 环境变量；
不要把密钥写进源码、提交文件或上传到 GitHub。

## 目录

| 目录 | 用途 |
| --- | --- |
| `src/`、`scripts/`、`tests/` | 继承的基础代码、研究脚本与测试 |
| `data/` | 演示数据、知识库和输出契约 |
| `annotations/` | 既有人工标注及标注说明 |
| `medqa_opm_task/` | 完整的既有评测任务及推理脚本，保留原标签 |
| `experiments/`、`outputs/` | 毕业版新实验输出位置，生成物默认忽略 |
| `paper/` | 毕业论文正文与提纲 |
| `references/conference/` | 从原工作目录继承的论文和研究记录 |
| `docs/` | 来源、数据审计与毕业论文计划 |

## 既有评测包

先阅读 `medqa_opm_task/README.md`。校验器入口为
`python medqa_opm_task/tools/validate_submission.py --submission <目录>`。
没有提交文件时，校验器返回缺失文件清单是预期行为。

```bash
# 只检查 100 道题的 ID 映射，不调用模型、不写结果
python run_inference.py --dry-run

# 从源码重新构建一个独立交付包
python scripts/build_task_delivery.py
```

`medqa_opm_task` 含供研究者使用的参考答案、评测器和历史运行。
若组织独立评测，须按其 `task.json` 隔离参测者可见内容。
其中 12 道题跨开发集与测试集重复，2 道标签冲突，详情见
[数据审计](docs/DATA_AUDIT.md)。在处理前，不将其视作独立的毕业论文测试集。

## 研究记录

- [项目来源和迁移边界](docs/PROJECT_ORIGIN.md)
- [毕业论文计划](docs/THESIS_PLAN.md)
- [既有材料说明](references/conference/README.md)

完整的原工作目录快照在本地另行备份。虚拟环境、缓存、重复压缩包、
上游仓库的 `.git` 目录和批量图输出不纳入新仓库。
MedQA 来源和 MIT 许可说明位于 `medqa_opm_task/reference/`；
项目自身尚未指定单独的软件许可。

本项目用于研究和教育，不用于临床诊断、治疗或决策。
