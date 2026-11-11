# 项目来源

- 建立日期：2026-11-11（Asia/Tokyo）。
- 原仓库：https://github.com/aiqishun/opm-medical-qa 。
- 迁移时原仓库 HEAD：`245dacc37c32634e242485e928782d268e855724`。
- 新仓库使用独立 Git 历史；通过本说明及本地备份追溯原版本。
- 输入为当前工作目录，包含未提交和未跟踪文件，并非仅使用 HEAD。
- 会议最终发表源码所对应的提交尚未确认；不能据此声明原目录就是最终发表版。
- 原仓库文件、分支、提交和远程配置不作修改。

## 本地备份

快照：`pre-thesis-working-copy-20261111.tar.gz`。

SHA-256：`8ef581602022fbcf74a5e5a5af240976eb3cab55be46d080a90323ea95193cc7`。

快照保留原 Git 历史、当前源码、数据和研究结果。排除虚拟环境、
Python 缓存、pytest 缓存和系统元数据；这些内容可重新生成。
快照仅本地保存，不上传到新仓库。

## 导入范围

导入核心源码、研究脚本、现有测试、样例和知识库、人工标注，
以及结构完整的既有 MedQA 评测交付包。补充原推理入口及其依赖的两份
处理后数据，因此新仓库的推理映射检查和交付包构建可以运行。
保留上游数据的 README 和 LICENSE，不复制上游 Git 元数据。

原工作目录的论文与报告归档到 `references/conference/`；
原目录中未导入的新旧批量结果仍在本地快照内。
`medqa_opm_task/legacy_project/` 是原评测任务依赖的组件，保留其相对路径。

继承的数据和结果保持原值。迁移本身不修正标签、不重新运行模型、
不产生毕业论文新结论。后续清洗与方法改动应分别提交并记录。

## 迁移时的原工作区状态

```text
 M .gitignore
 M README.md
 M paper/main_paper_draft.md
 M requirements.txt
?? annotations/opm_extraction_gold_20.csv
?? data/contracts/
?? delivery/
?? ground_truth_source_files.zip
?? medqa_opm_task/
?? paper/main_paper_draft.pdf
?? pytest.ini
?? run_inference.py
?? scripts/build_task_delivery.py
?? scripts/evaluate_opm_extraction_accuracy.py
?? scripts/generate_gold_test.py
```
