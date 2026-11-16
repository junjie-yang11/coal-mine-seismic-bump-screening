# 煤矿下一班次高能矿震分类预测研究

本研究用 UCI Seismic Bumps 的 2,584 条班次记录，检验班次监测信息能否为下一班次高能矿震提供危险排序，并比较四个模型在相同复核名额下的检出数。

This study evaluates next-shift seismic-bump screening using nested, stratified group cross-validation and compares hazardous records detected under equal review workloads.

**当前稿为文献与格式修订版。** [阅读 PDF](论文/煤矿下一班次高能矿震分类预测研究_文献与格式修订版.pdf) · [直接下载 PDF](https://raw.githubusercontent.com/junjie-yang11/coal-mine-seismic-bump-screening/main/%E8%AE%BA%E6%96%87/%E7%85%A4%E7%9F%BF%E4%B8%8B%E4%B8%80%E7%8F%AD%E6%AC%A1%E9%AB%98%E8%83%BD%E7%9F%BF%E9%9C%87%E5%88%86%E7%B1%BB%E9%A2%84%E6%B5%8B%E7%A0%94%E7%A9%B6_%E6%96%87%E7%8C%AE%E4%B8%8E%E6%A0%BC%E5%BC%8F%E4%BF%AE%E8%AE%A2%E7%89%88.pdf) · [Word 编辑稿](论文/煤矿下一班次高能矿震分类预测研究_文献与格式修订版.docx)

## 主要结果

四种模型的五折平均精确率（AP）为 0.2039–0.2320，恒定分数基线为 0.0658。各折使用各自冻结阈值时，随机森林比逻辑回归多检出 24 条危险记录，同时多出 220 条待复核记录。按各折前 20% 统一名额后，两者各复核 515 条，分别检出 100 条和 98 条。

[图 8：冻结阈值和相同名额下的检出比较](图件/图8_检出与复核名额/图8_检出与复核名额.png) · [等复核量汇总](补充分析/等复核量比较/results/pooled_results.csv)

## 从论文找到代码和结果

| 步骤 | 做了什么 | 对应文件 |
| --- | --- | --- |
| 1. 嵌套五折评价 | 在外层训练数据内选参和定阈值，评价同一批外层测试记录 | [锁定方案](实验/locked_plan.json) · [执行入口](实验/run_nested.py) · [五折汇总](实验/results/model_summary.csv) · [逐行预测](实验/results/outer_predictions.csv) |
| 2. 相同名额比较 | 读取已保存的外层分数，在各折内按相同名额比较危险检出 | [分析说明](补充分析/等复核量比较/README.md) · [计算脚本](补充分析/等复核量比较/code/run_comparison.py) · [独立核验](补充分析/等复核量比较/code/verify_comparison.py) |
| 3. 出图与成稿 | 从保存结果生成图件，Word 为编辑稿，PDF 为阅读稿 | [图 8 源数据](图件/图8_检出与复核名额/图8_检出与复核名额.csv) · [图 8 脚本](代码/绘制图8.py) · [PDF 导出](代码/导出PDF.ps1) |

## 文件组织

- `论文/` 集中存放当前 Word 和 PDF。GitHub 上一版可见于[发布历史](https://github.com/junjie-yang11/coal-mine-seismic-bump-screening/commit/e788e14a66b921a43c42dee2d9483dd86891a433)。
- `实验/` 包含原始数据、训练与评价脚本、模型、结果和回放核验。
- `补充分析/等复核量比较/` 包含固定名额分析的输入、计算、输出和核验。
- `图件/` 按格式保留原图及其来源和检查记录。论文使用的新版图 8 集中在 `图件/图8_检出与复核名额/`。
- `代码/` 包含绘图、图件检查和 PDF 导出工具。`打开材料.html` 是下载仓库后的本地阅读入口。

## 核验与重绘

从仓库根目录运行。实验依赖见[环境文件](实验/requirements.txt)，绘图依赖见[绘图环境](代码/requirements-绘图.txt)。重绘依赖 Windows 的微软雅黑与 Arial，PDF 导出使用 Microsoft Word。

```powershell
python 实验/test_edges.py
python 实验/verify_nested.py
python 补充分析/等复核量比较/code/verify_comparison.py
python 代码/绘制图8.py --output ../图8重绘检查
```

原始数据来自 Sikora 与 Wrobel（2010）的 [UCI Seismic Bumps](https://doi.org/10.24432/C5W902)，采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 授权。图件检查工具的上游和许可见[来源说明](代码/图件检查工具/来源说明.md)。

本文评价公开数据集内部的离线排序与筛查能力。数据缺少逐条时间戳和工作面标识。复核量以入选记录数计，实际检查工时和现场预警效果需要现场验证。
