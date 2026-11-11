# 煤矿下一班次高能矿震分类预测研究

当前成果为**等复核量整合版**，共15页、10张Python图件。研究以一个问题为主线：班次监测数据形成的危险排序，能否在有限复核名额下转化为更多高能矿震记录检出。

This study compares four classifiers for next-shift seismic-bump screening and evaluates how many hazardous records each ranking captures at the same review workload.

四种模型的五折平均精确率（AP）为0.2039～0.2320，是恒定分数基线的3.1～3.5倍。原冻结阈值下，随机森林（RF）较逻辑回归（LR）多检出24条危险记录，但名单也增加220条；统一为每折前20%复核名额后，RF与LR均复核515条，分别检出100条和98条。结果说明，模型应在相同复核资源下比较，不能把更长名单带来的检出增加直接解释为更强的排序能力。

## 从这里开始

1. 阅读[论文 PDF](论文/煤矿下一班次高能矿震分类预测研究_等复核量整合版.pdf)，也可[直接下载 PDF](https://raw.githubusercontent.com/junjie-yang11/coal-mine-seismic-bump-screening/main/%E8%AE%BA%E6%96%87/%E7%85%A4%E7%9F%BF%E4%B8%8B%E4%B8%80%E7%8F%AD%E6%AC%A1%E9%AB%98%E8%83%BD%E7%9F%BF%E9%9C%87%E5%88%86%E7%B1%BB%E9%A2%84%E6%B5%8B%E7%A0%94%E7%A9%B6_%E7%AD%89%E5%A4%8D%E6%A0%B8%E9%87%8F%E6%95%B4%E5%90%88%E7%89%88.pdf)。另附[当前 Word 稿](论文/煤矿下一班次高能矿震分类预测研究_等复核量整合版.docx)。PDF 早于 Word 的最后一次保存，参考文献格式及部分 DOI／URL 有差异。
2. 查看[论文整合说明](论文/整合说明.md)与[等复核量结果说明](补充分析/等复核量比较/README.md)。
3. 查看[五折汇总](实验/results/model_summary.csv)、[逐折结果](实验/results/outer_metrics.csv)、[逐行预测](实验/results/outer_predictions.csv)及[等复核量汇总](补充分析/等复核量比较/results/pooled_results.csv)。
4. 查看[锁定方案](实验/locked_plan.json)、[拟合与评价代码](实验/run_nested.py)、[模型回放核验](实验/verify_nested.py)及[等复核量独立核验](补充分析/等复核量比较/code/verify_comparison.py)。

## 目录

```text
最终版/
├─ 打开材料.html     浏览器阅读入口
├─ README.md         本说明
├─ 论文/             当前 Word、PDF 和整合说明
├─ 图件/
│  ├─ PNG/           10张600dpi图片
│  ├─ PDF/           10张矢量图
│  ├─ SVG/           10张可编辑文字图
│  ├─ 源数据/        曲线、相关矩阵、混淆计数等
│  └─ QA/            图件检查与来源记录
├─ 代码/             绘图、Word整合、PDF导出与图件同步工具
├─ 实验/             原嵌套五折实验、模型、结果和核验记录
└─ 补充分析/
   └─ 等复核量比较/
      ├─ README.md            结果摘要与文件说明
      ├─ analysis_plan.json   锁定的补充分析方案
      ├─ code/                计算、核验与报告脚本
      ├─ inputs/              保存预测与指标的来源副本
      ├─ results/             逐条名单及汇总结果
      └─ checks/              运行、迁入及论文整合核验记录
```

## 使用和执行顺序

Word是正文编辑源，PDF是阅读稿。等复核量分析只读取原实验保存的外层分数，不重新拟合模型，也不改动原阈值。

```powershell
# 原模型与计算核验（使用原实验环境）
python 实验/verify_nested.py
python 实验/test_edges.py

# 等复核量名单与计数核验（标准库Python即可）
python 补充分析/等复核量比较/code/run_comparison.py
python 补充分析/等复核量比较/code/verify_comparison.py

# 使用绘图环境重画图件；输出到独立目录
python 代码/绘图.py --project 实验 --output ../重绘检查输出
python 代码/核查图件.py --root ../重绘检查输出
```

实验环境沿用`实验/requirements.txt`，绘图环境见`代码/requirements-绘图.txt`。绘图需要Windows微软雅黑及Arial字体，PDF导出需要Microsoft Word。完整重新训练应在新的空目录中运行`run_nested.py`，原入口会拒绝覆盖已有模型和结果。

`代码/合并等复核量结果.py` 保留了整合过程，执行时需要通过 `--source` 提供符合脚本哈希校验的历史 Word 原稿。仓库提供的是当前稿。

## 证据与边界

数据来源为 Sikora 与 Wrobel（2010）的 [UCI Seismic Bumps](https://doi.org/10.24432/C5W902)，采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 授权。图件检查工具的上游及授权见[来源说明](代码/图件检查工具/来源说明.md)。

原实验保留数据来源哈希、外层与内层索引、180组候选审计、训练折外分数、全部外层预测、25个保存模型和真实运行记录。等复核量模块保留80组“口径×模型×折”名单、41344行排序决定及边界并列审计。

本文回答的是公开数据集内部的离线排序与筛查问题。数据缺少逐条时间戳和工作面标识，因此随机分层分组五折不代表时间或跨工作面验证；复核量按TP+FP定义，也不等同于真实检查工时或事故避免效果。
