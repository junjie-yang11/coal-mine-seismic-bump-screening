"""Build a short Chinese result report from actual computed CSVs."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def read(name):
    with (ROOT/'results'/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
summary=read('pooled_results.csv');folds=read('fold_results.csv');pairs=read('paired_fold_comparisons.csv');ties=read('cutoff_tie_audit.csv')
models=['LR','RF','RBF-SVM','XGBoost']
def row(setting,model):return next(r for r in summary if r['setting']==setting and r['model']==model)
lines=['# 等复核量比较结果','日期：2026年11月8日。基于已经保存的五折外层预测，未重新训练、调参或改变原冻结阈值。',
'## 核心判断',
'在每折相同20%复核名额下，RF汇总检出100条、LR为98条，RF点估计略高，优势为2条。按各折LR原名单长度匹配后，RF为97条、LR为94条，相差3条。原冻结阈值下RF多检出24条的结果仍成立，但该结果同时伴随较长名单，不能当作同工作量下的检出优势。',
'本轮结果支持将文章表述为：RF在所检查名额下具有较好的汇总检出表现，但相对LR的差距随名额和外层测试折变化；当前不能据此宣称RF具有稳定或显著的现场优势。',
'## 1. 比较如何进行',
'- 采用原有外层五折，每条记录沿用其外层测试分数，四模型使用相同记录和分区。',
'- 主比较为每折前20%，10%和30%作补充；固定名额K=floor(比例×该折样本数)，分别在每折排序，未跨折混合分数。',
'- 另以各折LR原冻结规则的判危险数作为K，称为LR名单长度参照；该口径与独立固定容量不同。',
'- 分数降序；并列时按预先固定的源行号SHA256优先级选取。为了严格相同名额，可以拆分边界并列组，故本分析是整批Top-K参照，不是新的数值阈值或实时策略。',
'- 测试标签只计算检出、误报和漏报，不参与名额、排序或并列规则。方案在本次计算前保存；既有数据和旧结果此前已查看，本轮属于新增回顾性补充，不是首次未见数据验证或前瞻注册。',
'## 2. 相同20%名额的主要结果',
'前四折各517条，第五折516条；每折均取103条，合计515条，占总记录19.93%。危险记录总数170条。',
'| 模型 | 复核记录 | 检出TP | 漏报FN | 误报FP | 召回率 | 精确率 |','|---|---:|---:|---:|---:|---:|---:|']
for m in models:
 r=row('fixed_20pct',m);lines.append(f"| {m} | {r['review_count']} | {r['tp']} | {r['fn']} | {r['fp']} | {float(r['recall']):.2%} | {float(r['precision']):.2%} |")
lines+=['','RF比LR多检出2条，召回率高1.18个百分点，在相同名单长度下少2条误报。这是本批记录上的描述性差值，不等于经过显著性检验的优势。',
'### 逐折检出','| 外层折 | 每模型名额K | LR | RF | RBF-SVM | XGBoost | RF−LR |','|---|---:|---:|---:|---:|---:|---:|']
diffs=[]
for f in range(1,6):
 rr={r['model']:r for r in folds if r['setting']=='fixed_20pct' and int(r['outer_fold'])==f};delta=int(rr['RF']['tp'])-int(rr['LR']['tp']);diffs.append(delta)
 lines.append('| '+ ' | '.join(map(str,[f,rr['LR']['review_count'],*[rr[m]['tp'] for m in models],f'{delta:+d}']))+' |')
lines += ['',f'RF相对LR在{sum(x>0 for x in diffs)}折领先、{sum(x<0 for x in diffs)}折落后、{sum(x==0 for x in diffs)}折持平。五折共享部分训练记录，未把五个差值当作独立样本做检验。']
rr=[r for r in pairs if r['setting']=='fixed_20pct' and r['model_a']=='LR' and r['model_b']=='RF']
shared=sum(int(r['shared_tp']) for r in rr);lr_only=sum(int(r['a_only_tp']) for r in rr);rf_only=sum(int(r['b_only_tp']) for r in rr)
lines += [f'同名额下，RF与LR共同检出{shared}条，RF独有检出{rf_only}条，LR独有检出{lr_only}条，因此净差为{rf_only-lr_only}条；两份名单并非包含关系。',
'## 3. 名额敏感性与LR名单长度参照','| 口径 | 每模型合计名额 | LR检出 | RF检出 | RBF-SVM检出 | XGBoost检出 | RF−LR |','|---|---:|---:|---:|---:|---:|---:|']
for setting,label in [('fixed_10pct','每折10%'),('fixed_20pct','每折20%（主比较）'),('fixed_30pct','每折30%'),('LR_count_reference','各折LR原名单长度')]:
 vals=[row(setting,m) for m in models];delta=int(vals[1]['tp'])-int(vals[0]['tp']);lines.append('| '+' | '.join(map(str,[label,vals[0]['review_count'],*[r['tp'] for r in vals],f'{delta:+d}']))+' |')
lines+=['','RF在所检查口径的汇总检出数最高，和LR的差额分别为2、2、10、3条。这说明领先幅度随名额变化，不能据此选择对RF最有利的名额作为新的主结论。',
'### LR名单长度参照的逐折结果','| 折 | K | LR | RF | RBF-SVM | XGBoost |','|---|---:|---:|---:|---:|---:|']
for f in range(1,6):
 rr={r['model']:r for r in folds if r['setting']=='LR_count_reference' and int(r['outer_fold'])==f};lines.append('| '+' | '.join(map(str,[f,rr['LR']['review_count'],*[rr[m]['tp'] for m in models]]))+' |')
lines+=['','LR在该参照下恢复原冻结名单，合计94条检出；RF按相同逐折名额排序后为97条。该比较直接补充原RF/LR工作点对照，但不是对原24条差额的唯一因果分解。',
'## 4. 并列分数审计',
f'共审计80个“口径×模型×折”组合，其中{sum(r["tie_split"]=="True" for r in ties)}个在边界拆分并列：']
for r in ties:
 if r['tie_split']=='True':lines.append(f"- {r['setting']}，第{r['outer_fold']}折，{r['model']}：边界{r['cutoff_tie_count']}条，纳入{r['cutoff_tie_selected']}条。")
assert all(r['tp_min_cutoff_tie']==r['tp_max_cutoff_tie'] for r in ties)
lines+=['所有80个组合的边界并列检出上下界均重合，当前检出数不受这些边界并列选择影响。上下界是只改变边界并列纳入对象的组合范围，不是置信区间。',
'## 5. 原冻结规则的净增复核负担',
'原冻结规则下，RF复核701条、检出118条；LR复核481条、检出94条。净增加220条复核和24条检出，220/24=9.17，约为每净增一条危险检出对应9.2条额外待复核记录。',
'该比值只描述原冻结名单的净差，不能称为实际经济成本、检查工时或因果边际成本，也不应套用到本轮相同Top-K名单。',
'## 6. 建议写入论文的结果表述',
'> 在相同复核名额下进一步比较四模型的危险检出。每个外层折按分数选取前20%的记录，各模型合计复核515条，RF、LR、RBF-SVM和XGBoost分别检出100、98、94和90条危险记录。RF相对LR的净增检出为2条，逐折差值为−2、+4、+1、−1和0。以各折LR原冻结规则的名单长度为参照时，RF与LR分别检出97和94条。结果表明，RF在当前同名额比较中的汇总检出略高，但优势随名额和测试分区变化；原冻结规则下增加24条检出同时伴随增加220条待复核记录，不能将其全部解释为同工作量下的排序收益。',
'这项补充使研究结论更具体：现有输入能够提供危险排序信息，但模型之间的使用差异还取决于名单长度和所选择的工作点。当前结果没有证明算法等效，也没有证明某个现场最优规则。',
'## 7. 文件与核验范围',
'- `analysis_plan.json`：计算前方案、名额、并列规则、解释边界及输入哈希。',
'- `code/run_comparison.py`：标准库Python分析脚本；从本目录运行 `python code/run_comparison.py`。',
'- `code/verify_comparison.py`：独立检查名单排序、源行对齐、容量及全部计数；运行 `python code/verify_comparison.py`。',
'- `inputs/`：原始保存预测和原指标的只读用途副本。',
'- `results/fold_results.csv`、`pooled_results.csv`：逐折与合并结果。',
'- `results/ranked_decisions.csv`：41344行逐条排序、名额与入选决定。',
'- `results/cutoff_tie_audit.csv`：80组边界并列审计。',
'- `results/paired_fold_comparisons.csv`：四模型六种配对的逐折检出重叠。',
'- `results/original_frozen_threshold_results.csv`、`incremental_review_burden.json`：原工作点与9.2比值。',
'- `checks/run_record.json`、`checks/verification.json`：实际运行及独立核验结果。',
'执行成功不能证明现场有效性。本轮未重训、未回放模型、未给出总体置信区间；原论文和实验结果文件保持不变。']
(ROOT/'README.md').write_text('\n\n'.join(lines[:0]) if False else '\n'.join(lines)+'\n',encoding='utf-8')
print(ROOT/'README.md')
