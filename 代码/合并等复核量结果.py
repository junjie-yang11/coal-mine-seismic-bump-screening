"""Rebuild the bounded manuscript integration from retained source and CSVs."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile
from lxml import etree
from docx import Document
from docx.shared import Mm, Pt
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table

parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parent.parent)
parser.add_argument('--source',type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args();project=args.project.resolve()
source=args.source or project/'论文/历史版本/文献与图件修订_20261107/煤矿下一班次高能矿震分类预测研究_文献与图件修订版.docx'
output=args.output or project/'论文/煤矿下一班次高能矿震分类预测研究_等复核量整合版.docx'
module=project/'补充分析/等复核量比较'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='1e4caf2b08d1b81cb14727b3ddc0f1f234a68611731149fdbc34bf065b50cd07','原稿已改变，需重新核对段落修改。'
plan=json.loads((module/'analysis_plan.json').read_text(encoding='utf-8'))
for n,h in plan['inputs'].items():
    assert hashlib.sha256((module/'inputs'/n).read_bytes()).hexdigest()==h
    assert hashlib.sha256((project/'实验/results'/n).read_bytes()).hexdigest()==h
def read(n):
    with (module/'results'/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
pooled=read('pooled_results.csv');folds=read('fold_results.csv')
models=['LR','RF','RBF-SVM','XGBoost']
def result(setting,m):return next(r for r in pooled if r['setting']==setting and r['model']==m)
assert [int(result('fixed_20pct',m)['tp']) for m in models]==[98,100,94,90]
assert [int(result('LR_count_reference',m)['tp']) for m in models]==[94,97,92,87]
d=Document(source);changes=[]
for section in d.sections:
    section.page_width=Mm(210)
    section.page_height=Mm(297)
def find(prefix):
    matches=[p for p in d.paragraphs if p.text.startswith(prefix)]
    assert len(matches)==1,(prefix,len(matches))
    return matches[0]
def settext(p,text):
    assert not p._p.xpath('.//w:drawing | .//m:oMath')
    old=p.text;rp=next((deepcopy(r._r.rPr) for r in p.runs if r.text and r._r.rPr is not None),None)
    p.clear()
    if text.startswith('(2) '):
        lead,rest=text.split('。',1);p.add_run(lead+'。').bold=True;p.add_run(rest)
    else:
        r=p.add_run(text)
        if rp is not None:r._r.insert(0,rp)
    changes.append({'old':old,'new':text})
def replace(prefix,text):settext(find(prefix),text)
normal_template=find('RF 的平均外层 AP')._p
caption_template=find('表 5 全部外层判定')._p
heading_template=find('4.4 输入班次')._p
table_template=deepcopy(d.tables[4]._tbl)
def insert_p(before,text,kind='normal'):
    el=deepcopy({'normal':normal_template,'caption':caption_template,'heading':heading_template}[kind])
    p=Paragraph(el,d._body);settext(p,text)
    before.addprevious(el)
    return p
def insert_table(before,headers,rows):
    el=deepcopy(table_template);t=Table(el,d._body)
    assert len(t.columns)==len(headers)==7
    prototypes=[deepcopy(t.rows[1]._tr),deepcopy(t.rows[2]._tr)]
    for row in list(t.rows)[1:]:el.remove(row._tr)
    for i,cell in enumerate(t.rows[0].cells):cell.text=str(headers[i])
    for i,row in enumerate(rows):
        el.append(deepcopy(prototypes[i%2]))
        for j,cell in enumerate(t.rows[-1].cells):
            p=cell.paragraphs[0];settext(p,str(row[j]))
    t.autofit=False
    widths=[22,28,20,20,28,28,23]
    for col,width in zip(t.columns,widths):col.width=Mm(width)
    for i,row in enumerate(t.rows):
        for j,cell in enumerate(row.cells):
            cell.width=Mm(widths[j])
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next=i<len(t.rows)-1
                for run in p.runs:run.font.size=Pt(9);run.bold=i==0
    before.addprevious(el)

replace('摘要','摘要   本研究检验班次监测数据能否为下一班次高能矿震筛查提供有效排序，并量化不同模型的检出—复核取舍。基于 UCI Seismic Bumps 的 2,584 条记录，比较逻辑回归（LR）、随机森林（RF）、RBF-SVM 和 XGBoost；采用外层五折、内层五折的嵌套分层分组交叉验证，将相同输入归组，模型配置与分类阈值均在对应外层训练数据内确定。四模型的五折平均精确率（AP）为 0.2039～0.2320，是恒定分数基线平均 AP 的 3.1～3.5 倍，说明现有班次输入包含明确的危险排序信息。RF 的平均 AP 最高，但与 RBF-SVM 接近，逐折领先模型也发生变化。原冻结阈值下，RF 较 LR 多检出 24 条危险记录，同时增加 220 条待复核记录；每折统一取分数最高的前20%后，四模型均复核515条，RF、LR、RBF-SVM和XGBoost分别检出100、98、94和90条，RF与LR差距缩小至2条。四模型在原冻结规则下共同漏报46条。研究表明，模型差异主要应在给定复核名额下比较：班次监测数据能够支持离线危险排序，但更长的筛查名单会显著改变检出数量。')
replace('本文采用统一的嵌套','本文采用统一的嵌套分层分组五折流程开展分析。首先，通过逐折 AP 及均值检验班次输入是否包含危险排序信息；其次，比较训练内冻结规则下的检出、漏报、误报和复核量，并在相同复核名额下直接比较四模型；最后，按源行号对齐原冻结判定，识别独有检出和共同漏报记录。整套分析围绕一个核心问题展开：班次监测数据形成的排序，能否在有限复核名额下转化为更多危险记录检出。')
before=find('4 结果')._p
insert_p(before,'3.4 相同复核名额的补充比较','heading')
insert_p(before,'为比较相同名单长度下的检出表现，在原嵌套评价完成后，利用保存的外层测试分数开展回顾性批次Top-K分析。每个外层折分别排序，各模型取相同的 K 条记录。主比较取该折记录数的20%，K=⌊0.20N⌋；10%和30%作为名额敏感性比较。另以各折LR原冻结规则的判危险数为K，作为LR名单长度参照。该补充不重新训练模型、选择配置或修改原数值阈值。')
insert_p(before,'按模型分数降序取前K条，边界分数并列时按预先固定、与标签无关的源行号哈希顺序补足名额，允许拆分并列组以保证严格相同的K。测试标签仅评价名单；并列规则及名额在本次补充计算前保存，但数据和原结果此前已查看。分析使用该折整批分数，属于离线排名参照，不作为实时预警策略。各折计数相加报告检出、漏报、误报、召回率和精确率，逐折结果与完整名额组合一并保留。')
before=find('4.4 输入班次')._p
insert_p(before,'4.4 相同复核名额下的危险检出','heading')
insert_p(before,'每折前20%的名额均为103条，五折各模型合计复核515条，占全部记录的19.93%。RF、LR、RBF-SVM和XGBoost分别检出100、98、94和90条危险记录（表6）。RF较LR多检出2条，召回率分别为58.82%和57.65%；相同名单长度下，RF相应少2条误报。')
insert_p(before,'表 6 相同20%复核名额下的外层检出结果','caption')
rows=[]
for m in models:
    r=result('fixed_20pct',m);rows.append([m,r['review_count'],r['tp'],r['fn'],r['fp'],f"{float(r['recall']):.2%}",f"{float(r['precision']):.2%}"])
insert_table(before,['模型','复核数','TP','FN','FP','召回率','精确率'],rows)
insert_p(before,'注：各折分别排序并取前K条，K向下取整；计数覆盖全部2,584条外层测试记录，其中危险记录170条。本表采用批次Top-K名单，与表5的原冻结数值阈值判定不同。')
insert_p(before,'RF−LR的逐折检出差为−2、+4、+1、−1和0，表现为2折领先、2折落后、1折持平。同名额下两模型共同检出87条，RF独有检出13条、LR独有检出11条。以各折LR原名单长度为参照，各模型合计复核481条，RF与LR分别检出97和94条。10%、30%及LR名额参照见表A2，20%逐折结果见表A3。RF在所比较名额下的汇总检出最高，但相对差距随名额和测试折变化。')
replace('4.4 输入班次','4.5 输入班次类型与检出差异')
replace('4.5 检出组合','4.6 检出组合与共同漏报')
replace('RF 的较高检出率伴随','原冻结规则把模型差异直接转化为不同长度的复核名单：RF较LR多检出24条，同时增加220条待复核记录，其中净增误报196条。按两套规则的净差计算，每净增一条危险检出对应约9.2条额外待复核记录（220/24）。这个比值概括了原工作点的检出—复核交换关系，不表示实际经济成本或复核工时。')
replace('RF 和 LR 分别独自','等名额比较显示，原冻结规则下24条检出差距同时伴随220条名单长度差异。每折统一取前20%时，RF与LR均复核515条，检出差额缩小为2条；匹配各折LR原名单长度时，差额为3条。20%名额下RF逐折为2折领先、2折落后、1折持平，小幅汇总优势没有在各分区保持同一方向。这个比较回答固定复核资源下谁检出更多，不对原24条差额作因果拆分。实际模型选择应先明确可用复核名额，再比较该名额下的危险检出。')
replace('(2) 危险检出','(2) 复核名额决定模型比较的工程含义。原冻结规则下，RF较LR多检出24条，同时增加220条待复核记录；每折统一取前20%后，两者分别检出100和98条，差距缩小至2条，且RF逐折为2折领先、2折落后、1折持平。按各折LR名单长度匹配后，两者分别检出97和94条。由此，模型应在相同复核资源下比较，不能把更长名单带来的检出增加直接解释为更强的排序能力。')
p=find('原始数据来源于 UCI')
replace('原始数据来源于 UCI',p.text+' 等复核量补充方案、输入副本、逐条名单和完整结果位于“补充分析/等复核量比较”；run_comparison.py从保存分数计算名单，verify_comparison.py独立核对来源、名额、排序和计数。本轮只补充保存分数的分析，未重新拟合或回放原模型。')
before=find('参考文献')._p
insert_p(before,'表 A2 不同复核名额下的汇总危险检出','caption')
rows=[]
for setting,label in [('fixed_10pct','10%'),('fixed_20pct','20%'),('fixed_30pct','30%'),('LR_count_reference','LR名额')]:
    rr=[result(setting,m) for m in models];rows.append([label,rr[0]['review_count'],*[r['tp'] for r in rr],f"{int(rr[1]['tp'])-int(rr[0]['tp']):+d}"])
insert_table(before,['口径','每模型名额','LR','RF','RBF-SVM','XGBoost','RF−LR'],rows)
insert_p(before,'注：数值为170条危险记录中的检出数。20%为本次补充分析的主比较；10%和30%为名额敏感性比较。LR名额参照的逐折K为93、88、116、74和110，合计481。不能根据这些结果重新选择对某模型最有利的名额。')
insert_p(before,'表 A3 相同20%名额下的逐折危险检出','caption')
rows=[]
for f in range(1,6):
    rr={r['model']:r for r in folds if r['setting']=='fixed_20pct' and int(r['outer_fold'])==f}
    rows.append([f,rr['LR']['review_count'],*[rr[m]['tp'] for m in models],f"{int(rr['RF']['tp'])-int(rr['LR']['tp']):+d}"])
insert_table(before,['外层折','各模型名额','LR','RF','RBF-SVM','XGBoost','RF−LR'],rows)
insert_p(before,'注：每折包含34条危险记录。五折共享部分训练数据，表中差值作描述性比较，未将五折作为独立样本进行显著性检验。全部80个组合均保存边界并列审计；其中2个XGBoost组合拆分并列组，但其并列记录的标签一致，当前检出计数不受边界取舍影响。逐折差值未汇总为置信区间。')

def border(parent, side, val='nil', size='0'):
    borders=parent.find(qn('w:tblBorders'))
    if borders is None:
        borders=OxmlElement('w:tblBorders');parent.append(borders)
    tag=qn('w:'+side);node=borders.find(tag)
    if node is None:
        node=OxmlElement('w:'+side);borders.append(node)
    node.set(qn('w:val'),val)
    if val!='nil':
        node.set(qn('w:sz'),size);node.set(qn('w:space'),'0');node.set(qn('w:color'),'000000')

def cell_bottom(cell, size='6'):
    tcPr=cell._tc.get_or_add_tcPr()
    old=tcPr.find(qn('w:tcBorders'))
    if old is not None:tcPr.remove(old)
    borders=OxmlElement('w:tcBorders');tcPr.append(borders)
    bottom=OxmlElement('w:bottom');bottom.set(qn('w:val'),'single');bottom.set(qn('w:sz'),size);bottom.set(qn('w:space'),'0');bottom.set(qn('w:color'),'000000');borders.append(bottom)

def set_table_font(run,bold=False):
    run.font.name='Times New Roman';run.font.size=Pt(9);run.bold=bold
    rPr=run._element.get_or_add_rPr();fonts=rPr.rFonts
    if fonts is None:
        fonts=OxmlElement('w:rFonts');rPr.insert(0,fonts)
    fonts.set(qn('w:ascii'),'Times New Roman');fonts.set(qn('w:hAnsi'),'Times New Roman');fonts.set(qn('w:eastAsia'),'宋体')

# All nine manuscript tables use one three-line academic-table convention.
for table in d.tables:
    table.alignment=WD_TABLE_ALIGNMENT.CENTER
    tblPr=table._tbl.tblPr
    for side in ['left','right','insideH','insideV']:border(tblPr,side)
    border(tblPr,'top','single','8');border(tblPr,'bottom','single','8')
    for row_i,row in enumerate(table.rows):
        trPr=row._tr.get_or_add_trPr()
        cant=trPr.find(qn('w:cantSplit'))
        if cant is None:trPr.append(OxmlElement('w:cantSplit'))
        if row_i==0 and trPr.find(qn('w:tblHeader')) is None:trPr.append(OxmlElement('w:tblHeader'))
        for cell in row.cells:
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcPr=cell._tc.get_or_add_tcPr()
            shd=tcPr.find(qn('w:shd'))
            if shd is not None:tcPr.remove(shd)
            old=tcPr.find(qn('w:tcBorders'))
            if old is not None:tcPr.remove(old)
            if row_i==0:cell_bottom(cell)
            for p in cell.paragraphs:
                p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1
                # Long explanatory cells remain left aligned; labels and numeric cells are centered.
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if row_i>0 and len(p.text.strip())>14 else WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:set_table_font(run,bold=row_i==0)

for p in d.paragraphs:
    if p.style.name=='Caption' and p.text.startswith('表'):
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name='Times New Roman';run.font.size=Pt(10);run.bold=True
            rPr=run._element.get_or_add_rPr();fonts=rPr.rFonts
            if fonts is None:fonts=OxmlElement('w:rFonts');rPr.insert(0,fonts)
            fonts.set(qn('w:ascii'),'Times New Roman');fonts.set(qn('w:hAnsi'),'Times New Roman');fonts.set(qn('w:eastAsia'),'宋体')
    if p.text.startswith('注：'):
        p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(4);p.paragraph_format.line_spacing=1
        for run in p.runs:set_table_font(run)

output.parent.mkdir(parents=True,exist_ok=True);d.save(output)
with ZipFile(source) as a,ZipFile(output) as b:
    media=[n for n in a.namelist() if n.startswith('word/media/')]
    assert all(a.read(n)==b.read(n) for n in media)
    aa=etree.fromstring(a.read('word/document.xml'));bb=etree.fromstring(b.read('word/document.xml'));ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math','w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    assert [etree.tostring(x,method='c14n') for x in aa.xpath('//m:oMath',namespaces=ns)]==[etree.tostring(x,method='c14n') for x in bb.xpath('//m:oMath',namespaces=ns)]
    old_tables=aa.xpath('//w:tbl',namespaces=ns);new_tables=bb.xpath('//w:tbl',namespaces=ns)
    def table_texts(nodes):
        return [[''.join(cell.xpath('.//w:t/text()',namespaces=ns)) for cell in tbl.xpath('.//w:tr/w:tc',namespaces=ns)] for tbl in nodes]
    old_table_texts=table_texts(old_tables);new_table_texts=table_texts(new_tables)
    assert all(texts in new_table_texts for texts in old_table_texts)
outtext='\n'.join(p.text for p in d.paragraphs)
assert '需要另作预先定义的同工作量比较' not in outtext
assert len(d.tables)==9 and len(media)==10
figs=['fig01_data_profile','fig02_feature_structure','fig03_nested_workflow','fig04_outer_ap','fig05_pr_curves','fig06_paired_folds','fig07_confusion','fig08_detection_workload','fig09_fold_shift','fig10_overlap'];fig_i=0;eq_i=0;markdown=[]
latex=[r'AP=\sum_{k=1}^{K}(R_k-R_{k-1})P_k',r'F_2=\frac{5TP}{5TP+4FN+FP}',r'\hat y_i=\begin{cases}1&s_i\geq\tau\\0&s_i<\tau\end{cases}']
for el in d._element.body:
    if el.tag==qn('w:p'):
        p=Paragraph(el,d._body)
        if el.xpath('.//w:drawing'):
            markdown.append(f'![图{fig_i+1}](../图件/PNG/{figs[fig_i]}.png)');fig_i+=1
        elif el.xpath('.//m:oMath'):
            markdown.append('$$\n'+latex[eq_i]+f'\\quad({eq_i+1})\n$$');eq_i+=1
        elif p.text:
            txt=p.text
            if p.style.name.startswith('Heading'):txt='#'*(2 if p.style.name=='Heading 1' else 3)+' '+txt
            markdown.append(txt)
    elif el.tag==qn('w:tbl'):
        t=Table(el,d._body);rr=['| '+' | '.join(c.text.replace('\n',' ') for c in row.cells)+' |' for row in t.rows];rr.insert(1,'|'+'|'.join(' --- ' for _ in t.rows[0].cells)+'|');markdown.append('\n'.join(rr))
assert fig_i==10 and eq_i==3
output.with_suffix('.md').write_text('\n\n'.join(markdown),encoding='utf-8')
log={'scope':'Bounded manuscript integration from retained source and saved-score comparisons; no training or model replay','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'original_media_preserved':len(media),'original_equations_preserved':3,'original_table_texts_preserved':len(old_tables),'tables_reformatted_as_three_line':len(d.tables),'added_tables':3,'total_tables':len(d.tables),'plan_sha256':hashlib.sha256((module/'analysis_plan.json').read_bytes()).hexdigest(),'comparison_input_hashes':plan['inputs'],'changes':changes}
(output.parent/'等复核量整合_修改记录.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in log.items() if k!='changes'},ensure_ascii=False,indent=2))
