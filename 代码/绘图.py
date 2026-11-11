"""Ten figures restyled from the author's plot_nested.py. Real saved observations only; no fitting or strategy selection. See 图件/QA/figure-contracts.json for the retained origin record."""
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from fontTools.ttLib import TTFont
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,default=ROOT/'实验')
parser.add_argument('--output',type=Path,default=ROOT,help='输出根目录；下面生成图件子目录')
args=parser.parse_args(); out=args.output.resolve()
for d in ['图件/PNG','图件/PDF','图件/SVG','图件/QA','图件/源数据']: (out/d).mkdir(parents=True,exist_ok=True)
import tempfile
font_cache=tempfile.TemporaryDirectory(prefix='seismic-figure-font-',ignore_cleanup_errors=True)
font=Path(font_cache.name)/'msyh.ttf'
with TTFont('C:/Windows/Fonts/msyh.ttc',fontNumber=0) as f: f.save(font)
font_manager.fontManager.addfont(font)
plt.style.use('default')
plt.rcParams.update({'font.family':['Arial',font_manager.FontProperties(fname=font).get_name()], 'font.size':9,
 'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.65,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'grid.alpha':.20,'grid.linewidth':.6,'axes.axisbelow':True,'axes.facecolor':'white','figure.facecolor':'white','savefig.facecolor':'white','xtick.major.size':3,'ytick.major.size':3,'axes.edgecolor':'#535D66','axes.labelcolor':'#222A30','text.color':'#222A30','xtick.color':'#535D66','ytick.color':'#535D66'})
sys.path.insert(0,str(Path(__file__).resolve().parent/'图件检查工具'))
from audit_panel_alignment import require_matplotlib_panel_alignment
colors={'LR':'#D55E00','RF':'#0072B2','RBF-SVM':'#009E73','XGBoost':'#8E6C8A','Dummy':'#8B949C'}
model_markers={'LR':'o','RF':'s','RBF-SVM':'^','XGBoost':'D'}
models=['LR','RF','RBF-SVM','XGBoost']
def read(name): return pd.read_csv(args.project/f'results/{name}.csv',float_precision='round_trip')
m=read('outer_metrics'); s=read('model_summary').set_index('model'); pairs=read('paired_fold_differences'); cases=read('positive_case_audit')
def title(ax,letter,text):
 ax.text(0,1.04,letter,transform=ax.transAxes,fontsize=9,weight='bold',va='bottom')
 ax.annotate(text,(0,1.04),xycoords='axes fraction',xytext=(12,0),textcoords='offset points',fontsize=9,va='bottom')
def export(fig,stem,axes,orientation='row',exemptions=None):
 opts={} if len(axes) in (1,4) else {orientation+'_groups':[list('ab')[:len(axes)]]}
 require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('abcdefghijklmnopqrstuvwxyz')[:len(axes)],strict=True,require_panel_labels=True,
  exemptions=exemptions or [],json_out=out/f'图件/QA/{stem}.alignment.json',**opts)
 fig.savefig(out/f'图件/PDF/{stem}.pdf'); fig.savefig(out/f'图件/SVG/{stem}.svg'); fig.savefig(out/f'图件/PNG/{stem}.png',dpi=600)
 plt.close(fig)
def two(height=3.0):
 fig,(a,b)=plt.subplots(1,2,figsize=(6.65,height+.25)); fig.subplots_adjust(left=.12,right=.97,bottom=.27,top=.87,wspace=.47)
 return fig,a,b

# 1. One common development workflow, repeated for every outer test fold.
fig=plt.figure(figsize=(6.65,3.45)); a=fig.add_axes([.03,.03,.94,.85]); a.set(xlim=(0,100),ylim=(0,100)); a.axis('off')
title(a,'a','统一四模型嵌套五折流程')
def box(x,y,w,h,text,fill='#F3F5F7'):
 a.add_patch(Rectangle((x,y),w,h,ec='#7F898F',fc=fill,lw=.6)); a.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7,linespacing=1.35)
def arrow(x1,y1,x2,y2): a.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','lw':.7,'color':'#59646B'})
box(2,86,96,11,'2,584 条班次记录 · 18 个输入 · 170 条危险记录\n相同输入同组；外层随机分层分组五折')
box(2,66,57,11,'每轮四折作为外层训练集\n内层分层分组五折，四模型分别按平均 AP 选参','#E7EFF4')
box(69,66,29,11,'另一折作为测试集\n外层测试始终隔离')
arrow(30,86,30,77); arrow(83.5,86,83.5,77)
box(2,45,57,12,'所选配置生成训练折外分数，按 F2 选阈值\n在全部外层训练集重拟合模型','#E7EFF4')
arrow(30,66,30,57)
box(69,45,29,12,'冻结模型和阈值\n评分外层测试记录')
arrow(59,51,69,51); arrow(83.5,66,83.5,57)
box(2,24,96,12,'轮换五个外层折：每条记录获得每种模型的一次外层测试预测\n留存每折参数、阈值、分数及分类判定；另设恒定分数基线')
arrow(83.5,45,83.5,36)
box(2,3,96,12,'主要：逐折 AP 与五折平均 AP    |    分类：检出、误报与复核量\n辅助：共同漏报与输入班次分组；混合折分数的 AP 仅作补充','#E7EFF4')
arrow(50,24,50,15)
export(fig,'fig03_nested_workflow',[a])

# 2. Fold-specific results and descriptive mean ± one fold SD, not a CI.
fig,a,b=two(3.20); title(a,'a','每折外层测试 AP'); title(b,'b','五折均值与折间标准差')
for model in models:
 r=m[m.model.eq(model)].sort_values('outer_fold'); a.plot(r.outer_fold,r.ap,marker=model_markers[model],ms=3.5,color=colors[model],label=model,lw=1)
a.plot([1,5],[s.loc['Dummy','mean_ap']]*2,color=colors['Dummy'],ls=':',lw=.8,label='恒定基线')
a.set(xticks=range(1,6),xlabel='外层测试折',ylabel='AP',ylim=(0,.45))
fig.legend(*a.get_legend_handles_labels(),ncol=3,loc='lower left',bbox_to_anchor=(.1,.005),columnspacing=.7)
for i,model in enumerate([*models,'Dummy']):
 b.errorbar(s.loc[model,'mean_ap'],i,xerr=s.loc[model,'sd_ap'],fmt='o',color=colors[model],capsize=2,ms=4,lw=.9)
b.set(yticks=range(5),yticklabels=[*models,'恒定基线'],ylim=(4.6,-.6),xlim=(0,.37),xlabel='平均 AP ± 1 折间标准差')
b.tick_params(axis='y',length=0)
export(fig,'fig04_outer_ap',[a,b])

# 3. All six model pairs and all five outer folds; no independence-based tests.
fig=plt.figure(figsize=(6.65,3.05)); a=fig.add_axes([.27,.22,.68,.63]); title(a,'a','同一外层测试折上的配对 AP 差')
comps=list(pairs[['model_a','model_b']].drop_duplicates().itertuples(index=False,name=None))
markers=['o','s','^','D','v']; foldcolors=['#0072B2','#009E73','#D55E00','#8E6C8A','#7F898F']
for i,(ma,mb) in enumerate(comps):
 r=pairs[pairs.model_a.eq(ma)&pairs.model_b.eq(mb)].sort_values('outer_fold')
 for j,row in enumerate(r.itertuples()): a.plot(row.ap_a_minus_b,i+(j-2)*.07,marker=markers[j],color=foldcolors[j],ls='none',ms=4,label=f'折{j+1}' if i==0 else None)
a.axvline(0,color='#7F898F',ls=':',lw=.8)
a.set(yticks=range(6),yticklabels=[f'{ma} − {mb}' for ma,mb in comps],ylim=(5.6,-.6),xlabel='AP 差（左侧模型减右侧模型）')
a.tick_params(axis='y',length=0); fig.legend(*a.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.60,.01),ncol=5,columnspacing=1)
export(fig,'fig06_paired_folds',[a])

# 4. Aggregated confusion counts are additive despite fold-specific cutoffs.
fig,a,b=two(3.15); title(a,'a','全部外层判定的检出与复核量'); title(b,'b','170 条危险记录的检出与漏报')
for model,offset in zip(models,[(6,4),(6,4),(6,7),(6,-12)]):
 r=s.loc[model]; a.plot(r.alerts,r.tp,'o',color=colors[model],ms=5)
 a.annotate(model,(r.alerts,r.tp),xytext=offset,textcoords='offset points',fontsize=7,color=colors[model])
a.set(xlim=(430,830),ylim=(80,135),xlabel='待复核记录数（TP + FP）',ylabel='检出危险记录数（TP）')
for i,model in enumerate(models):
 r=s.loc[model]; b.barh(i,r.tp,color=colors[model],height=.55); b.barh(i,r.fn,left=r.tp,color='#E5EAEE',height=.55)
 b.text(r.tp/2,i,str(int(r.tp)),ha='center',va='center',color='white',fontsize=7)
 b.text(r.tp+r.fn/2,i,str(int(r.fn)),ha='center',va='center',fontsize=7)
b.set(yticks=range(4),yticklabels=models,ylim=(3.5,-.5),xlim=(0,170),xlabel='危险记录数：检出 / 漏报')
b.tick_params(axis='y',length=0)
a.grid(False); b.grid(False)
export(fig,'fig08_detection_workload',[a,b])

# 5. Workload variation by fold and detection rates by input shift.
fig,a,b=two(3.0); title(a,'a','各折判为危险的比例'); title(b,'b','按输入班次类型的检出')
for model in models:
 r=m[m.model.eq(model)].sort_values('outer_fold'); a.plot(r.outer_fold,(r.tp+r.fp)/r.n,marker=model_markers[model],linestyle='-',color=colors[model],ms=3.5,label=model,lw=1)
a.set(xticks=range(1,6),xlabel='外层测试折',ylabel='判为危险比例',ylim=(0,.42)); a.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
a.legend(ncol=2,loc='lower left',bbox_to_anchor=(.1,.005),bbox_transform=fig.transFigure,columnspacing=.8)
by=read('by_shift').set_index(['model','shift'])
for i,model in enumerate(models):
 for shift,offset,mark,col in [('W',-.13,'o','#0072B2'),('N',.13,'s','#D55E00')]:
  r=by.loc[(model,shift)]; b.plot(r.recall,i+offset,mark,color=col,mfc=col if shift=='W' else 'white',ms=4,label=('采煤班' if shift=='W' else '准备班') if i==0 else None)
  b.annotate(f'{int(r.tp)}/{int(r.positives)}',(r.recall,i+offset),xytext=(5,0),textcoords='offset points',va='center',fontsize=7)
b.set(yticks=range(4),yticklabels=models,ylim=(3.5,-.5),xlim=(-.03,1.07),xlabel='危险记录检出比例'); b.tick_params(axis='y',length=0)
b.xaxis.set_major_formatter(PercentFormatter(1,decimals=0)); b.legend(loc='lower left',bbox_to_anchor=(.67,.005),bbox_transform=fig.transFigure,ncol=2)
b.grid(False)
export(fig,'fig09_fold_shift',[a,b])

# 6. Every positive record contributes to exactly one observed decision pattern.
codes=cases[models].astype(int).astype(str).agg(''.join,axis=1)
counts=codes.value_counts().sort_values(ascending=False)
fig=plt.figure(figsize=(6.65,3.25)); a=fig.add_axes([.14,.48,.81,.37]); b=fig.add_axes([.14,.16,.81,.18])
title(a,'a','危险记录的模型检出组合'); title(b,'b','各组合包含的检出模型')
for i,(code,n) in enumerate(counts.items()):
 col='#D55E00' if code=='0000' else '#0072B2'; a.bar(i,n,color=col,width=.64); a.text(i,n+2.5,str(n),ha='center',va='bottom',fontsize=7)
 active_rows=[j for j,bit in enumerate(code) if bit=='1']
 if len(active_rows)>1: b.plot([i,i],[min(active_rows),max(active_rows)],color='#0072B2',lw=.8,zorder=1)
 for j,bit in enumerate(code): b.plot(i,j,'o',color='#0072B2' if bit=='1' else '#D7DFE4',ms=4)
a.set(xlim=(-.6,len(counts)-.4),ylim=(0,counts.max()+9),xticks=[],ylabel='危险记录数')
b.set(xlim=a.get_xlim(),ylim=(3.5,-.5),xticks=[],yticks=range(4),yticklabels=models)
b.tick_params(axis='y',length=0); b.spines[['bottom','left']].set_visible(False)
a.grid(False); b.grid(False)
export(fig,'fig10_overlap',[a,b],orientation='column')
pd.DataFrame({'pattern':counts.index,'count':counts.values}).to_csv(out/'图件/源数据/decision_patterns.csv',index=False,encoding='utf-8-sig')


# Additional descriptive displays use retained observations and outer predictions.
import hashlib
import json
def precision_recall_curve(label, score):
 y=np.asarray(label,dtype=int); score=np.asarray(score,dtype=float)
 order=np.argsort(-score,kind='stable'); y=y[order]; ordered=score[order]
 ends=np.r_[np.flatnonzero(np.diff(ordered)),len(y)-1]
 tp=y.cumsum()[ends]; precision=tp/(ends+1); recall=tp/y.sum()
 return np.r_[precision[::-1],1.],np.r_[recall[::-1],0.],ordered[ends][::-1]
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
(out/'图件/源数据').mkdir(exist_ok=True)
plan=json.loads((args.project/'locked_plan.json').read_text(encoding='utf-8'))
source=args.project/'data/seismic-bumps.arff'
assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['arff_sha256']
raw=source.read_text(encoding='utf-8').splitlines()
fields=[line.split()[1] for line in raw if line.lower().startswith('@attribute ')]
start=next(i for i,line in enumerate(raw) if line.strip().lower()=='@data')+1
data=pd.DataFrame([line.split(',') for line in raw[start:] if line.strip() and not line.startswith('%')],columns=fields)
data.index+=1
labels=data['class'].astype(int)
assert len(data)==2584 and labels.sum()==170
audit=json.loads((args.project/'results/data_audit.json').read_text(encoding='utf-8'))
numeric=data[audit['numeric_features']].astype(float)
constants=numeric.columns[numeric.nunique().eq(1)].tolist()
active=numeric.loc[:,numeric.nunique().gt(1)]
assert np.isfinite(numeric.to_numpy()).all()
energy=numeric.energy
assert energy.ge(0).all()
metadata={'records':len(data),'positives':int(labels.sum()),'negatives':int(labels.eq(0).sum()),
          'constant_numeric_fields_omitted_from_correlation':constants,'correlation_fields':active.columns.tolist(),
          'scope':'Descriptive all-record summaries after the frozen evaluation; not feature selection or model fitting.'}
(out/'图件/源数据/数据描述.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')

# 1. Minority-label composition and the full observed energy distribution.
fig,a,b=two(3.05); title(a,'a','下一班次标签分布'); title(b,'b','输入班次能量分布')
counts=[int(labels.eq(0).sum()),int(labels.eq(1).sum())]
a.barh([0,1],counts,color=['#AAB4BE','#0072B2'],height=.52,edgecolor='white')
for i,count in enumerate(counts): a.annotate(f'{count:,}（{count/len(data):.2%}）',(count,i),xytext=(6,0),textcoords='offset points',va='center',fontsize=8)
a.set(yticks=[0,1],yticklabels=['非危险','危险'],ylim=(1.65,-.65),xlim=(0,3600),xlabel='记录数')
a.xaxis.set_major_locator(MaxNLocator(4,integer=True)); a.grid(False)
for lab,color,label in [(0,'#7F8993','非危险记录'),(1,'#d95f02','危险记录')]:
 values=np.sort(np.log10(1+energy[labels.eq(lab)].to_numpy()))
 b.step(values,np.arange(1,len(values)+1)/len(values),where='post',lw=1.6,color=color,label=label)
b.set(xlabel=r'$\log_{10}(1+E/\mathrm{J})$',ylabel='累积记录比例',ylim=(0,1.03))
b.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
b.legend(loc='lower right',fontsize=8)
export(fig,'fig01_data_profile',[a,b])

# 2. Rank correlation describes overlap, not causal relevance or importance.
corr=active.rank(method='average').corr(method='pearson')
corr.to_csv(out/'图件/源数据/输入变量_Spearman.csv',encoding='utf-8-sig')
n=len(corr)
fig=plt.figure(figsize=(6.65,5.1)); a=fig.add_axes([.16,.21,.70,.68]); title(a,'a','数值输入的 Spearman 相关性')
matrix=np.ma.masked_where(np.triu(np.ones((n,n),bool),1),corr.to_numpy())
image=a.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),aspect='equal')
for i in range(n):
 for j in range(i+1):
  v=corr.iloc[i,j]
  text=f'{v:.3f}' if i!=j and abs(v)>.995 else f'{v:.2f}'
  a.text(j,i,text,ha='center',va='center',fontsize=6.7,color='white' if abs(v)>.65 else '#263238')
a.set(xticks=range(n),yticks=range(n),xticklabels=corr.columns,yticklabels=corr.index)
a.tick_params(axis='x',labelrotation=90,labelsize=8); a.tick_params(axis='y',labelsize=8)
a.grid(False); a.spines[['top','right','bottom','left']].set_visible(False)
cax=fig.add_axes([.90,.25,.018,.55]); cb=fig.colorbar(image,cax=cax); cb.set_label('秩相关系数',fontsize=8)
export(fig,'fig02_feature_structure',[a])

# 5. Curves stay within fold-specific score scales; scores are never pooled.
pred=read('outer_predictions')
fold_colors=['#0072B2','#D55E00','#009E73','#C84B4B','#8E6C8A']
fig,axes=plt.subplots(2,2,figsize=(6.65,5.65))
fig.subplots_adjust(left=.12,right=.97,bottom=.14,top=.92,wspace=.34,hspace=.43)
curve_rows=[]
for letter,ax,model in zip('abcd',axes.flat,models):
 title(ax,letter,model)
 for fold in range(1,6):
  r=pred[pred.model.eq(model)&pred.outer_fold.eq(fold)]
  precision,recall,threshold=precision_recall_curve(r.label,r.score)
  ax.step(recall,precision,where='post',color=fold_colors[fold-1],lw=1.05,alpha=.88,label=f'折 {fold}')
  curve_rows.extend({'model':model,'outer_fold':fold,'recall':float(rr),'precision':float(pp)} for rr,pp in zip(recall,precision))
 ax.axhline(labels.mean(),color='#7F8993',ls='--',lw=.8,label='总体危险比例')
 ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='召回率',ylabel='精确率')
 ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
 ax.tick_params(axis='x',pad=8)
handles,legend=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,legend,loc='lower center',bbox_to_anchor=(.54,.005),ncol=3,fontsize=8)
pd.DataFrame(curve_rows).to_csv(out/'图件/源数据/逐折PR曲线.csv',index=False,encoding='utf-8-sig')
export(fig,'fig05_pr_curves',list(axes.flat))

# 7. Raw counts with within-true-class normalization and a common colour scale.
fig,axes=plt.subplots(2,2,figsize=(6.65,5.25))
fig.subplots_adjust(left=.15,right=.84,bottom=.14,top=.92,wspace=.60,hspace=.55)
confusion_rows=[]
for letter,ax,model in zip('abcd',axes.flat,models):
 title(ax,letter,model)
 r=s.loc[model]
 counts=np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int)
 fractions=counts/counts.sum(axis=1,keepdims=True)
 im=ax.imshow(fractions,cmap='Blues',vmin=0,vmax=1,aspect='equal')
 for i in range(2):
  for j in range(2):
   ax.text(j,i,f'{counts[i,j]:,}\n{fractions[i,j]:.1%}',ha='center',va='center',fontsize=8.2,linespacing=1.5,
           color='white' if fractions[i,j]>.55 else '#263238')
   confusion_rows.append({'model':model,'actual_label':i,'predicted_label':j,'count':counts[i,j],'fraction_within_true_class':fractions[i,j]})
 ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['判非危险','判危险'],yticklabels=['实际非危险','实际危险'])
 ax.tick_params(labelsize=8,length=0); ax.grid(False)
 for spine in ax.spines.values(): spine.set_visible(False)
cax=fig.add_axes([.88,.23,.02,.53]); cb=fig.colorbar(im,cax=cax,format=PercentFormatter(1,decimals=0)); cb.set_label('实际类别内的判定比例',fontsize=8)
pd.DataFrame(confusion_rows).to_csv(out/'图件/源数据/混淆矩阵计数及比例.csv',index=False,encoding='utf-8-sig')
export(fig,'fig07_confusion',list(axes.flat))
print('Ten restyled Matplotlib figures generated; original observations, folds, metrics and cutoffs retained.')
