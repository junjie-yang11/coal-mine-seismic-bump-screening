"""Execute a locked, grouped nested 5x5 comparison in a fresh output directory."""
import json
import platform
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_predict
from common import read_data, split_folds, pipeline, score_method, positive_scores, select_cutoff, metrics, digest

ROOT=Path(__file__).resolve().parent
def save_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(path, frame):
    frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')

def develop(name, history_x, history_y, history_groups, categories, numbers, plan):
    """This development API receives only the outer historical/training partition."""
    folds=split_folds(history_x,history_y,history_groups,plan['inner'])
    base=pipeline(name,categories,numbers,plan)
    method=score_method(name)
    candidates=[]
    if name=='Dummy':
        final=base.fit(history_x,history_y)
        oof=positive_scores(cross_val_predict(base,history_x,history_y,cv=folds,method=method,n_jobs=1))
        choice={'params':{},'candidate':None,'threshold':None,'score_method':method}
    else:
        grid={'model__'+key:values for key,values in plan['grids'][name].items()}
        search=GridSearchCV(base,grid,cv=folds,scoring='average_precision',n_jobs=1,error_score='raise',refit=True)
        search.fit(history_x,history_y)
        final=search.best_estimator_
        oof=positive_scores(cross_val_predict(clone(final),history_x,history_y,cv=folds,method=method,n_jobs=1))
        threshold,f2=select_cutoff(history_y,oof)
        choice={'params':search.best_params_,'candidate':int(search.best_index_), 'inner_mean_ap':float(search.best_score_),
                'threshold':threshold,'inner_oof_f2':f2,'score_method':method}
        cv=search.cv_results_
        for i,params in enumerate(cv['params']):
            candidates.append({'model':name,'candidate':i,'parameters':json.dumps(params,sort_keys=True),'selected':i==search.best_index_,
                'mean_ap':float(cv['mean_test_score'][i]),'std_ap':float(cv['std_test_score'][i]),
                **{f'inner_{j+1}_ap':float(cv[f'split{j}_test_score'][i]) for j in range(5)}})
    return final,choice,oof,candidates,folds

def evaluate(estimator, choice, x):
    """Scoring and decisions deliberately do not accept labels."""
    scores=positive_scores(getattr(estimator,choice['score_method'])(x))
    decision=np.zeros(len(x),dtype=int) if choice['threshold'] is None else (scores>=choice['threshold']).astype(int)
    return scores,decision

def run():
    started=time.monotonic()
    plan_path=ROOT/'locked_plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    for folder in ['results','models','checks']:
        (ROOT/folder).mkdir(exist_ok=True)
    if any((ROOT/'results').iterdir()) or any((ROOT/'models').iterdir()):
        raise FileExistsError('Use a fresh experiment directory; existing evidence must remain intact.')
    status=ROOT/'checks/run.json'
    save_json(status,{'status':'running','passed':False,'plan_sha256':digest(plan_path)})
    try:
        x,y,groups,categories,numbers=read_data(plan)
        outer=split_folds(x,y,groups,plan['outer'])
        assignment=np.zeros(len(x),dtype=int)
        for f,(_,test) in enumerate(outer,1): assignment[test]=f
        csv(ROOT/'results/outer_assignments.csv',pd.DataFrame({'source_row':np.arange(1,len(x)+1),'input_group':groups,'label':y,'outer_fold':assignment}))
        predictions=[]; candidate_rows=[]; oof_rows=[]; inner_rows=[]; metric_rows=[]; choices=[]
        for f,(train,test) in enumerate(outer,1):
            train,test=np.sort(train),np.sort(test)
            hx,hy,hg=x.iloc[train],y[train],groups[train]
            print(f'OUTER {f}/5 train={len(train)} test={len(test)} positive={y[test].sum()}',flush=True)
            fold_choices=[]
            for name in ['Dummy',*plan['model_pool']]:
                tick=time.monotonic()
                fitted,choice,oof,candidates,inner=develop(name,hx,hy,hg,categories,numbers,plan)
                choice.update({'model':name,'outer_fold':f})
                fold_choices.append(choice); choices.append(choice)
                inner_id=np.zeros(len(train),dtype=int)
                for j,(_,v) in enumerate(inner,1): inner_id[v]=j
                if name=='Dummy':
                    inner_rows.extend({'outer_fold':f,'source_row':int(t)+1,'input_group':int(groups[t]),'inner_fold':int(j)} for t,j in zip(train,inner_id))
                oof_rows.extend({'outer_fold':f,'model':name,'source_row':int(t)+1,'inner_fold':int(j),'label':int(label),'score':float(s)} for t,j,label,s in zip(train,inner_id,hy,oof))
                candidate_rows.extend({'outer_fold':f,**r} for r in candidates)
                joblib.dump(fitted,ROOT/f'models/fold{f}_{name}.joblib',compress=3)
                scores,decision=evaluate(fitted,choice,x.iloc[test])
                assert np.isfinite(scores).all()
                metric_rows.append({'outer_fold':f,'model':name,**metrics(y[test],scores,decision)})
                predictions.append(pd.DataFrame({'outer_fold':f,'model':name,'source_row':test+1,'input_group':groups[test],
                                                'shift':x.iloc[test]['shift'].to_numpy(),'label':y[test],'score':scores,'prediction':decision}))
                print(f'  {name} AP={metric_rows[-1]["ap"]:.6f} TP={metric_rows[-1]["tp"]} FP={metric_rows[-1]["fp"]} elapsed={time.monotonic()-tick:.1f}s',flush=True)
            save_json(ROOT/f'results/fold{f}_choices.json',fold_choices)
        pred=pd.concat(predictions,ignore_index=True)
        metric=pd.DataFrame(metric_rows)
        metric['ap_rank']=metric.groupby('outer_fold').ap.rank(ascending=False,method='min')
        for filename,frame in [('outer_predictions',pred),('outer_metrics',metric),('inner_candidates',pd.DataFrame(candidate_rows)),
                               ('inner_oof',pd.DataFrame(oof_rows)),('inner_assignments',pd.DataFrame(inner_rows))]:
            csv(ROOT/f'results/{filename}.csv',frame)
        save_json(ROOT/'results/frozen_choices.json',{'plan_sha256':digest(plan_path),'choices':choices})
        summary=[]
        for name,r in metric.groupby('model',sort=False):
            tp,fp,fn,tn=(int(r[k].sum()) for k in ['tp','fp','fn','tn'])
            pp=pred[pred.model.eq(name)].sort_values('source_row')
            summary.append({'model':name,'mean_ap':r.ap.mean(),'sd_ap':r.ap.std(ddof=0),'mean_roc_auc':r.roc_auc.mean(),
                'mean_rank':r.ap_rank.mean(),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'alerts':tp+fp,
                'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn),
                'f2':5*tp/(5*tp+4*fn+fp),'pooled_ap_supplement':metrics(pp.label,pp.score,pp.prediction)['ap']})
        csv(ROOT/'results/model_summary.csv',pd.DataFrame(summary))
        wide=metric.pivot(index='outer_fold',columns='model',values='ap')
        pairs=[]
        for i,a in enumerate(plan['model_pool']):
            for b in plan['model_pool'][i+1:]:
                for f in wide.index: pairs.append({'outer_fold':f,'model_a':a,'model_b':b,'ap_a_minus_b':wide.loc[f,a]-wide.loc[f,b]})
        csv(ROOT/'results/paired_fold_differences.csv',pd.DataFrame(pairs))
        decisions=pred[pred.model.ne('Dummy')].pivot(index=['source_row','outer_fold','input_group','shift','label'],columns='model',values='prediction').reset_index()
        decisions['common_miss']=decisions[plan['model_pool']].eq(0).all(axis=1)&decisions.label.eq(1)
        csv(ROOT/'results/positive_case_audit.csv',decisions[decisions.label.eq(1)])
        by_shift=[]
        for (name,shift),p in pred.groupby(['model','shift']):
            by_shift.append({'model':name,'shift':shift,**metrics(p.label,p.score,p.prediction)})
        csv(ROOT/'results/by_shift.csv',pd.DataFrame(by_shift))
        save_json(ROOT/'results/data_audit.json',{'n':len(x),'features':len(x.columns),'positives':int(y.sum()),
            'groups':int(len(np.unique(groups))),'repeated_after_first':int(x.duplicated().sum()),'categorical_features':categories,'numeric_features':numbers})
        save_json(status,{'status':'completed','passed':True,'scope':'actual nested training and saved outer predictions; separate replay verification required',
            'elapsed_seconds':time.monotonic()-started,'plan_sha256':digest(plan_path),'training_code_sha256':digest(ROOT/'run_nested.py'),
            'common_code_sha256':digest(ROOT/'common.py'),'data_sha256':digest(ROOT/'data/seismic-bumps.arff'),
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
                           'scikit_learn':sklearn.__version__,'xgboost':xgboost.__version__,'joblib':joblib.__version__},
            'outer_folds':5,'inner_folds':5,'saved_models':25,'prediction_rows':len(pred),'candidate_rows':len(candidate_rows)})
        print(pd.DataFrame(summary).to_string(index=False),flush=True)
    except Exception:
        save_json(status,{'status':'failed','passed':False,'error':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started})
        raise

if __name__=='__main__': run()
