"""Independent saved-model replay, split audit, AP and threshold recalculation."""
import json
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from common import read_data, split_folds, digest

ROOT=Path(__file__).resolve().parent
def read(name): return pd.read_csv(ROOT/f'results/{name}.csv',float_precision='round_trip')
def independent_ap(y,s):
    order=np.argsort(-s,kind='stable'); yy=y[order]; ss=s[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tp=np.cumsum(yy)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/y.sum())
def cutoff(y,s):
    order=np.argsort(-s,kind='stable'); ss=s[order]; yy=y[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tp=np.cumsum(yy)[ends]; fp=ends+1-tp; fn=y.sum()-tp
    f=5*tp/(5*tp+4*fn+fp)
    # Descending thresholds: the first exact maximum is the higher cutoff.
    i=np.flatnonzero(f==f.max())[0]
    return float(ss[ends[i]])
def verify():
    path=ROOT/'checks/verification.json'; started=time.monotonic()
    def record(data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    record({'status':'running','passed':False})
    try:
        plan=json.loads((ROOT/'locked_plan.json').read_text(encoding='utf-8'))
        run=json.loads((ROOT/'checks/run.json').read_text())
        assert run['status']=='completed' and run['passed']
        assert run['plan_sha256']==digest(ROOT/'locked_plan.json')
        assert run['training_code_sha256']==digest(ROOT/'run_nested.py')
        assert run['common_code_sha256']==digest(ROOT/'common.py')
        x,y,groups,categories,numbers=read_data(plan)
        outer=split_folds(x,y,groups,plan['outer'])
        assignments=read('outer_assignments'); inner_all=read('inner_assignments')
        preds=read('outer_predictions'); met=read('outer_metrics'); candidates=read('inner_candidates'); oof=read('inner_oof')
        frozen=json.loads((ROOT/'results/frozen_choices.json').read_text(encoding='utf-8'))
        assert frozen['plan_sha256']==digest(ROOT/'locked_plan.json')
        assert len(preds)==12920 and len(candidates)==180
        assert assignments.source_row.tolist()==list(range(1,2585))
        assert assignments.groupby('input_group').outer_fold.nunique().eq(1).all()
        np.testing.assert_array_equal(assignments.input_group,groups)
        for f,(train,test) in enumerate(outer,1):
            train,test=np.sort(train),np.sort(test)
            assert not set(groups[train])&set(groups[test])
            np.testing.assert_array_equal(assignments[assignments.outer_fold.eq(f)].source_row,test+1)
            inner=split_folds(x.iloc[train],y[train],groups[train],plan['inner'])
            imap=inner_all[inner_all.outer_fold.eq(f)].sort_values('source_row')
            np.testing.assert_array_equal(imap.source_row,train+1)
            for j,(fit,val) in enumerate(inner,1):
                assert not set(groups[train[fit]])&set(groups[train[val]])
                np.testing.assert_array_equal(np.sort(imap[imap.inner_fold.eq(j)].source_row),np.sort(train[val]+1))
            for choice in [c for c in frozen['choices'] if c['outer_fold']==f]:
                name=choice['model']
                p=preds[(preds.outer_fold.eq(f))&preds.model.eq(name)].sort_values('source_row')
                oo=oof[(oof.outer_fold.eq(f))&oof.model.eq(name)].sort_values('source_row')
                np.testing.assert_array_equal(p.source_row,test+1); np.testing.assert_array_equal(p.label,y[test])
                np.testing.assert_array_equal(oo.source_row,train+1); np.testing.assert_array_equal(oo.label,y[train])
                np.testing.assert_array_equal(oo.inner_fold,imap.inner_fold)
                estimator=joblib.load(ROOT/f'models/fold{f}_{name}.joblib')
                raw=getattr(estimator,choice['score_method'])(x.iloc[test]); score=raw[:,1] if raw.ndim==2 else raw
                np.testing.assert_array_equal(score,p.score)
                decision=np.zeros(len(test),int) if name=='Dummy' else (score>=choice['threshold']).astype(int)
                np.testing.assert_array_equal(decision,p.prediction)
                # Labels can be changed only for evaluation, leaving frozen scores and decisions unchanged.
                flipped=1-p.label.to_numpy(); before_score=score.copy(); before_decision=decision.copy()
                _=independent_ap(flipped,score)
                raw2=getattr(estimator,choice['score_method'])(x.iloc[test]); after=raw2[:,1] if raw2.ndim==2 else raw2
                np.testing.assert_array_equal(after,before_score)
                if name!='Dummy': np.testing.assert_array_equal((after>=choice['threshold']).astype(int),before_decision)
                m=met[(met.outer_fold.eq(f))&met.model.eq(name)].iloc[0]
                assert abs(independent_ap(p.label.to_numpy(),p.score.to_numpy())-m.ap)<1e-14
                for key,lab,d in [('tp',1,1),('fp',0,1),('fn',1,0),('tn',0,0)]:
                    assert int(((p.label==lab)&(p.prediction==d)).sum())==m[key]
                if name!='Dummy':
                    cs=candidates[(candidates.outer_fold.eq(f))&candidates.model.eq(name)].sort_values('candidate')
                    grid=list(ParameterGrid({'model__'+k:v for k,v in plan['grids'][name].items()}))
                    assert [json.loads(s) for s in cs.parameters]==grid
                    assert cs['selected'].sum()==1 and int(cs.iloc[np.argmax(cs.mean_ap)].candidate)==choice['candidate']
                    assert choice['params']==grid[choice['candidate']]
                    for _,c in cs.iterrows():
                        values=np.array([c[f'inner_{j}_ap'] for j in range(1,6)])
                        assert abs(values.mean()-c.mean_ap)<1e-14
                    assert cutoff(oo.label.to_numpy(),oo.score.to_numpy())==choice['threshold']
                    for k,v in choice['params'].items(): assert estimator.get_params()[k]==v
            print(f'REPLAY outer fold {f}: five models, thresholds, splits and independent metrics passed.',flush=True)
        summary=read('model_summary').set_index('model')
        for name in summary.index:
            m=met[met.model.eq(name)]
            assert abs(m.ap.mean()-summary.loc[name,'mean_ap'])<1e-14
            assert abs(m.ap.std(ddof=0)-summary.loc[name,'sd_ap'])<1e-14
            for k in ['tp','fp','fn','tn']: assert m[k].sum()==summary.loc[name,k]
            p=preds[preds.model.eq(name)]
            assert len(p)==2584 and p.source_row.is_unique and p.label.sum()==170
        positives=read('positive_case_audit')
        assert len(positives)==170
        np.testing.assert_array_equal(positives.common_miss,positives[plan['model_pool']].eq(0).all(axis=1))
        record({'status':'completed','passed':True,'scope':'25 saved-model replays, independent AP/confusion counts and F2 cutoff, candidate/choice audit, split isolation, frozen-score label perturbation, and summary reconciliation; no rerun of all candidate fits',
                'elapsed_seconds':time.monotonic()-started,'models_replayed':25,'scores_replayed':len(preds),'plan_sha256':digest(ROOT/'locked_plan.json'),
                'verifier_sha256':digest(Path(__file__))})
    except Exception:
        record({'status':'failed','passed':False,'error':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started})
        raise
if __name__=='__main__': verify()
