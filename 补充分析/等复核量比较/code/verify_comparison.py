"""Independent saved-list verification, without importing the analysis runner."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent.parent
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    start=time.perf_counter()
    plan=json.loads((ROOT/'analysis_plan.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/'inputs'/n).read_bytes()).hexdigest()==h for n,h in plan['inputs'].items())
    inputs=read(ROOT/'inputs/outer_predictions.csv')
    original={(r['model'],int(r['outer_fold']),int(r['source_row'])):r for r in inputs}
    groups=defaultdict(list)
    for r in read(ROOT/'results/ranked_decisions.csv'):
        groups[r['setting'],r['model'],int(r['outer_fold'])].append(r)
    stats={(r['setting'],r['model'],int(r['outer_fold'])):r for r in read(ROOT/'results/fold_results.csv')}
    selected={}
    for (setting,model,fold),rows in groups.items():
        assert len(rows)==(517 if fold<5 else 516)
        n=len(rows);k=int(rows[0]['k'])
        if setting.startswith('fixed_'):
            percent=int(setting.split('_')[1].replace('pct',''))
            assert k==percent*n//100
        else:
            assert k==sum(int(r['prediction']) for r in inputs if r['model']=='LR' and int(r['outer_fold'])==fold)
        assert sorted(int(r['rank']) for r in rows)==list(range(1,n+1))
        assert len({int(r['source_row']) for r in rows})==n
        ordered=sorted(rows,key=lambda r:int(r['rank']))
        for index,r in enumerate(ordered):
            raw=original[model,fold,int(r['source_row'])]
            assert float(raw['score'])==float(r['score']) and int(raw['label'])==int(r['label'])
            assert r['shift']==raw['shift'] and r['input_group']==raw['input_group']
            h=hashlib.sha256(('equal-review-v1|20261108|'+r['source_row']).encode('ascii')).hexdigest()
            assert h==r['tie_priority']
            assert int(r['selected'])==int(index<k)
            if index:
                prev=ordered[index-1]
                assert float(prev['score'])>=float(r['score'])
                if float(prev['score'])==float(r['score']):assert prev['tie_priority']<=r['tie_priority']
        chosen=[r for r in rows if r['selected']=='1'];selected[setting,model,fold]={int(r['source_row']) for r in chosen if r['label']=='1'}
        st=stats[setting,model,fold];tp=sum(int(r['label']) for r in chosen)
        assert int(st['review_count'])==len(chosen)==k
        assert int(st['tp'])==tp and int(st['fp'])==k-tp and int(st['fn'])==34-tp
        assert abs(float(st['precision'])-tp/k)<1e-14 and abs(float(st['recall'])-tp/34)<1e-14
    for r in read(ROOT/'results/pooled_results.csv'):
        rr=[x for (setting,m,f),x in stats.items() if setting==r['setting'] and m==r['model']]
        assert len(rr)==5
        for k in ['n','positives','review_count','tp','fp','fn','tp_min_cutoff_tie','tp_max_cutoff_tie']:assert int(r[k])==sum(int(x[k]) for x in rr)
    for r in read(ROOT/'results/paired_fold_comparisons.csv'):
        a=selected[r['setting'],r['model_a'],int(r['outer_fold'])];b=selected[r['setting'],r['model_b'],int(r['outer_fold'])]
        assert int(r['shared_tp'])==len(a&b) and int(r['a_only_tp'])==len(a-b) and int(r['b_only_tp'])==len(b-a)
        assert int(r['both_missed'])==34-len(a|b) and int(r['tp_a_minus_b'])==len(a)-len(b)
    result={'status':'completed','passed':True,'scope':'Independent trace from source rows to saved rank order, exact per-fold capacities, selected counts, pooled counts and all paired positive-list comparisons. No model replay/refitting or population inference.','groups_verified':len(groups),'ranked_rows_verified':sum(map(len,groups.values())),'elapsed_seconds':time.perf_counter()-start}
    checks=ROOT/'checks';checks.mkdir(exist_ok=True)
    (checks/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    # Revoke an earlier success before any current verification is attempted.
    checks=ROOT/'checks';checks.mkdir(exist_ok=True)
    status=checks/'verification.json'
    status.write_text(json.dumps({'passed':False,'status':'running'}),encoding='utf-8')
    try:main()
    except Exception as e:
        status.write_text(json.dumps({'passed':False,'status':'failed','error':repr(e)},ensure_ascii=False,indent=2),encoding='utf-8')
        raise
