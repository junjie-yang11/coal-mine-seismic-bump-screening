"""Supplementary equal-review analysis of saved outer scores; no model fitting."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (ROOT / 'results' / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def priority(row):
    return hashlib.sha256(f'equal-review-v1|20261108|{row}'.encode('ascii')).hexdigest()


def select(rows, k):
    assert 0 <= k <= len(rows)
    return sorted(rows, key=lambda r: (-r['score'], priority(r['source_row']), r['source_row']))[:k]


def counts(rows, selected):
    tp = sum(r['label'] for r in selected)
    positives = sum(r['label'] for r in rows)
    k = len(selected)
    return {'n': len(rows), 'positives': positives, 'review_count': k,
            'tp': tp, 'fp': k-tp, 'fn': positives-tp,
            'precision': tp/k if k else 0., 'recall': tp/positives if positives else 0.,
            'actual_review_rate': k/len(rows)}


def main():
    started = time.perf_counter()
    (ROOT / 'results').mkdir(exist_ok=True)
    plan_path = ROOT / 'analysis_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    for name, expected in plan['inputs'].items():
        assert hashlib.sha256((ROOT/'inputs'/name).read_bytes()).hexdigest() == expected, name
    rows = read(ROOT/'inputs/outer_predictions.csv')
    metrics = read(ROOT/'inputs/outer_metrics.csv')
    for r in rows:
        for k in ['outer_fold','source_row','input_group','label','prediction']:
            r[k] = int(r[k])
        r['score'] = float(r['score'])
        assert math.isfinite(r['score']) and r['label'] in (0,1)
    models = plan['models']
    blocks = {(m,f): [r for r in rows if r['model']==m and r['outer_fold']==f] for m in models for f in range(1,6)}
    for m in models:
        cohort = [r for r in rows if r['model']==m]
        assert len(cohort)==2584 and len({r['source_row'] for r in cohort})==2584
        assert sum(r['label'] for r in cohort)==170
        for f in range(1,6):
            b = blocks[m,f]
            assert len(b)==(517 if f<5 else 516) and sum(r['label'] for r in b)==34
            assert {(r['source_row'],r['label'],r['shift'],r['input_group']) for r in b} == {(r['source_row'],r['label'],r['shift'],r['input_group']) for r in blocks['LR',f]}
            c = counts(b,[r for r in b if r['prediction']])
            saved = next(r for r in metrics if r['model']==m and int(r['outer_fold'])==f)
            assert all(c[k]==int(saved[k]) for k in ['tp','fp','fn'])
    settings = [('fixed_20pct',.20),('fixed_10pct',.10),('fixed_30pct',.30),('LR_count_reference',None)]
    fold_results, selections, tie_audit, paired = [], [], [], []
    selected_ids = {}
    for setting, budget in settings:
        for f in range(1,6):
            k = math.floor(budget*len(blocks['LR',f])) if budget is not None else sum(r['prediction'] for r in blocks['LR',f])
            for m in models:
                b = blocks[m,f]
                chosen = select(b,k)
                ids = {r['source_row'] for r in chosen}
                # A label change must leave the list unchanged.
                flipped = [{**r,'label':1-r['label']} for r in b]
                assert ids == {r['source_row'] for r in select(flipped,k)}
                selected_ids[setting,m,f] = ids
                if setting=='LR_count_reference' and m=='LR':
                    assert ids=={r['source_row'] for r in b if r['prediction']}, 'LR comparison must recover its original list.'
                cutoff = chosen[-1]['score'] if k else None
                tied = [r for r in b if r['score']==cutoff] if cutoff is not None else []
                above = [r for r in b if cutoff is not None and r['score']>cutoff]
                slots = k-len(above)
                tied_pos = sum(r['label'] for r in tied)
                above_tp = sum(r['label'] for r in above)
                min_tp = above_tp+max(0,slots-(len(tied)-tied_pos))
                max_tp = above_tp+min(slots,tied_pos)
                common = {'setting':setting,'nominal_budget':budget if budget is not None else '', 'outer_fold':f,'model':m}
                fold_results.append({**common,**counts(b,chosen), 'tp_min_cutoff_tie':min_tp,'tp_max_cutoff_tie':max_tp})
                tie_audit.append({**common,'k':k,'cutoff_score':cutoff if cutoff is not None else '', 'above_cutoff_count':len(above),'cutoff_tie_count':len(tied),'cutoff_tie_positive_count':tied_pos,'cutoff_tie_selected':slots,'cutoff_tie_excluded':len(tied)-slots,'tie_split':bool(slots and slots<len(tied)),'tp_min_cutoff_tie':min_tp,'tp_max_cutoff_tie':max_tp})
                ordered = sorted(b,key=lambda r:(-r['score'],priority(r['source_row']),r['source_row']))
                for rank,r in enumerate(ordered,1):
                    selections.append({**common,'source_row':r['source_row'],'input_group':r['input_group'],'shift':r['shift'],'label':r['label'],'score':repr(r['score']),'tie_priority':priority(r['source_row']),'rank':rank,'selected':int(r['source_row'] in ids),'k':k})
            for a,b in itertools.combinations(models,2):
                positive = {r['source_row'] for r in blocks[a,f] if r['label']}
                aa=selected_ids[setting,a,f]&positive;bb=selected_ids[setting,b,f]&positive
                paired.append({'setting':setting,'outer_fold':f,'model_a':a,'model_b':b,'review_count_each':k,'shared_tp':len(aa&bb),'a_only_tp':len(aa-bb),'b_only_tp':len(bb-aa),'both_missed':len(positive-(aa|bb)),'tp_a_minus_b':len(aa)-len(bb)})
    pooled=[]
    for setting,budget in settings:
        for m in models:
            rr=[r for r in fold_results if r['setting']==setting and r['model']==m]
            totals={k:sum(r[k] for r in rr) for k in ['n','positives','review_count','tp','fp','fn','tp_min_cutoff_tie','tp_max_cutoff_tie']}
            pooled.append({'setting':setting,'nominal_budget':budget if budget is not None else '', 'model':m,**totals,'precision':totals['tp']/totals['review_count'],'recall':totals['tp']/totals['positives'],'actual_review_rate':totals['review_count']/totals['n']})
    write('fold_results.csv',fold_results);write('pooled_results.csv',pooled);write('ranked_decisions.csv',selections);write('cutoff_tie_audit.csv',tie_audit);write('paired_fold_comparisons.csv',paired)
    frozen=[]
    for m in models:
        b=[r for r in rows if r['model']==m];frozen.append({'model':m,**counts(b,[r for r in b if r['prediction']])})
    write('original_frozen_threshold_results.csv',frozen)
    rf=next(r for r in frozen if r['model']=='RF');lr=next(r for r in frozen if r['model']=='LR')
    delta={'tp_difference':rf['tp']-lr['tp'],'fp_difference':rf['fp']-lr['fp'],'review_difference':rf['review_count']-lr['review_count'],'net_additional_reviews_per_net_additional_detection':(rf['review_count']-lr['review_count'])/(rf['tp']-lr['tp']),'scope':'Descriptive net difference ratio for two non-nested saved lists; not money, time or causal marginal cost.'}
    (ROOT/'results/incremental_review_burden.json').write_text(json.dumps(delta,indent=2),encoding='utf-8')
    # Meaningful edge cases for exact-capacity ranking and ties.
    toy=[{'source_row':i,'score':1.,'label':i%2} for i in range(1,5)]
    assert select(toy,0)==[] and len(select(toy,4))==4
    assert len(select(toy,2))==2 and select(toy,2)==select(list(reversed(toy)),2)
    assert all(hashlib.sha256((ROOT/'inputs'/n).read_bytes()).hexdigest()==h for n,h in plan['inputs'].items())
    run={'scope':'Saved-score batch Top-K analysis; no model fitting or numeric threshold transfer. Descriptive only.','python':sys.version,'platform':platform.platform(),'elapsed_seconds':time.perf_counter()-started,'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'input_hashes':plan['inputs'],'fold_result_rows':len(fold_results),'ranked_decision_rows':len(selections),'label_flip_list_invariance_checks':len(fold_results),'lr_reference_matches_original_lists':True,'edge_checks':['zero K','all rows K','tied-score exact capacity','input-order independence'],'status':'completed'}
    checks=ROOT/'checks';checks.mkdir(exist_ok=True)
    (checks/'run_record.json').write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pooled':pooled,'net_frozen_RF_minus_LR':delta,'run':run},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
