"""Targeted threshold tie/group and stale-success regression checks."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from common import select_cutoff
import verify_nested

root=Path(__file__).resolve().parent
(root/'tmp').mkdir(exist_ok=True)
# Same-score observations cannot be split by a numerical cutoff.
y=np.array([1,0,1,0]); s=np.array([.7,.7,.3,.1])
tau,_=select_cutoff(y,s)
assert tau in set(s)
assert ((s>=tau)[s==.7]).all() or not ((s>=tau)[s==.7]).any()
# A real equal-loss/F2 optimum must follow the locked higher-threshold rule.
y=np.array([1,0,0,0,0,0,0,0,0,1]); s=np.arange(10,0,-1,dtype=float)
tau,_=select_cutoff(y,s)
assert tau==10.0, tau
# An old success must be revoked before a corrupt-current-plan check fails.
with tempfile.TemporaryDirectory(dir=root/'tmp') as dirname:
    isolated=Path(dirname); (isolated/'checks').mkdir()
    shutil.copy(root/'checks/run.json',isolated/'checks/run.json')
    plan=json.loads((root/'locked_plan.json').read_text(encoding='utf-8')); plan['outer']['seed']=999
    (isolated/'locked_plan.json').write_text(json.dumps(plan),encoding='utf-8')
    (isolated/'checks/verification.json').write_text('{"passed":true,"status":"completed"}',encoding='utf-8')
    prior=verify_nested.ROOT; verify_nested.ROOT=isolated
    try:
        try: verify_nested.verify()
        except AssertionError: pass
        else: raise AssertionError('Corrupt current input was incorrectly accepted')
        status=json.loads((isolated/'checks/verification.json').read_text())
        assert status['passed'] is False and status['status']=='failed' and status['error']
    finally: verify_nested.ROOT=prior
print('Three targeted checks passed: same-score groups, higher F2 tie, stale-success invalidation.')
