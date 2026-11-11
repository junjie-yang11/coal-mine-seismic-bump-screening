"""Run the nature-figure PDF text and collision checks on actual exports."""
import importlib.util
import json
from pathlib import Path
import sys

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
args=parser.parse_args()
here=args.root.resolve()
skill=Path(__file__).resolve().parent/'图件检查工具'
def load(name):
    spec = importlib.util.spec_from_file_location(name, skill / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
collision = load('audit_figure_collisions')
text = load('audit_pdf_text')
status = []
for pdf in sorted((here / '图件/PDF').glob('*.pdf')):
    geometry = collision.audit_pdf(pdf)
    fonts = text.audit_pdf(pdf.read_bytes(), minimum_pt=5.0)
    assert fonts['auditable'], f'Font audit could not inspect {pdf}'
    assert geometry['verdict'] != 'NOT AUDITABLE', f'Collision audit could not inspect {pdf}'
    record = {'pdf': pdf.name, 'geometry': geometry, 'fonts': fonts}
    (here / '图件/QA' / f'{pdf.stem}.pdf_audit.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    status.append({'pdf': pdf.name, 'verdict': geometry['verdict'], 'geometry_summary': geometry['summary'],
                   'minimum_font_pt': fonts['minimum_found_pt'], 'below_minimum': fonts['below_minimum_count']})
    print(json.dumps(status[-1], ensure_ascii=False))
    if geometry['verdict'] == 'FIX BEFORE DELIVERY' or fonts['below_minimum_count']:
        print(json.dumps(geometry['findings'], ensure_ascii=False, indent=2))
if any(r['verdict'] == 'FIX BEFORE DELIVERY' or r['below_minimum'] for r in status):
    raise SystemExit(1)
assert len(status) == 10
