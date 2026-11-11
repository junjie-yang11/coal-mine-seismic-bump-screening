"""Synchronize the current ten PNG figures into a separately saved Word file."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
STEMS = ['fig01_data_profile','fig02_feature_structure','fig03_nested_workflow',
         'fig04_outer_ap','fig05_pr_curves','fig06_paired_folds','fig07_confusion',
         'fig08_detection_workload','fig09_fold_shift','fig10_overlap']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=ROOT/'论文/煤矿下一班次高能矿震分类预测研究_等复核量整合版.docx')
    parser.add_argument('--images', type=Path, default=ROOT/'图件/PNG')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.paper.resolve():
        parser.error('输出必须是尚不存在的新Word文件，避免覆盖正文。')
    namespaces = {'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
                  'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                  'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    replacements, records = {}, []
    with ZipFile(args.paper) as source:
        xml = ET.fromstring(source.read('word/document.xml'))
        rels = ET.fromstring(source.read('word/_rels/document.xml.rels'))
        targets = {node.get('Id'):node.get('Target') for node in rels}
        blips = xml.findall('.//w:body//a:blip', namespaces)
        assert len(blips) == len(STEMS) == 10
        for stem, blip in zip(STEMS, blips):
            part = str(PurePosixPath('word')/targets[blip.get('{'+namespaces['r']+'}embed')])
            old = source.read(part)
            new = (args.images/(stem+'.png')).read_bytes()
            assert old.startswith(b'\x89PNG\r\n\x1a\n') and new.startswith(b'\x89PNG\r\n\x1a\n')
            assert struct.unpack('>II',old[16:24]) == struct.unpack('>II',new[16:24]), stem+'图片像素尺寸改变，请先确认论文缩放比例。'
            replacements[part] = new
            records.append({'figure':stem,'media_part':part,'sha256':hashlib.sha256(new).hexdigest()})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with ZipFile(args.output,'w') as dest:
            for item in source.infolist():
                dest.writestr(item,replacements.get(item.filename,source.read(item.filename)))
    with ZipFile(args.paper) as a, ZipFile(args.output) as b:
        assert a.namelist()==b.namelist()
        assert all(a.read(name)==b.read(name) for name in a.namelist() if name not in replacements)
    report = {'scope':'Only figure media replaced; all nonmedia DOCX parts identical',
              'source':str(args.paper),'output':str(args.output),'figures':records}
    args.output.with_suffix('.图件同步记录.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.output)


if __name__=='__main__':
    main()
