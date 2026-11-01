"""Checksum exceptions are explicit, version-pinned, and never rewrite the baseline."""
import json
from pathlib import Path
from src.utils.paths import ROOT, REPORTS, sha256


def audit_preservation(root=ROOT, baseline=None, approvals=None):
    root=Path(root).resolve()
    if baseline is None:
        baseline=json.loads((REPORTS/'week2_preservation_before.json').read_text(encoding='utf-8'))
    if approvals is None:
        approvals=json.loads((REPORTS/'week2_preservation_approvals.json').read_text(encoding='utf-8'))['files']
    allowed={(root/'data/processed'/f'{n}.csv').resolve() for n in ('train','valid')}
    entries=[]
    for name,original in baseline.items():
        relative=name.replace('\\','/')
        path=(root/relative).resolve()
        row={'path':relative,'original_sha256':original}
        # Decide BEFORE opening bytes. Also reject path traversal and raw non-CSV files.
        forbidden=(not path.is_relative_to(root) or (path.is_relative_to(root/'data/raw'))
                   or (path.suffix.lower()=='.csv' and path not in allowed))
        if forbidden:
            row.update(category='C',status='NOT VERIFIED',reason='Data Guard: file not opened, checksum not computed')
        elif not path.is_file():
            row.update(category='B',status='MISSING',reason='Restore only after review; no overwrite performed')
        else:
            current=sha256(path)
            row['current_sha256']=current
            approval=approvals.get(relative,{})
            if current==original:
                row.update(category='UNCHANGED',status='VERIFIED')
            elif (approval.get('original_sha256')==original and approval.get('approved_sha256')==current
                  and approval.get('reason') and approval.get('authorization')):
                row.update(category='A',status='APPROVED CHANGE',reason=approval['reason'],authorization=approval['authorization'])
            else:
                row.update(category='B',status='UNEXPECTED CHANGE',reason='Unapproved hash; preserve user file and review diff before restore')
        entries.append(row)
    return {'status':'PASS' if all(e['category'] in ('UNCHANGED','A') for e in entries) else 'NOT READY',
            'entries':entries,
            'restricted_data':{'raw':'NOT VERIFIED (not opened)','test_15_percent':'NOT VERIFIED (not opened or rehashed)'},
            'baseline_rewritten':False}
