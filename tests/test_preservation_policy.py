import hashlib
from src.utils.preservation import audit_preservation


def test_preservation_approved_change_is_version_pinned(tmp_path):
    file=tmp_path/'requirements.txt'
    file.write_text('reviewed')
    old=hashlib.sha256(b'old').hexdigest()
    approved=hashlib.sha256(b'reviewed').hexdigest()
    rules={'requirements.txt':{'original_sha256':old,'approved_sha256':approved,'reason':'Reviewed additions','authorization':'Explicit test approval'}}
    assert audit_preservation(tmp_path,{'requirements.txt':old},rules)['status']=='PASS'
    file.write_text('unreviewed later change')
    result=audit_preservation(tmp_path,{'requirements.txt':old},rules)
    assert result['status']=='NOT READY' and result['entries'][0]['category']=='B'


def test_preservation_forbidden_paths_never_hashed(tmp_path,monkeypatch):
    from src.utils import preservation
    def forbidden(*args):
        raise AssertionError('Must classify before reading bytes')
    monkeypatch.setattr(preservation,'sha256',forbidden)
    result=audit_preservation(tmp_path,{'data/raw/train.csv':'original','data/processed/test.csv':'original','data/raw/raw.txt':'original'}, {})
    assert result['status']=='NOT READY'
    assert all(e['category']=='C' and e['status']=='NOT VERIFIED' for e in result['entries'])
    assert all('current_sha256' not in e for e in result['entries'])
