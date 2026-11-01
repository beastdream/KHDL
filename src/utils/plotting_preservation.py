"""Validate the exact presentation-only migration against pinned historical evidence.

The model metadata continues to refer to the original research notebook. Its
bytes are archived, not replaced with a new checksum. All other notebook content
(including stored outputs) must remain identical.
"""
import json
from src.utils.paths import ROOT, REPORTS, sha256

OLD = '    center=0,\n'
NEW = '    vmin=-1, vmax=1,\n'


def verify_notebook_presentation():
    original = REPORTS / 'warning_resolution/notebooks__house_price_eda.ipynb'
    meta = json.loads((ROOT / 'artifacts/model_metadata.json').read_text(encoding='utf-8'))
    assert sha256(original) == meta['notebook_sha256'], 'Historical research notebook must remain pinned'
    old = json.loads(original.read_text(encoding='utf-8'))
    current = json.loads((ROOT / 'notebooks/house_price_eda.ipynb').read_text(encoding='utf-8'))
    source = ''.join(old['cells'][15]['source'])
    assert source.count(OLD) == 1 and 'sns.heatmap(' in source
    old['cells'][15]['source'] = source.replace(OLD, NEW).splitlines(keepends=True)
    assert current == old, 'Only the reviewed symmetric heatmap limits may differ'
    return True


def verify_eda_presentation():
    original = REPORTS / 'warning_resolution/src__data__eda.py'
    reference = json.loads((REPORTS / 'ui_refactor_before.json').read_text(encoding='utf-8'))
    assert sha256(original) == reference['protected_hashes']['src/data/eda.py']
    expected = original.read_text(encoding='utf-8').replace('import json\n', 'import json\nimport inspect\n', 1)
    expected = expected.replace("    sns.set_theme(style='whitegrid')\n",
        "    sns.set_theme(style='whitegrid')\n"
        "    # Seaborn 0.13 forwards a deprecated bool; opt into the modern bxp API.\n"
        "    # Older Matplotlib without orientation retains its original supported call.\n"
        "    box_options = ({'vert': None, 'orientation': 'vertical'}\n"
        "                   if 'orientation' in inspect.signature(matplotlib.axes.Axes.bxp).parameters else {})\n", 1)
    for feature in ('OverallQual', 'BedroomAbvGr'):
        expected = expected.replace(f"sns.boxplot(data=df,x='{feature}',y=TARGET)",
                                    f"sns.boxplot(data=df,x='{feature}',y=TARGET,**box_options)", 1)
    expected = expected.replace("    sns.heatmap(corr,annot=True,fmt='.2f',cmap='vlag',vmin=-1,vmax=1,center=0)",
        "    # Symmetric limits already center zero; avoid redundant Seaborn recentering.\n"
        "    sns.heatmap(corr,annot=True,fmt='.2f',cmap='vlag',vmin=-1,vmax=1)", 1)
    assert (ROOT / 'src/data/eda.py').read_text(encoding='utf-8') == expected, 'Unexpected EDA change beyond plotting compatibility'
    return True
