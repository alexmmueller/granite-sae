"""Offline integrity checks for the complete provisional contrast subset."""
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def test_complete_subset_and_scope():
    data = json.loads((ROOT / 'feature_validation_results.json').read_text())
    rows = data['features']
    assert len(rows) == len({(r['layer'], r['feature']) for r in rows}) == 36
    assert Counter(r['layer'] for r in rows) == {10: 12, 20: 12, 30: 12}
    assert Counter(r['disposition'] for r in rows) == data['counts'] == {
        'supported-pattern': 24, 'inconclusive': 4, 'ABSTAIN': 8}
    assert sum(len(r['pairs']) for r in rows) == data['synthetic_pairs'] == 112
    assert data['synthetic_sequences'] == 224
    assert data['natural_replay'] == {
        'records': 300, 'sequences': 298, 'max_abs_error': 0.0, 'failures': []}
    assert data['independent_assessment'] == {
        'status': 'UNFINISHED', 'judge_calls': 0, 'heldout_metrics': None}
    for layer in (10, 20, 30):
        counts = Counter(r['stratum'] for r in rows if r['layer'] == layer)
        assert sorted(counts.values()) == [2, 4, 6]


def test_numeric_scoring_and_no_source_text():
    data = json.loads((ROOT / 'feature_validation_results.json').read_text())
    for row in data['features']:
        assert 0 <= row['feature'] < 81920
        if row['disposition'] == 'ABSTAIN':
            assert not row['pairs']
            continue
        assert len(row['pairs']) == 4
        q = row['q25_top_example_peak']
        passed = reversed_pairs = 0
        for pair in row['pairs']:
            p, n = pair['positive'], pair['negative']
            assert math.isfinite(p) and math.isfinite(n) and min(p, n) >= 0
            assert math.isclose(pair['difference'], p-n, abs_tol=1e-12)
            ok = p >= .25*q and p-n >= .10*q
            assert pair['pass'] == ok
            passed += ok
            reversed_pairs += n >= .25*q and n-p >= .10*q
        expected = ('supported-pattern' if passed >= 3 else
                    'refuted-contrast-prediction' if passed <= 1 and reversed_pairs >= 2
                    else 'inconclusive')
        assert row['disposition'] == expected
    def inspect(value):
        if isinstance(value, dict):
            assert not ({'text', 'token_ids', 'tokens', 'source_url', 'document_id',
                         'doc_key', 'target_char_span'} & value.keys())
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
    inspect(data)


def test_manifest_annotation_scope():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    assert manifest['labels_included'] is True
    assert manifest['provisional_subset_descriptions_included'] is True
    assert manifest['full_dictionary_labels_included'] is False
    assert manifest['data_excerpts_included'] is False
    assert manifest['license_status'] == manifest['license'] == 'Apache-2.0'
    assert manifest['publication_status'] == 'EXPERIMENTAL_RESEARCH_RELEASE'
    assert manifest['licensing']['copyright'] == 'Copyright 2026 Alex M. Müller'
    assert manifest['licensing']['license_file'] == 'LICENSE'
    import hashlib
    assert hashlib.sha256((ROOT / 'LICENSE').read_bytes()).hexdigest() == manifest['licensing']['canonical_license_sha256'] == 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'
    assert 'Copyright 2026 Alex M. Müller' in (ROOT / 'NOTICE').read_text()
    for artifact in manifest['artifacts']:
        assert artifact['license'] == artifact['metadata']['license'] == 'Apache-2.0'
        assert json.loads((ROOT / f"layer{artifact['layer']}/metadata.json").read_text()) == artifact
        # Weight files are optional downloads, excluded from the Git checkout.
        # When available, also check their preserved export-time metadata.
        if (ROOT / artifact['path']).is_file():
            with (ROOT / artifact['path']).open('rb') as stream:
                header = json.loads(stream.read(int.from_bytes(stream.read(8), 'little')))
            assert header['__metadata__']['license'] == artifact['embedded_metadata_provenance']['license'] == 'LICENSE_CHOICE_PENDING'
        assert artifact['embedded_metadata_provenance']['superseded_by'] == 'LICENSE'
    assert manifest['feature_validation']['independent_heldout_metrics'] is None
