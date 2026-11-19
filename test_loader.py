import importlib.util
from pathlib import Path
import torch


def test_literal_math_negative_topk_and_inverse():
    path = Path(__file__).with_name('loader.py')
    assert path.exists(), 'standalone loader not implemented'
    spec = importlib.util.spec_from_file_location('loader', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    t = {'W_enc': torch.eye(3), 'W_dec': torch.eye(3) * 2,
         'b_enc': torch.tensor([-5., -4., -3.]), 'b_dec': torch.tensor([.2, .3, .4]),
         'mean': torch.tensor([[2., 4., 6.]]), 'std': torch.tensor([[2., 3., 4.]]),
         'k': torch.tensor(2)}
    sae = m.GraniteSAE(t)
    h = torch.tensor([[2., 4., 6.], [20., 4., 6.]])
    x = (h-t['mean'])/t['std']
    pre = (x-t['b_dec']) @ t['W_enc'] + t['b_enc']
    v, i = torch.topk(pre, 2, dim=-1)
    acts = torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))
    expected = (acts @ t['W_dec'] + t['b_dec']) * t['std'] + t['mean']
    assert torch.equal(sae.encode(h), acts)
    assert torch.equal(sae.reconstruct(h), expected)
    assert sae.encode(h)[0].count_nonzero() == 0


def test_hook_preserves_excluded_position_and_dtype():
    import loader
    t = {'W_enc': torch.eye(3), 'W_dec': torch.eye(3),
         'b_enc': torch.zeros(3), 'b_dec': torch.zeros(3),
         'mean': torch.zeros(1,3), 'std': torch.ones(1,3), 'k': torch.tensor(1)}
    sae = loader.GraniteSAE(t)
    h = torch.tensor([[[3., 2., 1.], [1., 3., 2.]]], dtype=torch.bfloat16)
    out = sae.replace_block_output(h)
    assert out.dtype == h.dtype
    assert torch.equal(out[:,0], h[:,0])
    assert torch.equal(out[:,1], torch.tensor([[0.,3.,0.]], dtype=h.dtype))
    assert torch.equal(h[:,1], torch.tensor([[1.,3.,2.]], dtype=h.dtype))
    tup = sae.replace_block_output((h, 'unchanged'))
    assert tup[1] == 'unchanged'
    assert torch.equal(tup[0], out)


def test_package_integrity_and_scope():
    import json
    import hashlib
    root = Path(__file__).resolve().parent
    sums = root/'SHA256SUMS'
    assert sums.exists(), 'package checksum inventory missing'
    for line in sums.read_text().splitlines():
        expected, rel = line.split('  ',1)
        with (root/rel).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest() == expected, rel
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['license_status'] == 'Apache-2.0'
    assert manifest['optimizer_state_included'] is False
    assert manifest['data_excerpts_included'] is False
    assert {x['layer'] for x in manifest['artifacts']} == {10,20,30}
    from safetensors import safe_open
    for artifact in manifest['artifacts']:
        # Binary weights are separate downloads; validate them when present.
        if not (root/artifact['path']).is_file():
            continue
        with safe_open(str(root/artifact['path']),framework='pt') as f:
            assert set(f.keys()) == {'W_enc','W_dec','b_enc','b_dec','mean','std','k'}
            assert f.get_tensor('k').item() == 32
            assert f.metadata()['model_revision'] == '4e9c275e232f10211ec88644d106a1f6a1b60e06'
