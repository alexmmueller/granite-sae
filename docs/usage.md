# Usage

Python 3.11 or newer. The reference loader uses CPU PyTorch; it does not require the base model to inspect or reconstruct saved residual activations.

```bash
pip install -r requirements-test.txt
python -m pytest -q -p no:cacheprovider
python verify_release.py
```

Download the three weight files from the [v0.1.0 release](https://github.com/alexmmueller/granite-sae/releases/tag/v0.1.0) into their corresponding layer directories:

```bash
for layer in 10 20 30; do
  curl --fail --location \
    "https://github.com/alexmmueller/granite-sae/releases/download/v0.1.0/layer${layer}-sae.safetensors" \
    --output "layer${layer}/sae.safetensors"
done
python verify_release.py --weights
```

Each file is approximately 1.68 GB; all three total approximately 5.03 GB. Loading a dictionary requires additional RAM.

```python
from loader import GraniteSAE
import torch

sae = GraniteSAE.load('layer10/sae.safetensors')
h = torch.randn(2, 2560)  # synthetic example, not semantic evidence
features = sae.encode(h)
reconstructed = sae.reconstruct(h)
```

Inputs and reconstructed outputs use original model activation units. Encode only eligible positions: training excluded local position zero. The optional `replace_block_output` adapter preserves local position zero and tuple auxiliary outputs.

The trained hook is the full output of `model.model.layers[10]`, `[20]` or `[30]`, not residual-pre, attention output or MLP output. Preserve the pinned base-model revision, normalization and TopK-before-ReLU convention in [technical details](technical.md).

Generic SAELens loading, cached generation, padding handling and hosted inference have not been validated. No activating text examples or base-model weights are distributed. Training and full-model audits require additional dependencies beyond this inference/test environment.
