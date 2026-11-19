"""Standalone inference-only custom Granite SAE. No base-model dependency.

Inputs and outputs are ORIGINAL model residual units; internal SAE units are
per-dimension standardized. No decoder renormalization or affine folding.
Use encode on eligible positions only (training excludes local position zero).
"""
from pathlib import Path
import torch
from safetensors.torch import load_file


class GraniteSAE:
    def __init__(self, tensors):
        self.t = tensors
        self.k = int(tensors['k'].item())
        if not 0 < self.k <= tensors['W_enc'].shape[1]:
            raise ValueError('invalid k')
        if not bool((tensors['std'] > 0).all()):
            raise ValueError('std must be positive')

    @classmethod
    def load(cls, path):
        return cls(load_file(str(Path(path)), device='cpu'))

    @torch.inference_mode()
    def encode(self, h):
        t = self.t
        x = (h.to(device=t['mean'].device, dtype=torch.float32) - t['mean']) / t['std']
        pre = (x - t['b_dec']) @ t['W_enc'] + t['b_enc']
        values, indices = torch.topk(pre, self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, indices, torch.relu(values))

    @torch.inference_mode()
    def reconstruct(self, h):
        t = self.t
        return (self.encode(h) @ t['W_dec'] + t['b_dec']) * t['std'] + t['mean']

    @torch.inference_mode()
    def replace_block_output(self, output):
        """CPU reference hook adapter, batch × local-sequence × model-width.

        Keeps position zero unchanged. Tuple auxiliary outputs are preserved.
        This is not a certified hosted-inference integration.
        """
        h = output[0] if isinstance(output, tuple) else output
        if h.ndim != 3:
            raise ValueError('expected batch, sequence, model-width')
        result = h.clone()
        if h.shape[1] > 1:
            result[:, 1:] = self.reconstruct(h[:, 1:]).to(device=h.device, dtype=h.dtype)
        return (result, *output[1:]) if isinstance(output, tuple) else result
