# Granite sparse autoencoders

Three TopK sparse autoencoders trained on block outputs of [IBM Granite 4.2-3B](https://huggingface.co/ibm-granite/granite-4.2-3b): layers 10, 20 and 30, with 81,920 features and K=32 per dictionary.

Includes a CPU reference loader, training and audit code, reconstruction results, and a small provisional feature-description subset.

**[Download weights](https://github.com/alexmmueller/granite-sae/releases/tag/v0.1.0) · [Usage](docs/usage.md) · [Evaluation](docs/technical.md) · [Feature descriptions](docs/feature-validation.md)**

Experimental research release. Reconstruction fidelity has been measured; feature meanings are not independently validated. No claim of monosemanticity, causal control, or parity with other SAE releases.

Apache-2.0. See [rights and attribution](RIGHTS.md). Weights are separate downloads, not stored in Git.
