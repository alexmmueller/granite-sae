# Training and evaluation

## Model and hook

Base model: `ibm-granite/granite-4.2-3b`, revision `4e9c275e232f10211ec88644d106a1f6a1b60e06`.

Three independently trained TopK SAEs attach to full block outputs at zero-index layers 10, 20 and 30 (one-based blocks 11, 21 and 31). The base model has 40 blocks and width 2,560. Each SAE has 81,920 features, K=32, fp32 weights, and final checkpoint step 97,700. Frozen base-model forwards used bf16.

## Inference convention

For row vectors in original model activation units:

```text
x = (h - mean) / std
pre = (x - b_dec) @ W_enc + b_enc
values, indices = topk(pre, 32)
acts = zeros_like(pre).scatter(-1, indices, relu(values))
x_hat = acts @ W_dec + b_dec
h_hat = x_hat * std + mean
```

TopK runs before ReLU, across the full dictionary. Saved per-dimension mean/std have shape `[1, 2560]`; `b_dec` is in normalized coordinates. Use the saved std literally. No decoder renormalization, affine folding, index permutation or quantization was applied. TopK can produce fewer than 32 positive activations; tie-breaking across backends is not guaranteed.

Training excluded each sequence's local position zero. The reference hook adapter leaves that position unchanged, casts reconstructed activations to the original dtype/device and preserves tuple auxiliary outputs. Generation caches, padding, other hook runtimes and hosted inference are not validated. Decoder rows are not raw-unit steering vectors.

## Training

FineWeb-Edu `sample-10BT`, train split; seed 0; configured budget 200M tokens per SAE. This is the configured target, not an independently reconstructed consumed-token count. The streaming dataset revision and consumed document IDs were not preserved, limiting exact reproduction and overlap analysis.

Tokenizer configuration: no added special tokens; EOS appended between documents; packed length 512; 16 sequences per base forward; local position zero excluded. Each layer used a 500K activation shuffle buffer with 0.5 refill fraction and per-dimension normalization from seed activations.

SAE training used geometric-median decoder-bias initialization, Adam with peak learning rate 4e-4, 1,000 warmup steps, decay over the second half, activation batch size 2,048, projected decoder normalization, AuxK=512 with coefficient 1/32 and idle threshold 500, and dead-feature threshold/resampling interval 2,000 steps. Source and configuration: [train_sae.py](../provenance/train_sae.py), [config.py](../provenance/config.py).

## Reconstruction audit

The audit used 32 documents, represented as 32 prefixes and 32 packed sequences; the packed sequences draw on 17 of those same documents. These are not 64 independent samples. Each layer has 29,988 eligible positions/next-token targets.

Evaluation corpus: FineWeb-Edu revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, shard `013_00000.parquet`, skipping the first 1,000 rows. Training disjointness and cross-shard deduplication are not proven.

Intact next-token cross-entropy was **2.814258 nats/token**. Loss uses fp32 logits and shifted labels.

| Layer | Reconstruction CE | CE increase | Mean-ablation CE | Loss recovered | Normalized MSE | Centered FVU | Normalized variance explained |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | 2.895126 | 0.080868 | 10.685271 | 0.989726 | 0.312995 | 0.336825 | 0.663865 |
| 20 | 2.999158 | 0.184900 | 8.924596 | 0.969740 | 0.432185 | 0.457031 | 0.543826 |
| 30 | 3.035445 | 0.221187 | 8.317151 | 0.959805 | 0.371395 | 0.382983 | 0.617898 |

For normalized inputs x and error e=x_hat−x, pooled over eligible tokens/dimensions:

- MSE = sum(e²)/(N × d_model).
- Centered FVU = sum(e²)/sum((x−mean_audit(x))²).
- Variance explained = 1−sum((e−mean_audit(e))²)/sum((x−mean_audit(x))²). Centering the error makes this different from 1−FVU.
- Loss recovered = (CE_mean−CE_reconstruction)/(CE_mean−CE_intact). The baseline is destructive mean ablation; this is not percentage of model capability retained.

Identity replacement matched full logits exactly on all 192 layer/input pairs. Normalization roundtrip alone had maximum error 9.53674316e-07. See [audit code](../provenance/audit.py), its [math-only tests](../provenance/test_audit.py), and [complete results](../provenance/audit_summary.json). Running package tests does not rerun the full-model audit.

## Dictionary and export limits

Training dead-feature counts were 6,178 / 9,565 / 4,531, using at least 2,000 idle steps. Nonobserved audit features were 51,395 / 47,502 / 38,638; nonobservation in a small sample does not establish death. Always-on audit features occupied 8 / 5 / 1 of the K32 slots. Queries of 512 sampled decoder directions found 2 / 1 / 3 near-duplicate matches above cosine 0.99; these are not exhaustive unique-pair counts. Per-feature resampling ages are unavailable.

Exports contain seven inference tensors: W_enc, b_enc, W_dec, b_dec, mean, std and scalar int64 k; no optimizer state. Recorded export checks compared tensors bitwise and reference inference against independent checkpoint math on deterministic synthetic `[4,2560]` inputs per layer, with exact agreement. This validates implementation equivalence, not semantics or equivalence on saved natural residuals. Hashes, sizes, tensor shapes and provenance are in [manifest.json](../manifest.json).

The audit covers one precision/configuration and a modest sample, with no population confidence intervals or comparative-quality claim. Feature descriptions are provisional; see [feature evaluation](feature-validation.md). No complete label bank, token-aligned examples, SAELens conversion or feature-browser integration is included.

## Licensing

Custom code, weights, documentation and annotations: Apache-2.0; see [LICENSE](../LICENSE), [NOTICE](../NOTICE), and [RIGHTS.md](../RIGHTS.md). Base-model weights and corpus excerpts are not distributed. The original weight headers retain an export-time pending-licence marker; external metadata binds the current licence to the unchanged file hashes.
