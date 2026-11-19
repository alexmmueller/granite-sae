"""Training implementation for Granite 4.2-3B TopK SAEs.

A frozen, pinned bf16 base model supplies block-output activations from packed
FineWeb-Edu sequences. Each layer has a separate shuffle buffer and fp32 SAE.
Local position zero is excluded. Seed activations define per-dimension mean/std
and geometric-median decoder-bias initialization. Training uses TopK before ReLU,
AuxK loss, projected decoder gradients and periodic dead-feature resampling with
optimizer-state reset. No causal claim about other models follows from these
normalization and exclusion choices.

Checkpoints include optimizer state and firing counters; inference exports do not.
The streaming dataset revision is unpinned and consumed document IDs are not saved.
Resuming restores the SAE state, not the exact corpus iterator or shuffle buffer.
The base-input-token pacing counter differs from sampled activation draws.
"""
import os
import time
import argparse
import torch
import torch.nn as nn
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import config as C

parser = argparse.ArgumentParser()
parser.add_argument("--token_budget", type=int, default=None,
                     help="Override C.TOKEN_BUDGET, e.g. 20_000_000 for a staged measurement run.")
parser.add_argument("--resume", action="store_true",
                     help="Resume from the latest checkpoint per layer in C.CHECKPOINT_DIR, if any exist.")
parser.add_argument("--layer", type=int, default=None,
                     help="Train only this layer in the current process; reduces optimizer memory.")
args = parser.parse_args()
if args.token_budget is not None:
    C.TOKEN_BUDGET = args.token_budget
    C.TOTAL_SAE_STEPS = C.TOKEN_BUDGET // C.TRAIN_BATCH_SIZE
    print(f"[override] TOKEN_BUDGET={C.TOKEN_BUDGET:,}  TOTAL_SAE_STEPS={C.TOTAL_SAE_STEPS:,}")
if args.layer is not None:
    assert args.layer in C.TARGET_LAYERS, f"--layer {args.layer} not in config.TARGET_LAYERS={C.TARGET_LAYERS}"
    C.TARGET_LAYERS = [args.layer]
    print(f"[override] training only layer {args.layer} in this process")

torch.manual_seed(C.SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "Training requires a CUDA GPU."


# ---------------------------------------------------------------------------
# 1. Load frozen base model (pinned revision)
# ---------------------------------------------------------------------------
print(f"Loading {C.MODEL_NAME} @ {C.MODEL_REVISION} ...")
tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME, revision=C.MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(
    C.MODEL_NAME, revision=C.MODEL_REVISION,
    dtype=getattr(torch, C.BASE_MODEL_DTYPE),
).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)


# ---------------------------------------------------------------------------
# 2. Streaming data pipeline: tokenize + pack into fixed-length sequences
# ---------------------------------------------------------------------------
print(f"Streaming {C.DATASET_NAME}/{C.DATASET_CONFIG} ...")
corpus = load_dataset(C.DATASET_NAME, C.DATASET_CONFIG, split="train", streaming=True)

def token_packer(stream, seq_len=C.SEQ_LEN, batch=C.FORWARD_BATCH_SEQS):
    buf = []
    for doc in stream:
        ids = tokenizer.encode(doc["text"], add_special_tokens=False)
        buf.extend(ids + [tokenizer.eos_token_id])
        while len(buf) >= batch * seq_len:
            chunk = buf[: batch * seq_len]
            buf = buf[batch * seq_len :]
            yield torch.tensor(chunk, dtype=torch.long).view(batch, seq_len)

token_stream = token_packer(iter(corpus))


# ---------------------------------------------------------------------------
# 3. Multi-layer activation capture (one forward pass -> all target layers)
# ---------------------------------------------------------------------------
captured = {}

def make_hook(layer_idx):
    def hook(module, inp, out):
        captured[layer_idx] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook

handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in C.TARGET_LAYERS]


# ---------------------------------------------------------------------------
# TopK SAE: select preactivations before ReLU
# ---------------------------------------------------------------------------
class TopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k, b_dec_init):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.01)
        self.b_dec = nn.Parameter(b_dec_init.clone())

    def forward(self, x):
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc
        topk_vals, topk_idx = torch.topk(pre_acts, self.k, dim=-1)
        topk_vals = torch.relu(topk_vals)
        acts = torch.zeros_like(pre_acts).scatter_(-1, topk_idx, topk_vals)
        recon = acts @ self.W_dec + self.b_dec
        return recon, acts, pre_acts


def geometric_median(X, n_iter=30, eps=1e-6):
    y = X.mean(dim=0)
    for _ in range(n_iter):
        dists = torch.norm(X - y, dim=1).clamp(min=eps)
        w = 1.0 / dists
        y = (w.unsqueeze(1) * X).sum(0) / w.sum()
    return y


def find_latest_checkpoint(layer_idx):
    """Return the path to this layer's checkpoint if one exists, else None."""
    path = f"{C.CHECKPOINT_DIR}/layer{layer_idx}_latest.pt"
    return path if os.path.exists(path) else None


class LayerSAETrainer:
    """One shuffle buffer + one SAE + one optimizer, per target layer."""

    def __init__(self, layer_idx, seed_activations, resume_path=None):
        self.layer_idx = layer_idx
        mean = seed_activations.mean(0, keepdim=True)
        # Per-dimension standardization from bootstrap activations.
        std = seed_activations.std(dim=0, keepdim=True).clamp(min=1e-6)
        self.mean, self.std = mean.to(device), std.to(device)
        norm = (seed_activations - mean) / std
        geo_median = geometric_median(norm[torch.randperm(norm.shape[0])[: min(2000, norm.shape[0])]].to(device))
        self.sae = TopKSAE(C.D_MODEL, C.D_SAE, C.K, geo_median).to(device)
        self.opt = torch.optim.Adam(self.sae.parameters(), lr=C.LR)
        self.buffer = norm  # kept on CPU -- only minibatches move to GPU, see train_step
        self.step = 0
        self.steps_since_fired = torch.zeros(C.D_SAE, dtype=torch.long, device=device)
        if resume_path is not None:
            self.load_checkpoint(resume_path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=device)
        self.sae.load_state_dict(ckpt["sae_state"])
        self.opt.load_state_dict(ckpt["opt_state"])
        self.step = ckpt["step"]
        self.steps_since_fired = ckpt["steps_since_fired"].to(device)
        self.mean = ckpt["mean"].to(device)
        self.std = ckpt["std"].to(device)
        print(f"  [resume] layer{self.layer_idx}: loaded checkpoint at step {self.step} from {path}", flush=True)

    def add_activations(self, raw_acts):
        norm = (raw_acts.to(device) - self.mean) / self.std
        self.buffer = torch.cat([self.buffer, norm.cpu()], dim=0)
        # Keep only the most recent activations within the configured capacity.
        if self.buffer.shape[0] > C.SHUFFLE_BUFFER_TOKENS:
            self.buffer = self.buffer[-C.SHUFFLE_BUFFER_TOKENS :]

    def resample_dead(self):
        dead_mask = self.steps_since_fired >= C.DEAD_THRESHOLD_STEPS
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return 0
        with torch.no_grad():
            idx = torch.randint(0, self.buffer.shape[0], (n_dead,))
            new_dir = self.buffer[idx].to(device) - self.sae.b_dec
            new_dir = new_dir / (new_dir.norm(dim=-1, keepdim=True) + 1e-8)
            dead_idx = dead_mask.nonzero(as_tuple=True)[0]
            self.sae.W_dec.data[dead_idx] = new_dir
            self.sae.W_enc.data[:, dead_idx] = new_dir.T * 0.2
            self.sae.b_enc.data[dead_idx] = 0.0
            for p, kind in [(self.sae.W_dec, "dec"), (self.sae.W_enc, "enc"), (self.sae.b_enc, "benc")]:
                state = self.opt.state.get(p)
                if state and "exp_avg" in state:
                    if kind == "dec":
                        state["exp_avg"][dead_idx] = 0; state["exp_avg_sq"][dead_idx] = 0
                    elif kind == "enc":
                        state["exp_avg"][:, dead_idx] = 0; state["exp_avg_sq"][:, dead_idx] = 0
                    else:
                        state["exp_avg"][dead_idx] = 0; state["exp_avg_sq"][dead_idx] = 0
        self.steps_since_fired[dead_mask] = 0
        return n_dead

    def train_step(self):
        warmup_scale = min(1.0, (self.step + 1) / C.WARMUP_STEPS)
        # Linear decay over the back half of training, down to 10% of peak LR.
        decay_start = C.TOTAL_SAE_STEPS * 0.5
        if self.step < decay_start:
            decay_scale = 1.0
        else:
            progress = (self.step - decay_start) / max(1, C.TOTAL_SAE_STEPS - decay_start)
            decay_scale = max(0.1, 1.0 - 0.9 * progress)
        lr_scale = warmup_scale * decay_scale
        for g in self.opt.param_groups:
            g["lr"] = C.LR * lr_scale
        idx = torch.randint(0, self.buffer.shape[0], (C.TRAIN_BATCH_SIZE,))
        batch = self.buffer[idx].to(device)
        recon, acts, pre_acts = self.sae(batch)
        loss = ((recon - batch) ** 2).mean()

        # AuxK: give currently-idle features their own gradient signal by having
        # them try to reconstruct the residual the main top-k path left
        # unexplained, instead of relying purely on periodic random resampling
        # to revive them. Standard fix for TopK winner-take-all entrenchment
        # (OpenAI, "Scaling and evaluating sparse autoencoders") -- adopted
        # after two failed validation attempts at a decoder-norm-based fix.
        idle_mask = self.steps_since_fired >= C.AUX_DEAD_THRESHOLD_STEPS
        n_idle = int(idle_mask.sum().item())
        if n_idle > 0:
            aux_k = min(C.AUX_K, n_idle)
            pre_acts_idle = pre_acts.masked_fill(~idle_mask.unsqueeze(0), float("-inf"))
            aux_vals, aux_idx = torch.topk(pre_acts_idle, aux_k, dim=-1)
            aux_vals = torch.relu(aux_vals)
            aux_acts = torch.zeros_like(pre_acts).scatter_(-1, aux_idx, aux_vals)
            aux_recon = aux_acts @ self.sae.W_dec  # no b_dec -- reconstructing a residual, not raw input
            residual = (batch - recon).detach()
            aux_loss = ((aux_recon - residual) ** 2).mean()
            total_loss = loss + C.AUX_LOSS_COEF * aux_loss
        else:
            total_loss = loss
        self.opt.zero_grad()
        total_loss.backward()
        with torch.no_grad():
            # Remove the component of the decoder gradient parallel to each
            # row's current (unit-norm) direction, so the optimizer step is
            # already tangent to the unit-norm constraint. The naive version
            # tried earlier (renormalize AFTER opt.step()) let Adam's momentum
            if self.sae.W_dec.grad is not None:
                dec = self.sae.W_dec.data
                dec_unit = dec / dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                grad = self.sae.W_dec.grad
                parallel = (grad * dec_unit).sum(dim=-1, keepdim=True) * dec_unit
                self.sae.W_dec.grad = grad - parallel
        self.opt.step()
        with torch.no_grad():
            # Project decoder rows back to unit norm after the optimizer update.
            self.sae.W_dec.data /= self.sae.W_dec.data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        fired = (acts.detach() > 0).any(dim=0)
        self.steps_since_fired += 1
        self.steps_since_fired[fired] = 0
        self.step += 1
        resampled = 0
        if self.step % C.RESAMPLE_EVERY_STEPS == 0:
            resampled = self.resample_dead()
        if self.step % C.CHECKPOINT_EVERY_STEPS == 0:
            self.save_checkpoint()
        return loss.item(), (acts.detach() > 0).float().sum(-1).mean().item(), resampled

    def save_checkpoint(self):
        os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
        # Keep one atomically replaced checkpoint per layer.
        path = f"{C.CHECKPOINT_DIR}/layer{self.layer_idx}_latest.pt"
        tmp_path = path + ".tmp"
        torch.save({
            "sae_state": self.sae.state_dict(),
            "opt_state": self.opt.state_dict(),
            "step": self.step,
            "steps_since_fired": self.steps_since_fired.cpu(),
            "mean": self.mean, "std": self.std,
            "config": {k: v for k, v in vars(C).items() if not k.startswith("_")},
        }, tmp_path)
        os.replace(tmp_path, path)  # atomic -- never leaves a half-written "latest" file


# ---------------------------------------------------------------------------
# Bootstrap: collect a first buffer-full per layer, then train
# ---------------------------------------------------------------------------
wandb.init(project=C.WANDB_PROJECT, config={k: v for k, v in vars(C).items() if not k.startswith("_")})

print("Bootstrapping initial activation buffers...")
seed_acts = {l: [] for l in C.TARGET_LAYERS}
tokens_seen = 0
with torch.no_grad():
    while tokens_seen < C.SHUFFLE_BUFFER_TOKENS:
        batch = next(token_stream).to(device)
        model(batch)
        for l in C.TARGET_LAYERS:
            acts = captured[l]
            if C.EXCLUDE_LOCAL_POSITION_0:
                acts = acts[:, 1:, :]
            seed_acts[l].append(acts.reshape(-1, C.D_MODEL).float().cpu())
        tokens_seen += batch.numel()
        print(f"  bootstrap: {tokens_seen:,}/{C.SHUFFLE_BUFFER_TOKENS:,} tokens", flush=True)

trainers = {
    l: LayerSAETrainer(
        l, torch.cat(seed_acts[l], dim=0)[: C.SHUFFLE_BUFFER_TOKENS],
        resume_path=find_latest_checkpoint(l) if args.resume else None,
    )
    for l in C.TARGET_LAYERS
}
print(f"Bootstrapped {tokens_seen} tokens across {len(C.TARGET_LAYERS)} layers. Starting training.\n")

# ---------------------------------------------------------------------------
# Main loop: interleave data collection (refill) and SAE training (drain)
# ---------------------------------------------------------------------------
start_time = time.time()
resumed_step = min(t.step for t in trainers.values())
if resumed_step > 0:
    # Keep total_tokens consistent with the pacing formula at the resumed step,
    # rather than restarting the pacing count from the bootstrap tokens alone.
    total_tokens = int(resumed_step * (C.TOKEN_BUDGET / C.TOTAL_SAE_STEPS))
    print(f"Resuming from step {resumed_step}; total_tokens set to {total_tokens:,} to match pacing.\n")
else:
    total_tokens = tokens_seen
while min(t.step for t in trainers.values()) < C.TOTAL_SAE_STEPS:
    # Train a few steps on current buffers
    for _ in range(50):
        for l, trainer in trainers.items():
            loss, l0, resampled = trainer.train_step()
            if trainer.step % C.LOG_EVERY_STEPS == 0:
                dead_now = (trainer.steps_since_fired >= C.DEAD_THRESHOLD_STEPS).sum().item()
                wandb.log({
                    f"layer{l}/loss": loss, f"layer{l}/l0": l0,
                    f"layer{l}/dead_features": dead_now, f"layer{l}/resampled": resampled,
                    f"layer{l}/step": trainer.step,
                })
                elapsed = time.time() - start_time
                print(f"[{elapsed/60:5.1f}min] layer{l} step {trainer.step:6d}/{C.TOTAL_SAE_STEPS} "
                      f"loss={loss:.4f} L0={l0:.1f} dead={dead_now}")

    # Refill follows base-input-token pacing, not unique activation consumption.
    current_step = min(t.step for t in trainers.values())
    target_tokens_so_far = current_step * (C.TOKEN_BUDGET / C.TOTAL_SAE_STEPS)
    while total_tokens < target_tokens_so_far:
        with torch.no_grad():
            batch = next(token_stream).to(device)
            model(batch)
            for l, trainer in trainers.items():
                acts = captured[l]
                if C.EXCLUDE_LOCAL_POSITION_0:
                    acts = acts[:, 1:, :]
                trainer.add_activations(acts.reshape(-1, C.D_MODEL).float())
            total_tokens += batch.numel()

for h in handles:
    h.remove()
for l, trainer in trainers.items():
    trainer.save_checkpoint()
print(f"\nDone. Total tokens processed: {total_tokens}. Total time: {(time.time()-start_time)/60:.1f} min.")
