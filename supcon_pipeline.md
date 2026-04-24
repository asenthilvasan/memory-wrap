# SupCon / SimCLR / Hybrid Pretraining Pipeline

A focused walkthrough of the code I added on top of the base Memory Wrap
repository, written so you can delete everything and rebuild it from scratch
with full understanding.

If you want the architectural walkthrough of Memory Wrap itself (the base
paper's model), see `memory_wrap_walkthrough.md`. This document is
complementary: it only covers the **contrastive pretraining additions**.

---

## 1. What the pipeline does

Original Memory Wrap trains the encoder end-to-end with cross-entropy on
the downstream classification task. The sparse-attention retrieval geometry
(cosine similarity between encoder outputs) is whatever CE happens to
produce as a byproduct.

This pipeline adds a **pretraining stage** that shapes the encoder's feature
geometry explicitly before Memory Wrap training begins:

```
┌────────────────────────────┐     ┌─────────────────────────────┐
│  Stage A: PRETRAINING      │     │  Stage B: DOWNSTREAM        │
│  (new — paper/pretrain_... │     │  (existing — paper/train.py │
│   supcon.py)               │     │   with new flags)           │
│                            │     │                             │
│  Shape encoder features    │  →  │  Freeze encoder, train      │
│  with contrastive loss:    │     │  only the Memory Wrap head  │
│    supcon / simclr / hybrid│     │  (linear-probe style)       │
└────────────────────────────┘     └─────────────────────────────┘
       ↓                                    ↓
  forward_encoder weights            final classifier
  (saved checkpoint)                 (saved checkpoint)
```

Three contrastive objectives are supported:

| Objective | Uses labels? | Positives | Retrieval bias |
|---|---|---|---|
| **SupCon** | Yes | All same-class feature pairs | Same-class memories |
| **SimCLR** | No | Only other augmented view of same image | Visually similar memories |
| **Hybrid** | Yes | α·supcon + (1-α)·simclr | Both: same-class AND visually similar |

**Pilot result on SVHN (15 runs each, 2000 training examples):**

| Variant | Val acc (mean ± std) | vs scratch |
|---|---|---|
| Scratch (no pretraining) | 80.9% ± 1.6% | baseline |
| + SimCLR | 86.2% ± 0.2% | +5.3pp |
| + Hybrid | 91.4% ± 0.2% | +10.5pp |
| **+ SupCon** | **93.8% ± 0.1%** | **+12.9pp** |

SupCon dominates on this task. Variance collapse under frozen-pretrained is
also striking: std drops from 1.6% (scratch) to 0.1% (SupCon).

---

## 2. Files added / modified

### Added

| File | Lines | Purpose |
|---|---|---|
| `paper/pretrain_supcon.py` | ~330 | The pretraining script. Self-contained: CLI, 2-view augmentation, contrastive loss, training loop, checkpoint save. |

### Modified

| File | Lines changed | What changed |
|---|---|---|
| `paper/train.py` | ~15 | Two new flags: `--pretrained_encoder`, `--freeze_encoder`. Checkpoint-loading logic. Path suffix to keep per-variant runs separate. Per-epoch flushed prints. |
| `paper/config/train.yaml` | 0 | No structural changes. You only edit this to switch `dataset_name: SVHN` ↔ `CIFAR10`. |
| `deployment.yml` | +7 | Kubernetes `/dev/shm` tmpfs volume (cluster-specific, skip if not using the lab's k8s cluster). |

That's it. The entire addition is a single new script plus ~15 lines touched
in `train.py`. Everything else — the Memory Wrap core (`memory.py`), the
backbones (`paper/architectures/`), the data loaders, the eval scripts — is
untouched.

---

## 3. `paper/pretrain_supcon.py` — design walkthrough

The script is intentionally self-contained (~330 lines). Imports only:
`torch`, `torchvision`, `absl` (for CLI, matching `train.py`'s convention),
and the existing `utils.utils.get_model` factory.

### 3a. CLI flags (`paper/pretrain_supcon.py:54-91`)

```
--model           mobilenet    # any backbone name from utils.get_model
--dataset         CIFAR10      # or SVHN
--loss            supcon       # supcon / simclr / hybrid
--temperature     0.07         # softmax temperature (0.07 supcon, 0.5 simclr)
--hybrid_alpha    0.5          # mix weight for hybrid (only used if loss=hybrid)
--epochs          100
--batch_size      256
--lr              0.5          # SGD learning rate
--num_workers     8            # DataLoader workers
--data_dir        datasets
```

Why these defaults?

- **`batch_size=256`** — SupCon benefits from large batches because more
  samples means more negatives per anchor, sharpening the contrastive
  signal. 256 is a single-24GB-GPU compromise.
- **`lr=0.5`** — Large LR is standard for contrastive pretraining (the
  cosine schedule anneals it to 0 smoothly).
- **`temperature=0.07`** — SupCon default. For SimCLR, the original paper
  uses 0.5, so pass `--temperature=0.5` when `--loss=simclr`.
- **`epochs=100`** — SupCon paper's CIFAR-10 recipe. More helps slightly.

### 3b. Dataset specs (`paper/pretrain_supcon.py:98-113`)

```python
DATASET_SPECS = {
    'CIFAR10': { ..., 'hflip': True  },   # cats/planes ~symmetric
    'SVHN':    { ..., 'hflip': False },   # a flipped '3' is not a '3'
}
```

Two things per dataset:
1. **Normalization stats** — copied from `paper/utils/datasets.py` so
   downstream Memory Wrap sees features on the same numerical scale.
2. **Whether horizontal flip is identity-preserving** — critical for SVHN
   because digits aren't symmetric; flipping destroys the label.

### 3c. The contrastive loss (`paper/pretrain_supcon.py:116-185`)

```python
def contrastive_loss(features, labels, temp):
    # features: [2B, d], L2-normalized
    # labels:   [B] for SupCon, None for SimCLR

    if labels is None:
        # SimCLR: positive for anchor i is the OTHER view of same image.
        # First B rows = view-1, next B = view-2 of same images in order.
        # Positive mask is identity rolled by B columns.
        mask = torch.eye(2*B, device=...).roll(B, dims=1)
    else:
        # SupCon: duplicate labels (views share class), then same-label
        # pairs are positives. Zero diagonal (anchor is never its own pos).
        labels = torch.cat([labels, labels])
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)

    # Cosine similarity matrix (since features are L2-normalized).
    logits = features @ features.T / temp

    # Numerical stability: subtract per-row max.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Log-softmax over all non-self columns.
    not_self = 1 - torch.eye(2*B, device=...)
    log_prob = logits - torch.log((logits.exp() * not_self).sum(dim=1, keepdim=True) + 1e-12)

    # For each anchor: average log_prob over its positives; negate.
    return -(mask * log_prob).sum(dim=1).div(mask.sum(dim=1).clamp(min=1)).mean()
```

**Key insight: SupCon and SimCLR share the same loss structure.** The only
difference is which pairs count as positives. One function handles both
objectives via the `labels` argument.

For **hybrid**, we call the same function twice (once with labels, once
without) and take a weighted sum (`paper/pretrain_supcon.py:295-298`):

```python
l_sup = contrastive_loss(feat, y,    temp)
l_sim = contrastive_loss(feat, None, temp)
loss  = alpha * l_sup + (1 - alpha) * l_sim
```

### 3d. The TwoViews wrapper (`paper/pretrain_supcon.py:188-198`)

```python
class TwoViews:
    def __init__(self, t): self.t = t
    def __call__(self, x): return (self.t(x), self.t(x))
```

Dead-simple: call the stochastic transform pipeline twice on the same
image, return a tuple. The DataLoader then yields batches shaped as
`((view1_batch, view2_batch), label_batch)`.

This is the **only** reason we get a known positive pair for the
contrastive loss — the two views of the same image are guaranteed to
share a class and a common source, so they're the "anchor" of the loss.

### 3e. Augmentations (`paper/pretrain_supcon.py:215-233`)

SimCLR-style pipeline:

```
RandomResizedCrop(32, scale=(0.2, 1.0))   # forces spatial invariance
RandomHorizontalFlip()                    # only for CIFAR10, not SVHN
RandomApply(ColorJitter, p=0.8)           # strong color perturbation
RandomGrayscale(p=0.2)                    # anti-color-shortcut
ToTensor + Normalize                      # match downstream normalization
```

Critical design choice: the augmentations must be **strong enough** that
two views look meaningfully different, but **not so strong** that
class-identifying content is destroyed. The above is the standard SimCLR
recipe that survives ~3 years of ablations in the self-supervised
literature.

### 3f. Training loop (`paper/pretrain_supcon.py:273-313`)

```python
for ep in range(1, epochs+1):
    for (v1, v2), y in loader:
        # Concatenate views into a [2B, 3, 32, 32] batch.
        # IMPORTANT: same forward pass so BatchNorm stats are consistent
        # across views.
        imgs = torch.cat([v1, v2]).to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # forward_encoder returns [2B, d] raw features.
            # F.normalize projects onto unit hypersphere → cosine sim = matmul.
            feat = F.normalize(model.forward_encoder(imgs), dim=1)

            if loss == 'supcon':  l = contrastive_loss(feat, y,    temp)
            if loss == 'simclr':  l = contrastive_loss(feat, None, temp)
            if loss == 'hybrid':  l = alpha*...(y, ...) + (1-alpha)*...(None, ...)

        scaler.scale(l).backward()
        scaler.step(opt)
        scaler.update()
    sched.step()
    print(f'Epoch {ep}/{epochs}  loss={l.item():.4f}', flush=True)
```

Three things worth internalizing:

1. **Concatenating both views into one forward pass** is non-negotiable.
   If you pass them separately, BatchNorm computes different running
   means/stds per view and the contrastive signal degrades. One batch,
   one forward, one BN update.

2. **`F.normalize(features, dim=1)`** is what makes `features @ features.T`
   equivalent to pairwise cosine similarity. Without L2-normalization the
   loss degenerates.

3. **AMP (autocast + GradScaler)** is here for ~2x speedup on modern GPUs.
   If you cut it for simplicity, you lose ~50% throughput on Ampere+ cards.

### 3g. Checkpoint format (`paper/pretrain_supcon.py:322-327`)

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'model_name':       FLAGS.model,          # so train.py can rebuild
    'num_classes':      10,
    'modality':         f'{FLAGS.loss}_pretrained',
    'dataset_name':     FLAGS.dataset,
}, f'models/{dataset}/{loss}/{model}/1.pt')
```

The format matches what `train.py` saves, so downstream tools
(`generate_memory_images.py`, eval scripts) can load it and reconstruct
the architecture via `utils.get_model`.

**Path convention** `models/<dataset>/<loss>/<model>/1.pt` — keeps each
variant in its own directory so nothing clobbers.

---

## 4. `paper/train.py` — changes walkthrough

Only ~15 lines added. The core training loop is unchanged.

### 4a. Two new flags (`paper/train.py:17-22`)

```python
absl.flags.DEFINE_string("pretrained_encoder", None,
    "Optional path to a SupCon-pretrained encoder checkpoint ...")
absl.flags.DEFINE_bool("freeze_encoder", False,
    "If True, freeze all parameters except the Memory Wrap head ...")
```

### 4b. Checkpoint loading + freezing (`paper/train.py:198-204`)

Inside the per-run loop, right after instantiating a fresh model:

```python
if FLAGS.pretrained_encoder:
    ckpt = torch.load(FLAGS.pretrained_encoder, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
if FLAGS.freeze_encoder:
    for n, p in model.named_parameters():
        if not n.startswith('mw.'): p.requires_grad_(False)
```

Two design choices:

1. **`strict=False`** — the pretraining checkpoint has an uninitialized
   `self.mw` head (Memory Wrap isn't used during pretraining). When we
   load it into a freshly constructed model with a freshly-initialized
   head, `strict=False` allows the mismatched keys to be silently
   ignored.

2. **Freeze by prefix `'mw.'`** — Memory Wrap lives as `self.mw` on every
   backbone (see `paper/architectures/mobilenet.py:102, 157` etc.). So
   "everything except the head" is "every parameter whose name doesn't
   start with `'mw.'`". Simple, consistent across all backbones.

### 4c. Optimizer on only unfrozen params (`paper/train.py:207`)

```python
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                            **dict_optim)
```

When the encoder is frozen, only `mw.*` params receive updates. The
optimizer must only see those — passing frozen params is wasteful and
sometimes triggers warnings.

### 4d. Per-variant save path (`paper/train.py:164-172`)

```python
suffix = ''
if FLAGS.pretrained_encoder:
    suffix = '_pretrained'
    for tag in ('supcon', 'simclr', 'hybrid'):
        if f'/{tag}/' in FLAGS.pretrained_encoder:
            suffix = f'_{tag}'
            break
modality_dir = FLAGS.modality + suffix
path_saving_model = f'models/{dataset_name}/{modality_dir}/{config["model"]}/{train_examples}/'
```

Infers which pretraining variant produced the encoder from the path
(e.g. `.../supcon/...` → `_supcon`). This keeps the four downstream runs
(scratch, +supcon, +simclr, +hybrid) in separate `models/` subdirs so
you can run them in parallel without collision.

**Resulting paths:**
```
models/SVHN/encoder_memory/mobilenet/2000/{1,...,15}.pt           # scratch baseline
models/SVHN/encoder_memory_supcon/mobilenet/2000/{1,...,15}.pt   # + SupCon
models/SVHN/encoder_memory_simclr/mobilenet/2000/{1,...,15}.pt   # + SimCLR
models/SVHN/encoder_memory_hybrid/mobilenet/2000/{1,...,15}.pt   # + Hybrid
```

### 4e. Flushed epoch prints (`paper/train.py:83, 132, 253`)

```python
print(f'[memory] Epoch {epoch}/{num_epochs}  loss={loss.item():.4f}', flush=True)
```

Without `flush=True`, Python block-buffers stdout when redirected to a
file, and 30-char epoch prints won't flush until the process exits
(takes 100+ epochs to fill the 4KB buffer). This is the single most
important operational-ergonomics fix when running jobs in parallel with
output redirection.

---

## 5. End-to-end workflow

### Step 1: Pretrain the encoder (3 variants)

```bash
cd paper/

# On a multi-GPU pod, run all three in parallel:
CUDA_VISIBLE_DEVICES=0 python pretrain_supcon.py --dataset=SVHN --loss=supcon                       &
CUDA_VISIBLE_DEVICES=1 python pretrain_supcon.py --dataset=SVHN --loss=simclr --temperature=0.5     &
CUDA_VISIBLE_DEVICES=2 python pretrain_supcon.py --dataset=SVHN --loss=hybrid                       &
wait
```

Output checkpoints:
```
models/SVHN/supcon/mobilenet/1.pt
models/SVHN/simclr/mobilenet/1.pt
models/SVHN/hybrid/mobilenet/1.pt
```

### Step 2: Train Memory Wrap on top of each (frozen)

```bash
# From-scratch baseline (no pretraining)
CUDA_VISIBLE_DEVICES=0 python train.py --modality=encoder_memory &

# Three frozen-encoder variants
CUDA_VISIBLE_DEVICES=1 python train.py --modality=encoder_memory \
    --pretrained_encoder=models/SVHN/supcon/mobilenet/1.pt  --freeze_encoder=True &
CUDA_VISIBLE_DEVICES=2 python train.py --modality=encoder_memory \
    --pretrained_encoder=models/SVHN/simclr/mobilenet/1.pt  --freeze_encoder=True &
CUDA_VISIBLE_DEVICES=3 python train.py --modality=encoder_memory \
    --pretrained_encoder=models/SVHN/hybrid/mobilenet/1.pt  --freeze_encoder=True &
wait
```

Output checkpoints (per `config['runs']`, default 15):
```
models/SVHN/encoder_memory/mobilenet/2000/{1..15}.pt            # scratch
models/SVHN/encoder_memory_supcon/mobilenet/2000/{1..15}.pt    # + SupCon
models/SVHN/encoder_memory_simclr/mobilenet/2000/{1..15}.pt    # + SimCLR
models/SVHN/encoder_memory_hybrid/mobilenet/2000/{1..15}.pt    # + Hybrid
```

### Step 3: Extract accuracies

```bash
for f in log_down_*.txt; do
    echo "=== $f ==="
    grep "Run:" "$f" | tail -1     # final Run:N line has mean accuracy
done
```

### Step 4: Generate memory retrieval images

`scripts/generate_memory_images.py` uses `checkpoint['modality']` to pick
the save directory, but all four runs share `modality='encoder_memory'`
— you must rename the output between runs:

```bash
cd paper/scripts/
for tag in '' _supcon _simclr _hybrid; do
    name=${tag:-_scratch}
    name=${name#_}

    python generate_memory_images.py \
        --path_model=../models/SVHN/encoder_memory${tag}/mobilenet/2000/1.pt

    mv ../images/mem_images/SVHN/encoder_memory/mobilenet \
       ../images/mem_images/SVHN/encoder_memory_${name}_out
done
```

---

## 6. Design decisions worth preserving

### Why one script, not three?

SupCon and SimCLR share ~95% of the code path. Keeping them in one file
with a `--loss` flag and a single `contrastive_loss` function:

- Eliminates duplicated augmentation / optimizer / checkpoint code.
- Makes the "hybrid" variant trivial (same function, two calls).
- Forces consistency: you can't accidentally change temperature for
  SupCon but forget to for SimCLR.

### Why `utils.get_model(..., model_type='encoder_memory')` and not `'std'`?

We want to pretrain `forward_encoder` so its output feeds Memory Wrap
downstream. The simplest way to ensure downstream state-dict compatibility
is to use the same architecture class — `EncoderMemory*` — during
pretraining. The `self.mw` head is instantiated but never called, so its
random-init weights persist into the saved checkpoint. Downstream
`train.py` loads with `strict=False`, re-initializing `self.mw` for
fine-tuning.

Alternative designs considered:
- Use the `std` variant and write custom state-dict remapping. More
  code, more failure modes.
- Expose `forward_encoder` as a standalone `nn.Module`. Requires
  modifying every backbone.

### Why `persistent_workers=True` and `prefetch_factor=4`?

The 2-view augmentation is CPU-bound. Without these, the GPU starves
between epochs (workers tear down and respawn). Measured impact:
~15-20% faster epochs on a 64-vCPU pod.

### Why L2-normalize before computing similarity?

`F.normalize(features, dim=1)` projects every feature vector to the unit
hypersphere. Then `features @ features.T` is the full pairwise cosine
similarity matrix in a single matmul. Without normalization, you'd need
to divide by vector norms element-wise (slower, less numerically stable).

### Why `mask.sum(dim=1).clamp(min=1)`?

Edge case: an anchor in a minibatch might have zero positives (rare, but
possible with balanced sampling). Without the clamp, you'd divide by 0
and get NaN. With the clamp, the numerator `(mask * log_prob).sum(dim=1)`
is also 0 for such anchors, so their contribution to the batch mean is
0, not NaN.

### Why `--freeze_encoder=True` in downstream training?

Linear-probe style evaluation is the **canonical test** for how good a
learned representation is. Fine-tuning the encoder during downstream
training makes it impossible to tell whether any accuracy gain comes
from the pretraining or from extra backprop on the downstream task.
Freezing isolates the pretraining's contribution cleanly.

For a production system you'd probably unfreeze after a few epochs of
linear probing (warm-start fine-tuning). But for the experimental
question "does pretraining learn useful features?", freezing is the
right protocol.

---

## 7. Rebuilding from scratch

If you're going to delete this and rewrite, the minimum viable pipeline is:

1. **One file (`pretrain.py`):**
   - CLI flags for dataset, loss, temperature, epochs, batch size, lr
   - Dataset spec dict (normalization, hflip per-dataset)
   - `TwoViews` class (~3 lines)
   - `contrastive_loss(features, labels, temp)` (~30 lines with comments)
   - SimCLR-style augmentation pipeline
   - `get_model(..., model_type='encoder_memory')` → `forward_encoder` → loss
   - SGD + cosine LR + AMP
   - Save `state_dict` + metadata to path including dataset/loss

2. **Two flags added to existing `train.py`:**
   - `--pretrained_encoder=<path>`: `load_state_dict(..., strict=False)`
   - `--freeze_encoder`: loop over named_parameters, freeze non-`mw.*`
   - Save path suffix logic so variants don't clobber

3. **Use existing everything else:**
   - `memory.py` (Memory Wrap core)
   - `paper/architectures/*.py` (backbones)
   - `paper/utils/utils.py` `get_model`, `get_loaders`, `eval_memory`

That's it. The entire pretraining + variant-comparison pipeline is **one
new file and ~15 added lines in `train.py`**.

---

## 8. Pitfalls & gotchas (learned the hard way)

### `/dev/shm` too small in Kubernetes

Default k8s pods give you 64MB of `/dev/shm`. PyTorch DataLoader workers
use shared memory for tensor sharing between processes. 8 workers × 2-view
augmentation × 256 batch overflows this instantly.

Fix: either add `torch.multiprocessing.set_sharing_strategy('file_system')`
at the top of the script (done at `paper/pretrain_supcon.py:41`) or mount
an `emptyDir` tmpfs over `/dev/shm` in the pod spec (done in
`deployment.yml`).

### Buffered stdout when redirected

`python script.py > log.txt` → Python block-buffers stdout to 4KB. Epoch
prints are ~30 chars. 100 epochs of prints never fill the buffer → you
see nothing in the log for the entire run.

Fix: `print(..., flush=True)` explicitly (done in `pretrain_supcon.py`
and `train.py`). Alternative: `python -u script.py` for unbuffered mode.

### `modality` in checkpoint controls image output dir

`scripts/generate_memory_images.py` computes the output dir from
`checkpoint['modality']`. All four downstream variants save as
`modality='encoder_memory'`, so their retrieval images overwrite each
other. Must rename the output dir between runs (or modify
`generate_memory_images.py` to take an explicit `--output_suffix` flag).

### `runs: 15` in `config/train.yaml`

Each `python train.py` invocation runs 15 complete training+eval cycles
for statistical robustness. At 40 epochs/run on SVHN, that's 600 epochs
per invocation. For a quick pilot, reduce to 3-5:

```yaml
# paper/config/train.yaml
runs: 5
```

### SVHN has no horizontal-flip augmentation

A flipped '3' is not a '3'. The `hflip=False` in `DATASET_SPECS['SVHN']`
is load-bearing — if you accidentally enable hflip on SVHN, the encoder
learns features that blur distinct digits.

### SupCon needs large batches

Contrastive loss quality scales with number of negatives. At `batch=64`
you have ~128 possible negatives per anchor; at `batch=256` you have ~512.
Cutting batch size below 128 significantly degrades the learned
representation. If you're VRAM-constrained, either use gradient
accumulation or a smaller backbone.

---

## 9. Extending the pipeline

### Adding a new contrastive objective

Both SupCon and SimCLR are instances of the softmax-over-similarities
family. To add a new one (e.g. NT-Xent with hard negative mining,
triplet loss, DINO-style self-distillation):

1. Add a new branch to the `contrastive_loss` function (or sibling
   function) in `paper/pretrain_supcon.py`.
2. Add the objective name to the `--loss` enum flag.
3. Dispatch it in the training loop.

### Adding a new backbone

`pretrain_supcon.py` uses `utils.get_model` for the model factory.
Any backbone that has a `forward_encoder` method returning `[B, d]` will
work with zero pipeline changes. See `memory_wrap_walkthrough.md`
section "Swapping in your own encoder" for the contract.

### Adding a new dataset

Add an entry to `DATASET_SPECS` with:
- `cls`: the torchvision dataset class
- `split_kwargs`: kwargs for the train split (e.g. `{'train': True}` or
  `{'split': 'train'}`)
- `mean`, `std`: per-channel normalization (must match what the
  downstream data loader in `paper/utils/datasets.py` uses)
- `hflip`: whether horizontal flip preserves class identity

---

## 10. File index

| File | Role |
|---|---|
| `paper/pretrain_supcon.py` | **New.** Self-contained contrastive pretraining script. ~330 lines. |
| `paper/train.py` | **Modified.** Added `--pretrained_encoder`, `--freeze_encoder` flags + checkpoint loading + save-path suffix logic. ~15 lines changed. |
| `paper/config/train.yaml` | **Unchanged structurally.** Only edit `dataset_name` to switch datasets, `runs` to control statistical robustness. |
| `memory.py` | **Unchanged.** Core Memory Wrap (base paper). |
| `paper/architectures/*.py` | **Unchanged.** Backbones with the three-variant pattern (std, memory, encoder_memory). |
| `paper/utils/*.py` | **Unchanged.** Model factory, data loaders, eval. |
| `paper/scripts/generate_memory_images.py` | **Unchanged.** Generates visual retrieval comparisons (but has a gotcha — see §8). |
| `memory_wrap_walkthrough.md` | Architectural walkthrough of Memory Wrap (base paper). |
| `supcon_pipeline.md` | **This file.** The pretraining pipeline added on top. |
| `deployment.yml` | Kubernetes deployment (lab-specific). |
