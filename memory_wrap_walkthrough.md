# Memory Wrap: Code ↔ Diagram ↔ Paper Equations

A walkthrough connecting the architecture diagram and Section 3.1 equations from
La Rosa et al., *"A self-interpretable module for deep image classification on
small data"* (Applied Intelligence, 2022), to the actual PyTorch implementation
in this repository.

The entire Memory Wrap module lives in a single ~160-line file, and the diagram
maps almost 1-to-1 onto it. The **Encoder** block and the block showing **MLP**
+ **Content Attention** are split across two files: the encoder lives inside
each backbone under `paper/architectures/`, while everything inside the dashed
"Memory Wrap" box lives in `memory.py`.

---

## Core file: `memory.py`

The single forward pass that implements the diagram is:

```python
# memory.py, lines 91–120 (MemoryWrapLayer.forward)
def forward(self, encoder_output, memory_set, return_weights=False):
    # compute content weights
    dist = _distance(encoder_output, memory_set, self.distance_name)
    content_weights = sparsemax(-dist, dim=1)

    # compute memory vector
    memory_vector = torch.matmul(content_weights, memory_set)

    # classification
    final_input = torch.cat([encoder_output, memory_vector], 1)
    output = self.classifier(final_input)

    if return_weights:
        return output, content_weights
    else:
        return output
```

---

## Mapping to the diagram

| Diagram block | Where it lives | What it does |
|---|---|---|
| **Input** (`x_i`) | `data` in `paper/train.py:49–54` — a mini-batch from `train_loader` | Current image(s) to classify. |
| **Memory Set** (`S_i`) | `memory_input` from a *second* loader at `paper/train.py:55–56`; loader built in `paper/datasets.py:139` (`mem_loader`) | A random batch of training samples drawn fresh every step — the "memories of past training samples". |
| **Encoder** (shared, pink block) | `forward_encoder()` in each backbone, e.g. `paper/architectures/mobilenet.py:113–123` | The usual CNN minus its last linear classifier. The **same `forward_encoder` is called twice** at `paper/architectures/mobilenet.py:125–133` — once on the input, once on the memory set — which is why the diagram shows a single encoder with two arrows. |
| **Encoding Input** (`e_i = f(x_i)`) | `out = self.forward_encoder(x)` at `paper/architectures/mobilenet.py:128` | Feature vector `[b, encoder_output_dim]`. Passed as `encoder_output`. |
| **Encoding Memory Set** (`E_{S_i}`) | `out_ss = self.forward_encoder(ss)` at `paper/architectures/mobilenet.py:129` | Feature matrix `[m, encoder_output_dim]`. Passed as `memory_set`. |
| **Content Attention** | `memory.py:107–108` | Cosine distance input↔each memory, then `sparsemax(-dist)` → sparse weights `w_j`. |
| **Memory Vector** (`v_{S_i}`) | `memory.py:111` | Weighted sum of memory encodings with the sparse weights. |
| **⊕ (concat)** | `memory.py:114` | `torch.cat([encoder_output, memory_vector], 1)` — despite being drawn as ⊕ in the figure, this is a **concatenation**, not a sum (the paper states this explicitly in Eq. 4). |
| **MLP** | `memory.py:53–69` (class `MLP`), instantiated as the `classifier` at `memory.py:87` | Two-layer perceptron `[2·d → 4·d → num_classes]` applied to the concatenation. Corresponds to the paper's last layer `l_f`. |
| **Output** | `memory.py:115` (`output`) | Class logits. Cross-entropy loss applied in `paper/train.py:60–61`. |

---

## Mapping to the paper's Section 3.1 equations

Symbols follow the paper: `f` = encoder, `x_i` = input, `S_i` = memory set,
`w_j` = content weight, `v_{S_i}` = memory vector, `l_f` = final layer.

### Eq. 1 — Encoding

$$
e_i = f(x_i), \qquad
E_{S_i} = \{\, f(x^i_{m_1}),\ \ldots,\ f(x^i_{m_n}) \,\}
$$

**Code:** `paper/architectures/mobilenet.py:125–133`

```python
def forward(self, x, ss, return_weights=False):
    # input
    out = self.forward_encoder(x)
    out_ss = self.forward_encoder(ss)

    # prediction
    out_mw = self.mw(out, out_ss, return_weights)
    return out_mw
```

The same `forward_encoder` is used for both input and memory set — the encoder
is **shared**, matching the single "Encoder" block in the diagram.

### Eq. 2 — Sparse content attention

$$
w_j = \text{sparsemax}\!\left(\cos\!\big(e_i,\ f(x^i_{m_j})\big)\right)
$$

**Code:** `memory.py:107–108`

```python
dist = _distance(encoder_output, memory_set, self.distance_name)   # cosine distance
content_weights = sparsemax(-dist, dim=1)                          # sparsemax on similarity
```

Two details worth noting:

- `_distance` with `'cosine'` (default) returns `1 − cos(x, y)` — a **distance**,
  so the code feeds `−dist` into `sparsemax` to recover a similarity (the
  argument of sparsemax in the paper). See `memory.py:35–38`.
- `sparsemax` comes from the `entmax` package (`memory.py:3`). The paper
  explicitly cites Martins & Astudillo's sparsemax [22] and stresses that the
  *sparsity* is what makes the model interpretable: most `w_j` are exactly zero.

### Eq. 3 — Memory vector

$$
v_{S_i} = \sum_{j=1}^{n} w_j \cdot f(x^i_{m_j})
$$

**Code:** `memory.py:111`

```python
memory_vector = torch.matmul(content_weights, memory_set)
```

A single matrix multiply `[b, m] × [m, d] → [b, d]` *is* the weighted sum over
the `m` memory encodings.

### Eq. 4 — Final output

$$
g(x_i) = l_f\!\big([\,e_i\,;\,v_{S_i}\,]\big)
$$

**Code:** `memory.py:114–115`

```python
final_input = torch.cat([encoder_output, memory_vector], 1)
output = self.classifier(final_input)
```

The classifier `l_f` is the MLP built at `memory.py:87`:

```python
self.classifier = classifier or MLP(encoder_output_dim*2, encoder_output_dim*4, output_dim)
```

Input dim is `2·d` because of the concatenation, matching Eq. 4.

---

## Two variants in the codebase

`memory.py` actually defines **two** classes. The diagram depicts the first one.

- **`MemoryWrapLayer`** (`memory.py:72–120`) — the Memory Wrap described in the
  paper and the diagram: concatenates `encoder_output` with `memory_vector` and
  runs the MLP on both.
- **`BaselineMemory`** (`memory.py:124–162`) — an ablation baseline used in the
  paper: it computes the same memory vector but **drops the input encoding**
  and runs the MLP on `memory_vector` only (`memory.py:157`). In diagram terms:
  remove the "Encoding Input → ⊕" arrow.

### ⚠️ Gotcha in the backbone files

The architecture files use swapped aliases that are easy to misread:

```python
# paper/architectures/mobilenet.py:9–10 (same pattern in every backbone)
from memorywrap import MemoryWrapLayer as EncoderMemoryWrapLayer
from memorywrap import BaselineMemory as MemoryWrapLayer
```

So inside the backbone files:

- `self.mw = MemoryWrapLayer(...)` is actually the **baseline**
  (`BaselineMemory`) — used by `MemoryMobileNetV2` at
  `paper/architectures/mobilenet.py:102`.
- `self.mw = EncoderMemoryWrapLayer(...)` is the **real Memory Wrap** from the
  diagram — used by `EncoderMemoryMobileNetV2` at
  `paper/architectures/mobilenet.py:157`.

The training script dispatches between them with the `--modality` flag
(`memory` = baseline, `encoder_memory` = Memory Wrap): `paper/train.py:184`.

---

## Quick mental model for future modifications

1. **Two independent `DataLoader`s:** one normal train loader, one "memory"
   loader over the same training set (`paper/datasets.py:133, 139`). Every
   training step samples a *fresh* random memory set.
2. **Encoder runs twice per step** — once on the input, once on the memory set
   — and both feature tensors are handed to `MemoryWrapLayer.forward`
   (`paper/architectures/mobilenet.py:128–132`).
3. **All of "Memory Wrap"** (the dashed box in the diagram) is the ~15 lines
   inside `MemoryWrapLayer.forward` in `memory.py:91–120`. If you want to
   change how Memory Wrap works, this is almost certainly the function you'll
   touch — plus `_distance` at `memory.py:20–50` if the similarity metric is
   changing, and the `classifier`/`MLP` at `memory.py:53–87` if the head is
   changing.
4. **Local file vs. PyPI package.** The installable `memorywrap` PyPI package
   is what the `paper/architectures/*.py` files import, *not* the local
   `memory.py`. The local `memory.py` is the reference source — if you modify
   it and want your changes picked up by the training scripts, either install
   it locally (e.g. edit the package in site-packages or `pip install -e .`
   from a local copy) or point the imports at `memory.py` directly.

---

## Backbone architectures in `paper/architectures/`

Each file (except `autoencoder.py`) is a **CNN backbone** from the
computer-vision literature, reimplemented for CIFAR-sized inputs (3×32×32).
Each backbone defines **three variants** of the same architecture:

1. **Standard variant** — the ordinary classifier with a final `nn.Linear`
   (baseline for comparison, used when `modality = 'std'`).
2. **`Memory*` variant** — uses `BaselineMemory` as the head (the ablation
   from the paper that *drops* the input encoding and predicts from the
   memory vector alone). Used when `modality = 'memory'`.
3. **`EncoderMemory*` variant** — uses `MemoryWrapLayer` as the head (the
   actual Memory Wrap from the diagram and Section 3.1). Used when
   `modality = 'encoder_memory'`.

The dispatcher that picks the right class is `get_model()` at
`paper/utils/utils.py:83–163`. The training script reads `config['model']`
from `paper/config/train.yaml` and passes it to `get_model`; the checkpoint
file stores this name so `generate_memory_images.py` can reconstruct the
correct class at inference time.

### File-by-file

| File | Architecture | Reference | Variants exposed |
|---|---|---|---|
| `paper/architectures/mobilenet.py` | **MobileNetV2** (CIFAR-adapted: first conv stride 2→1, pool kernel 7→4) | Sandler et al., 2018 | `MobileNetV2`, `MemoryMobileNetV2`, `EncoderMemoryMobileNetV2` |
| `paper/architectures/resnet.py` | **ResNet-18 / ResNet-34** (CIFAR-adapted) | He et al., 2015 | `ResNet18`, `ResNet34`, `MemoryResNet18`, `EncoderMemoryResNet18` |
| `paper/architectures/wide_resnet.py` | **Wide ResNet-28-10** | Zagoruyko & Komodakis, 2016 | `wrn28_10`, `memory_wrn28_10`, `encoder_wrn28_10` |
| `paper/architectures/densenet.py` | **DenseNet-BC** (cifar config) | Huang et al., 2016 | `densenet_cifar`, `memory_densenet_cifar`, `encoder_memory_densenet_cifar`, plus `densenet169` |
| `paper/architectures/efficientnet.py` | **EfficientNet-B0** | Tan & Le, 2019 | `EfficientNetB0`, `MemoryEfficientNetB0`, `EncoderMemoryEfficientNetB0` |
| `paper/architectures/googlenet.py` | **GoogLeNet / Inception-v1** | Szegedy et al., 2014 | `GoogLeNet`, `MemoryGoogLeNet`, `EncoderMemoryGoogLeNet` |
| `paper/architectures/shufflenet.py` | **ShuffleNetV2** (0.5× and 1× size) | Ma et al., 2018 | `ShuffleNetV2`, `MemoryShuffleNetV2`, `EncoderMemoryShuffleNetV2` |
| `paper/architectures/autoencoder.py` | **Tiny convolutional autoencoder** (encoder+decoder, no classifier) | in-house | `Encoder`, `Decoder`, `AutoEncoder` |

Everything except `autoencoder.py` is used as a drop-in encoder for the CNN
classification experiments in the paper. The **autoencoder** is a different
beast — it's not a classifier at all: it's used by the *uncertainty /
out-of-distribution detection* experiments in Appendix A.8, trained
separately by `paper/scripts/train_aes.svhn.py` and wrapped by the files in
`paper/scripts/wrappers/`.

### The common three-variant pattern

Using `mobilenet.py` as the canonical example:

- `MobileNetV2` at `paper/architectures/mobilenet.py:42–79` — standard model,
  ends in `self.linear = nn.Linear(1280, num_classes)`.
- `MemoryMobileNetV2` at `paper/architectures/mobilenet.py:81–133` — replaces
  that linear with `self.mw = MemoryWrapLayer(1280, num_classes)` (which, due
  to the alias swap at lines 9–10, is actually `BaselineMemory`). Adds a
  `forward_encoder(x)` helper and a `forward(x, ss)` that encodes both input
  and memory set and calls the head.
- `EncoderMemoryMobileNetV2` at `paper/architectures/mobilenet.py:136–157` —
  subclasses `MemoryMobileNetV2` and only overrides `__init__` to swap the
  head for `EncoderMemoryWrapLayer(1280, num_classes)` (= the real
  `MemoryWrapLayer` from the paper). Inherits `forward_encoder` and `forward`
  unchanged.

You'll see the exact same "plain / baseline-memory / real-memory-wrap"
triplet in every other backbone file, and all of them import Memory Wrap
with the same aliased imports:

```python
from memorywrap import MemoryWrapLayer as EncoderMemoryWrapLayer
from memorywrap import BaselineMemory as MemoryWrapLayer
```

### Architectures reported in the paper

Per the main tables and Appendix A.5 ("Additional Architectures") of La Rosa
et al., results are reported on: **MobileNetV2, ResNet-18, Wide ResNet-28-10,
DenseNet, EfficientNet-B0, GoogLeNet, and ShuffleNetV2**. MobileNetV2 is the
default in `paper/config/train.yaml:3` and the backbone shown in most of the
example figures (heatmaps, memory images) in the repo.

---

## Swapping in your own encoder

If you want to replace the pink "Encoder" block in the diagram with a custom
architecture (a different CNN, a ViT variant, a pretrained feature extractor,
an SSM, …), you do **not** need to touch `memory.py` or rewrite the other
backbones. Memory Wrap is encoder-agnostic — you just need to produce a
feature tensor with the right shape.

### The encoder contract

Memory Wrap only cares about three things:

1. **`forward_encoder(x: Tensor[b, 3, 32, 32]) -> Tensor[b, d]`** — maps a
   batch of CIFAR-10 images to a flat feature vector of dimension `d`. The
   spatial dims must be collapsed (global pool or flatten), because the
   sparse attention in `memory.py:107-108` operates on 1-D vectors.
2. **`self.mw = EncoderMemoryWrapLayer(d, num_classes)`** — construct the
   Memory Wrap head with the same `d` that `forward_encoder` produces. Any
   mismatch blows up at the `torch.matmul` on `memory.py:111`.
3. **`forward(x, ss, return_weights=False)`** — call `forward_encoder` on
   both the input and the memory set and pass both to `self.mw`. The memory
   set `ss` has shape `[m, 3, 32, 32]` and must go through the **exact same**
   encoder (weight-shared).

Nothing else matters. That's the entire interface.

### Answering your question directly

> *"I know I will have to basically write a version of the architectures
> that work with the CIFAR-10 dataset as the authors did right?"*

**Partly yes — but narrower than it sounds.** You only need to write **one**
new encoder (your own), not CIFAR-adapted copies of all seven existing
backbones. The existing ones are already CIFAR-adapted and stay as-is; your
new encoder just becomes an **eighth** option in `get_model()`.

What "CIFAR-adapted" means in practice (look at the `NOTE` comments in
`paper/architectures/mobilenet.py:45, 54, 75`):

- **Early stride of 1 instead of 2** — stock ImageNet backbones downsample
  aggressively in the first conv because they expect 224×224 input. On
  32×32, that kills the feature map before it's done anything useful.
- **Final pool sized to match the remaining spatial dims** — MobileNet's
  feature map is 4×4 before the global pool, so it uses `avg_pool2d(out, 4)`
  instead of the ImageNet-style `avg_pool2d(out, 7)`.
- **No aux heads, no 7×7 input stem** — GoogLeNet and ResNet have similar
  simplifications. Compare `paper/architectures/resnet.py:117-127` to a
  torchvision ResNet-18 to see them.

### Step-by-step: adding a new encoder

1. **Create `paper/architectures/my_encoder.py`**. At the top:

   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from memorywrap import MemoryWrapLayer as EncoderMemoryWrapLayer
   from memorywrap import BaselineMemory as MemoryWrapLayer
   ```

2. **Write your standard classifier** (optional, but useful as a baseline).
   The only thing it needs to do differently from ImageNet-style code is
   downsample more gently and pool over whatever spatial size is left at the
   end. Let `d` be the dimension of your final feature vector.

3. **Write the `Memory*` variant** following the MobileNet template
   (`paper/architectures/mobilenet.py:81–133`):

   ```python
   class MemoryMyEncoder(nn.Module):
       def __init__(self, num_classes=10):
           super().__init__()
           # ... your conv/attention/whatever stack ending in a [b, d] vector
           self.mw = MemoryWrapLayer(d, num_classes)   # BaselineMemory

       def forward_encoder(self, x):
           # ... your encoder forward, return a [b, d] tensor
           return out

       def forward(self, x, ss, return_weights=False):
           out = self.forward_encoder(x)
           out_ss = self.forward_encoder(ss)
           return self.mw(out, out_ss, return_weights)
   ```

4. **Write the `EncoderMemory*` variant** by subclassing `MemoryMyEncoder`
   and only swapping the head. Compare
   `paper/architectures/mobilenet.py:136–157`:

   ```python
   class EncoderMemoryMyEncoder(MemoryMyEncoder):
       def __init__(self, num_classes=10):
           super().__init__(num_classes)
           self.mw = EncoderMemoryWrapLayer(d, num_classes)   # real Memory Wrap
   ```

5. **Register it in `get_model()`** at `paper/utils/utils.py:83–163` by
   adding a new branch:

   ```python
   elif model_name == 'my_encoder':
       if model_type == 'memory':
           model = my_encoder.MemoryMyEncoder(num_classes)
       elif model_type == 'encoder_memory':
           model = my_encoder.EncoderMemoryMyEncoder(num_classes)
       else:
           model = my_encoder.MyEncoder(num_classes)
   ```

   And add the import at the top: `from architectures import my_encoder`.

6. **Train** by setting `model: my_encoder` in `paper/config/train.yaml:3`
   and running `python train.py`. Everything downstream (evaluation,
   `generate_memory_images.py`, `generate_heatmaps.py`) reads the model name
   from the checkpoint and will pick up your new class automatically.

### Common pitfalls

- **Shape mismatch at `matmul`.** If `forward_encoder` returns `[b, C, H, W]`
  instead of `[b, d]`, the matmul at `memory.py:111` will throw. Always
  finish with a global pool + `.view(b, -1)` (or `.flatten(1)`).
- **Batch-norm with `m=1` memory**. If you ever run with a memory set of
  size 1, BatchNorm on the encoder breaks in training mode because it has
  no variance to compute. Keep `batch_size_memory ≥ 2`, or use GroupNorm /
  LayerNorm in your encoder if you need to support `m=1`.
- **Non-shared encoder**. The same module instance must encode both input
  and memory set. Don't duplicate the encoder or deep-copy it — the weights
  need to be tied, otherwise the cosine similarity becomes meaningless and
  you lose the interpretability guarantee.
- **Huge `d`**. The MLP head is `[2d → 4d → num_classes]` (`memory.py:87`).
  If `d = 2048`, that's an 8192-unit hidden layer — can dominate parameter
  count and slow training. Consider a projection down to a smaller `d`
  inside your encoder, or pass a custom `classifier` to
  `MemoryWrapLayer(..., classifier=...)` — see the signature at
  `memory.py:72–87`.
- **Different input size than 32×32**. If your encoder needs 224×224 (e.g.
  you're porting a pretrained ViT), you also need to change the transforms
  in `paper/datasets.py` so the `DataLoader` upscales CIFAR images. The
  encoder contract doesn't care about `H, W` — only that the output is
  `[b, d]` — but the transforms have to match whatever your encoder expects.

---

## Contrastive encoder pretraining

The original Memory Wrap paper trains the encoder with cross-entropy — the
content-attention cosine geometry is whatever CE happens to produce as a
byproduct. This repo adds a pretraining script that lets you choose the
encoder's training signal explicitly:

- **SupCon** (Khosla et al., 2020) — supervised. All same-class image pairs
  are positives. Biases content attention toward **same-class** memories.
- **SimCLR** (Chen et al., 2020) — self-supervised, no labels. Only positive
  for each anchor is the *other augmented view of the same image*. Biases
  content attention toward **visually similar** memories, regardless of class.
- **Hybrid** — weighted sum `α · L_supcon + (1-α) · L_simclr`. Hierarchical
  feature geometry: augmentation-invariant individual clusters nested inside
  class clusters. Biases retrieval toward **both visually similar AND
  same-class** memories.

See `encoder_pretraining_design.md` for the deeper design discussion of why
these (and not e.g. PIP-Net) are the natural first experiments.

### What was added

- **`paper/pretrain_supcon.py`** — one script that supports SupCon, SimCLR,
  and their hybrid via a `--loss` flag (and `--hybrid_alpha` for the mix
  weight). Supports both CIFAR10 and SVHN via `--dataset`. Inlines the
  contrastive loss, 2-view augmentation wrapper, and training loop.
- **`paper/train.py`** — two new flags: `--pretrained_encoder=<path>` and
  `--freeze_encoder=<bool>`. ~10 lines of additions total.

### Two-stage pilot: SVHN first, CIFAR10 later

The repo ships with one pretrained checkpoint, `paper/models/2000.pt`, for
`mobilenet` on `SVHN` with `train_examples=2000` in `encoder_memory`
modality. Stage 1 reuses it as the CE baseline (saves ~300 epochs of
training) to validate the pipeline end-to-end. Stage 2 is the real
experiment on CIFAR10 where the qualitative comparison actually matters.

#### Stage 1 — SVHN sanity check (reuses pretrained CE baseline)

```bash
cd paper/

# Confirm train.yaml has: dataset_name: SVHN, train_examples: 2000
# (already the default in this repo)

# Pretrain three encoder variants on SVHN
python pretrain_supcon.py --dataset=SVHN --loss=supcon
python pretrain_supcon.py --dataset=SVHN --loss=simclr --temperature=0.5
python pretrain_supcon.py --dataset=SVHN --loss=hybrid

# Train Memory Wrap head on top of each pretrained encoder (frozen)
python train.py --modality=encoder_memory \
    --pretrained_encoder=models/SVHN/supcon/mobilenet/1.pt --freeze_encoder=True
python train.py --modality=encoder_memory \
    --pretrained_encoder=models/SVHN/simclr/mobilenet/1.pt --freeze_encoder=True
python train.py --modality=encoder_memory \
    --pretrained_encoder=models/SVHN/hybrid/mobilenet/1.pt --freeze_encoder=True

# CE baseline is the existing paper/models/2000.pt — no training needed

# Generate retrieval images for all four
python scripts/generate_memory_images.py --path_model=models/2000.pt
for tag in supcon simclr hybrid; do
  python scripts/generate_memory_images.py \
    --path_model=models/SVHN/encoder_memory_${tag}/mobilenet/2000/1.pt
done
```

#### Stage 2 — CIFAR10 real experiment

```bash
# Edit paper/config/train.yaml:
#   dataset_name: CIFAR10
#   train_examples: 2000   (or whatever you want)

# Pretrain three CIFAR10 encoders (--dataset=CIFAR10 is the default)
python pretrain_supcon.py --loss=supcon
python pretrain_supcon.py --loss=simclr --temperature=0.5
python pretrain_supcon.py --loss=hybrid

# CE baseline must be retrained (no pretrained CIFAR10 checkpoint exists)
python train.py --modality=encoder_memory

# Memory Wrap heads on top of each pretrained encoder
python train.py --modality=encoder_memory \
    --pretrained_encoder=models/CIFAR10/supcon/mobilenet/1.pt --freeze_encoder=True
python train.py --modality=encoder_memory \
    --pretrained_encoder=models/CIFAR10/simclr/mobilenet/1.pt --freeze_encoder=True
python train.py --modality=encoder_memory \
    --pretrained_encoder=models/CIFAR10/hybrid/mobilenet/1.pt --freeze_encoder=True

# Generate retrieval images for all four
for path in \
    models/CIFAR10/encoder_memory/mobilenet/*/1.pt \
    models/CIFAR10/encoder_memory_{supcon,simclr,hybrid}/mobilenet/*/1.pt; do
  python scripts/generate_memory_images.py --path_model="$path"
done
```

### What to look for when comparing

For each test query image, `generate_memory_images.py` renders the retrieved
memories (sparsemax-attended, non-zero weights only). Expected behaviour
across the four conditions:

- **CE baseline:** retrievals are whatever's class-discriminative for CE
  — often same-class but with some visual noise.
- **SupCon:** retrievals are more consistently same-class; within-class
  variation is averaged out.
- **SimCLR:** retrievals look *visually* similar to the query (similar
  pose, background, color palette) but may include other classes.
- **Hybrid:** retrievals should be same-class AND look like the query.
  This is the condition that directly matches the original motivation
  for the experiment — "explanations that look right AND are right."

Beyond visual inspection, useful quantitative metrics:

- **Retrieval purity** — of the non-zero attended memories, what fraction
  share the query's class? SupCon should win here.
- **Perceptual similarity** — e.g. LPIPS distance between query and
  attended memories. SimCLR should win here.
- **Test accuracy** — overall classification performance. Ambiguous which
  wins; probably CE ≥ SupCon ≥ SimCLR, but the gap matters.

### Swapping the loss

Both SupCon and SimCLR share the same softmax-over-similarities structure
in `paper/pretrain_supcon.py:66-138`. To add a third objective (e.g.
triplet loss, NT-Xent with hard negative mining), extend the
`contrastive_loss` function with another `labels=None`-style branch or
add a sibling function.

---

## File index

| File | Role |
|---|---|
| `memory.py` | Core Memory Wrap + Baseline implementation (the dashed box in the diagram). |
| `paper/train.py` | Training loop; dispatches `std`/`memory`/`encoder_memory` modalities. Supports `--pretrained_encoder` and `--freeze_encoder`. |
| `paper/pretrain_supcon.py` | Self-contained SupCon pretraining for any backbone's `forward_encoder`. |
| `paper/datasets.py` | (Legacy, unused) Top-level dataset helpers. |
| `paper/utils/datasets.py` | Canonical dataset/loader builders used by `train.py` via `utils.get_loaders`. |
| `paper/config/train.yaml` | Hyperparameters for downstream classification training. |
| `paper/architectures/*.py` | Backbone CNNs. Each defines a standard variant, `Memory*` variant (= baseline), and `EncoderMemory*` variant (= real Memory Wrap). |
| `paper/utils/utils.py` | Model factory (`get_model`), eval loops, and data wiring. |
| `paper/utils/counterfactuals_utils.py` | Explanation-by-example and counterfactual extraction using the sparse `content_weights`. |
| `encoder_pretraining_design.md` | Design notes on SupCon vs PIP-Net and how to add other encoder-pretraining variants. |
