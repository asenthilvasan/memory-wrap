# Memory Wrap Purity Score

A PIP-Net-inspired metric for evaluating whether a trained encoder produces
class-coherent memory sets. Implemented in `paper/scripts/purity_score.py`.

Each metric is computed per test query, averaged over the test set, averaged
over `--num_redraws` random memory-set draws (default 5), then aggregated
mean ± std across the seed checkpoints in a directory.

## Setup

For one test query `q` with true class `y_q` and a memory set of `M` images
(default `M = 100` on SVHN) with labels `l_1, ..., l_M`:

- The encoder produces a feature vector for `q` and for each memory image.
- Cosine **distance** is computed between `q`'s vector and each memory
  vector (`paper/utils/utils.py:vector_distance`).
- The Memory Wrap layer applies **sparsemax** to the negated distances to
  produce non-negative attention weights `w_1, ..., w_M` summing to 1
  (`memory.py`, used at `paper/architectures/*.py:self.mw`).

The purity script reads off two quantities from this state:

| Metric | What it asks | What it depends on |
|---|---|---|
| **Soft purity** | "How much attention mass landed on same-class memory items?" | Encoder geometry + sparsemax head behaviour |
| **Top-K hard purity** | "Among the K closest memory items, how many share the query's class?" | Encoder geometry only |

---

## 1. Soft purity

### Formula

```
purity_soft(q) = sum_i  w_i * 1[l_i == y_q]
```

where `w_i` is the sparsemax weight on memory item `i` and `1[...]` is 1 if
`l_i == y_q` else 0. Since `sum_i w_i = 1`, the result lies in `[0, 1]`.

### What it represents

The **total fraction of attention mass** the model put on same-class memory
items at inference time. Equivalently: the expected probability of landing
on a same-class memory item if you sampled one according to the model's own
attention distribution.

### Motivation

This is the only metric that reflects what Memory Wrap **actually does** at
inference. The classifier consumes `memory_vector = sum_i w_i * mem_feat_i`,
so attention mass on wrong-class items directly distorts the prediction.
Soft purity captures this exactly.

### Drawbacks

- **Conflates encoder and head.** A change in soft purity could come from
  the encoder improving (better geometry) or from sparsemax becoming
  sharper / flatter. The metric cannot distinguish these.
- **Cannot isolate the SupCon contribution.** SupCon is an encoder-only
  intervention. If SupCon tightens clusters but sparsemax does not adapt,
  soft purity might barely move even though the representation is better.
- **Bounded above by class balance.** With 10 balanced classes and `M=100`,
  ~10 memory items share `y_q`, so the maximum achievable soft purity per
  query is the fraction of total weight that can be placed on those ~10
  items.

### When it is the right metric

When the claim is *"the model retrieves class-coherent memories at
inference time"* — i.e., a statement about behaviour, not representation.

---

## 2. Top-K hard purity

### Formula

```
TopK(q) = indices of the K memory items with smallest cosine distance to q
purity_topK(q) = (1/K) * sum_{i in TopK(q)} 1[l_i == y_q]
```

Sparsemax is **not used**; ranking is by raw cosine similarity. Reported
for `K = 1, 5, 10` by default.

### What it represents

- **Top-1**: is the single nearest memory item the same class as the query?
  Equivalent to 1-NN classifier accuracy in the encoder's feature space.
- **Top-5 / Top-10**: of the K nearest memory items, what fraction match
  the query's class? Larger K probes farther into the local neighbourhood;
  PIP-Net's canonical K is 10.

### Motivation

A direct measure of **encoder feature geometry**, independent of the
attention head. It answers *"does the encoder put same-class images near
each other?"* — exactly the property SupCon explicitly optimises. If a
SupCon-pretrained encoder fails to improve Top-K, the pretraining did not
shape the geometry as intended.

### Drawbacks

- **Ignores the head.** A model with very high Top-K but a poorly
  calibrated sparsemax can still misclassify; Top-K will not show this.
- **K is a free parameter.** Different K values can disagree on which
  variant looks best; you must commit to one (or report a sweep). PIP-Net
  uses K=10.
- **Class-balance sensitive.** With ~10 same-class items in `M=100`, the
  theoretical ceiling for Top-K is `min(1, num_same_class_in_memory / K)`;
  bounded above by 1.0 for K ≤ 10 in expectation.
- **Equal weighting.** All K items count the same regardless of how close
  they are. A memory at rank 5 that is barely closer than rank 50 still
  gets a full vote.

### When it is the right metric

When the claim is *"SupCon (or any encoder-side intervention) produced a
better feature space"* — i.e., a statement about representation quality.

---

## Relationship to PIP-Net

PIP-Net's published "purity" metric (`util/eval_cub_csv.py` in the official
repo) is structurally hard top-K, but with two differences worth noting:

1. **Object scored.** PIP-Net scores **learned prototypes** (fixed across
   queries). Memory Wrap has no learned prototypes; the natural inversion
   scores **queries**, whose nearest memory items play the role of the
   prototype's top-K activating patches.
2. **Predicate.** PIP-Net uses CUB ground-truth part annotations (does
   body part X lie inside the patch?) and reports `max_part` purity per
   prototype. The closest Memory Wrap analogue is `max_class` over the
   top-K (mode purity) — "are the K closest memories internally
   coherent?" — independent of the query's true label.

The script's Top-K reports **class-supervised** purity (predicate uses
ground-truth `y_q`), which corresponds to 1-NN / K-NN accuracy. PIP-Net's
faithful analogue (mode purity, max over classes) is not implemented; the
two coincide when the dominant class in the top-K is also the query's
true class, which is the common case for a working model.

Soft purity is a Memory-Wrap-specific extension with no PIP-Net analogue;
PIP-Net prototypes do not have continuous attention weights over patches.

---

## Reading the script output

For a directory of 15 seed checkpoints:

```
Run:1  | Soft:0.4123 | Top1:0.5215 | Top5:0.4814 | Top10:0.4502  E:0.83min
...
Run:15 | Soft:0.4178 | Top1:0.5277 | Top5:0.4847 | Top10:0.4519  E:0.81min
SUMMARY (n=15) | Soft: 0.4148 +/- 0.0034 | Top1: 0.5251 +/- 0.0045 | Top5: 0.4827 +/- 0.0029 | Top10: 0.4511 +/- 0.0021
```

- Each `Run:N` line is the mean across `--num_redraws` memory draws for
  one seed checkpoint.
- The `SUMMARY` line is the mean ± std across seeds; this is the number
  to cite in tables.

### Sanity benchmarks (SVHN, 10 balanced classes, `M=100`)

- A random encoder yields ~0.10 on every metric.
- A well-trained MobileNet on SVHN typically lands in `[0.45, 0.85]`
  depending on data budget and pretraining recipe.
- Soft is usually slightly below Top-1 on a working model — sparsemax
  spreads some mass to runners-up, which dilutes the rank-1 verdict.

### Expected ordering across variants

If the SupCon-clustering hypothesis holds, both metrics should follow:

```
supcon  >=  hybrid  >  simclr  >  scratch
```

with the **Top-K gap larger than the Soft gap** — SupCon shapes geometry
directly (lifts Top-K) but does not retrain the sparsemax head (only
indirectly lifts Soft). The size of the Top-K-vs-Soft gap quantifies how
much of the encoder improvement the head fails to exploit.
