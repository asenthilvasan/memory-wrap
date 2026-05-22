# SVHN Ablation Analysis: SupCon × Memory Wrap × Encoder Freezing

## Setup

We evaluate the contributions of (i) supervised contrastive pretraining
(SupCon, Khosla et al. 2020), (ii) the Memory Wrap (MW) attention head, and
(iii) downstream encoder freezing on the small-data SVHN classification task.
All experiments use a MobileNetV2 backbone trained on a 2,000-sample subset of
SVHN, with 15 random seeds per cell. SupCon pretraining is performed for 40
epochs with batch size 256; downstream training runs for 40 epochs with batch
size 128. Reported accuracies are mean test accuracy over 15 seeds, with
standard deviation across seeds.

## Results

| Type                       | Train Ex | Epochs (pre/down) | Batch (pre/down) | Test Acc          |
|----------------------------|----------|-------------------|------------------|-------------------|
| Scratch + Linear           | 2000     | — / 40            | — / 128          | 71.29 ± 5.75      |
| Scratch + MW               | 2000     | — / 40            | — / 128          | 81.08 ± 1.19      |
| SupCon + Linear (frozen)   | 2000     | 40 / 40           | 256 / 128        | 82.93 ± 0.39      |
| SupCon + Linear (fine-tune)| 2000     | 40 / 40           | 256 / 128        | 84.65 ± 0.68      |
| SupCon + MW (frozen)       | 2000     | 40 / 40           | 256 / 128        | 82.16 ± 0.32      |
| SupCon + MW (fine-tune)    | 2000     | 40 / 40           | 256 / 128        | **84.90 ± 0.48**  |

## Key Findings

**1. Memory Wrap is a strong inductive prior on its own.**
A from-scratch MW model reaches 81.08% test accuracy without any pretraining,
a +9.79 pp improvement over the equivalently-trained linear-head baseline
(71.29%). The MW attention mechanism — which classifies a query by softmax
attention over a set of memory exemplars — appears to act as a regularizer
during training, both improving accuracy and substantially reducing seed
variance (σ: 5.75 → 1.19, a 4.8× reduction).

**2. SupCon dramatically improves the linear head, but only marginally
improves Memory Wrap.**
Pretraining gives the linear head a +13.36 pp boost (71.29 → 84.65 fine-tuned)
— enough to lift it to MW's level. In contrast, SupCon adds only +3.82 pp on
top of MW (81.08 → 84.90 fine-tuned), and only +1.08 pp when the SupCon
encoder is frozen (82.16). The two methods appear to be partially redundant:
both shape the encoder toward a class-discriminative feature space, but
through different mechanisms — SupCon during self-supervised pretraining via a
contrastive objective, MW during supervised downstream training via the
gradient flowing back through its attention mechanism.

**3. With SupCon pretraining, MW and Linear converge to the same accuracy
ceiling.**
The fine-tuned numbers (Linear: 84.65, MW: 84.90) are statistically
indistinguishable given their respective standard deviations. Once the
encoder produces sufficiently linearly-separable features, the additional
modeling capacity of MW provides no measurable benefit. This is consistent
with the linear-probe convention used in self-supervised learning: a strong
representation reduces the importance of the classification head.

**4. Frozen SupCon features slightly handicap Memory Wrap.**
In the frozen-encoder setting, the linear head outperforms MW (82.93 vs
82.16). This is consistent with the expected geometric conflict: SupCon
explicitly minimizes intra-class feature variance to encourage tight class
clusters on the unit hypersphere, while MW's content-based attention requires
intra-class diversity in order to differentiate among same-class memory
exemplars. When the encoder is frozen, MW cannot recover this diversity.
Fine-tuning (Cell 6) closes this gap (+2.74 pp over frozen) by allowing the
classification gradient to perturb the SupCon-shaped features.

**5. SupCon's principal contribution to MW is variance reduction, not
accuracy.**
Standard deviation across seeds drops from 1.19 (Scratch + MW) to 0.32
(SupCon + MW frozen), a 3.7× reduction. SupCon initialization places the
optimizer in a more consistent region of the loss landscape, even when the
final accuracy is comparable. For practitioners, this translates to fewer
"bad seed" runs and more reliable comparisons across hyperparameter sweeps.

## Discussion

The traditional narrative of self-supervised pretraining frames it as a
universal accuracy booster: "any downstream classifier benefits from a
SupCon encoder." Our results complicate this picture: the magnitude of
SupCon's benefit depends critically on the inductive bias of the downstream
classifier. A weak head (linear, no parameters that affect feature geometry)
gains substantially; a strong head (MW, whose training already shapes
features through attention) gains relatively little.

This suggests that MW and SupCon may be viewed as *substitutes* rather than
*complements* in the regime studied here. Both methods aim to recover a
class-discriminative feature space; whether one applies SupCon at pretraining
time or MW at downstream time, the final encoder converges to representations
of similar quality. Combining them yields diminishing returns.

We note an additional consequence: if MW is the downstream architecture of
choice, the SupCon pretraining stage's primary practical value is **stability,
not accuracy**. This is non-trivial for low-resource regimes, where reliable
seed-to-seed reproducibility may be more valuable than a small headline
accuracy gain.

## Limitations and Future Work

- **Single dataset and sample size.** All results are on SVHN with 2,000
  training examples. The asymmetry between linear and MW gains may not
  replicate on harder datasets (CINIC-10, full SVHN, CIFAR-100) or larger
  training sets, where the encoder is the genuine bottleneck. CINIC-10
  experiments are in progress.

- **Single backbone.** Only MobileNetV2 was evaluated. Stronger backbones
  (ResNet-50, ViT) may shift the encoder-vs-head balance and change which
  recipe wins.

- **Compute budget.** SupCon pretraining adds 40 encoder-only epochs to the
  training cost. The marginal +3.82 pp gain for MW (fine-tune) does not
  obviously justify this additional compute; the marginal +13.36 pp gain
  for Linear clearly does. We recommend reporting compute-normalized
  comparisons in the final paper.

- **Joint training not evaluated.** Combined SupCon + MW training in a single
  stage was not tested, but is expected to underperform two-stage training
  due to known geometric and pipeline incompatibilities (Khosla et al. 2020,
  ablation Table 7).

## Reproducibility

All commands and configurations are in `paper/config/run_svhn_ablation.sh`.
Pretrained encoder is saved at `models/SVHN/supcon/mobilenet/2000/1.pt`.
Per-cell results are in `/root/cinic_run/logs/0[1-6]_*.txt`.
