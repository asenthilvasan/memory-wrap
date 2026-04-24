"""Minimal contrastive pretraining for Memory Wrap encoders.

Supports three objectives:
  - supcon  (Khosla et al., 2020): class-label-supervised. All same-class
    features are positives. Biases retrieval toward same-class memories.
  - simclr  (Chen et al., 2020): self-supervised, no labels. Only positive
    for each anchor is the OTHER augmented view of the same image. Biases
    retrieval toward visually similar memories regardless of class.
  - hybrid: weighted sum  alpha * supcon + (1-alpha) * simclr.  Produces a
    hierarchical feature geometry: tightest clusters around each individual
    image (augmentation-invariant), medium clusters around each class,
    far separation between classes. "Looks similar AND same-class."

All three train `forward_encoder` so that cos(f(x), f(y)) is larger for
pairs the loss considers positive. The downstream effect is in Memory
Wrap's `sparsemax(cos(encoder(query), encoder(memory_i)))` attention.

Usage:
    python pretrain_supcon.py                        # CIFAR10, supcon, mobilenet
    python pretrain_supcon.py --dataset=SVHN         # SVHN instead of CIFAR10
    python pretrain_supcon.py --loss=simclr          # SimCLR (no labels)
    python pretrain_supcon.py --loss=hybrid          # SupCon + SimCLR (50/50)
    python pretrain_supcon.py --loss=hybrid --hybrid_alpha=0.7  # 70% supcon
    python pretrain_supcon.py --model=resnet18 --epochs=200

Output: models/<dataset>/{supcon,simclr,hybrid}/<model>/1.pt

Plug the checkpoint into downstream Memory Wrap training via:
    python train.py --modality=encoder_memory \\
        --pretrained_encoder=models/<dataset>/<loss>/<model>/1.pt \\
        --freeze_encoder=True
"""
import os
# absl is used for CLI flags to match the convention in train.py.
import absl.app, absl.flags
import torch
# Kubernetes pods default /dev/shm to 64MB, which is far too small for
# PyTorch DataLoader's default shared-memory tensor sharing. Switching to
# the 'file_system' strategy uses file descriptors instead and avoids the
# "unable to allocate shared memory" error on constrained pods.
torch.multiprocessing.set_sharing_strategy('file_system')
# F provides L2 normalization (F.normalize); we need unit vectors because
# SupCon works on cosine similarity = dot product of L2-normalized features.
import torch.nn.functional as F
from torchvision import datasets, transforms
# Reuse the existing model factory so SupCon-pretrained checkpoints use the
# exact same backbone as downstream Memory Wrap training.
import utils.utils as utils


# --- CLI flags ---------------------------------------------------------------
# Default hyperparameters follow the SupCon paper's CIFAR-10 recipe.

# Which backbone to pretrain. Must be a key accepted by utils.get_model (e.g.
# 'mobilenet', 'resnet18', 'densenet', 'efficientnet', 'googlenet', ...).
absl.flags.DEFINE_string('model', 'mobilenet', 'Backbone (see utils/utils.py get_model)')
# 100 epochs is the typical CIFAR-10 SupCon budget; more helps slightly.
absl.flags.DEFINE_integer('epochs', 100, 'Pretraining epochs')
# SupCon benefits from large batches because more samples = more negatives
# per anchor = sharper contrastive signal. 256 is a single-GPU compromise.
absl.flags.DEFINE_integer('batch_size', 256, 'Pretraining batch size')
# Lower temperature = sharper softmax = harder negatives dominate the loss.
# 0.07 is the SupCon default; SimCLR's original paper used 0.5. Tune per task.
absl.flags.DEFINE_float('temperature', 0.07, 'Softmax temperature (0.07 supcon, 0.5 simclr typical)')
# Choice of contrastive objective. 'supcon' uses class labels as in Khosla et
# al. 2020; 'simclr' ignores labels and only treats the other view of the same
# image as a positive (Chen et al. 2020); 'hybrid' is a weighted sum of both.
# Retrieval behaviour: supcon -> same-class; simclr -> visually similar;
# hybrid -> both (tighter within-class, augmentation-invariant).
absl.flags.DEFINE_enum('loss', 'supcon', ['supcon', 'simclr', 'hybrid'],
                       'Contrastive objective.')
# Only used when --loss=hybrid. Weight on the SupCon term; (1-alpha) goes to
# the SimCLR term. 0.5 = equal. Higher -> more class-clustered; lower -> more
# visually-invariant.
absl.flags.DEFINE_float('hybrid_alpha', 0.5, 'SupCon weight in hybrid loss (0..1)')
# Large LR is standard for contrastive pretraining; cosine schedule below
# anneals it smoothly to zero. If you cut batch_size, cut lr proportionally.
absl.flags.DEFINE_float('lr', 0.5, 'Learning rate (SGD)')
absl.flags.DEFINE_string('data_dir', 'datasets', 'Dataset directory')
# Which image dataset to pretrain on. Both are 32x32 10-class datasets so the
# augmentation recipe and backbone architectures work for either, but they
# need different normalization stats and (for SVHN) no horizontal flip
# because flipped digits aren't digits.
absl.flags.DEFINE_enum('dataset', 'CIFAR10', ['CIFAR10', 'SVHN'],
                       'Dataset to pretrain on.')
# Data loading parallelism. With 2-view augmentation this pipeline is
# CPU-bound (each batch needs 2B independent RandomResizedCrop+ColorJitter
# passes). On a modern GPU (L4/L40/A100/H100) the default of 4 workers
# typically starves the GPU; 8-16 is a better starting point.
absl.flags.DEFINE_integer('num_workers', 8, 'DataLoader worker processes')
FLAGS = absl.flags.FLAGS


# Per-dataset specs: torchvision dataset class, its train-split kwargs,
# per-channel normalization stats (must match paper/utils/datasets.py so
# downstream Memory Wrap training sees features on the same scale), and
# whether horizontal flip is an identity-preserving augmentation.
DATASET_SPECS = {
    'CIFAR10': {
        'cls': datasets.CIFAR10,
        'split_kwargs': {'train': True},
        'mean': [0.4914, 0.4822, 0.4465],
        'std':  [0.2023, 0.1994, 0.2010],
        'hflip': True,   # cats/planes/etc. are roughly symmetric
    },
    'SVHN': {
        'cls': datasets.SVHN,
        'split_kwargs': {'split': 'train'},
        'mean': [0.485, 0.456, 0.406],   # matches paper/utils/datasets.py get_SVHN
        'std':  [0.229, 0.224, 0.225],
        'hflip': False,  # '3' flipped is not a '3'
    },
}


def contrastive_loss(features, labels, temp=0.07):
    """SupCon (Khosla et al., 2020) or SimCLR (Chen et al., 2020) loss.

    Both losses share the same softmax-over-similarities structure. The only
    difference is WHICH pairs count as positives:
      - SupCon: all same-class feature pairs (uses labels).
      - SimCLR: only the two-view pair of the same image (labels=None).

    Args:
        features: [2B, d] L2-normalized feature vectors. First B rows are
            view-1 of each image, next B rows are view-2 of the same images
            (in the same order).
        labels:   [B] class labels (SupCon mode) or None (SimCLR mode).
        temp:     softmax temperature; lower = harder negatives dominate.

    Returns:
        Scalar loss averaged over all 2B anchors.
    """
    twoB = features.size(0)
    B = twoB // 2

    # Positive mask: mask[i, j] = 1 iff feature j is a positive for anchor i.
    # Shape: [2B, 2B].
    if labels is None:
        # SimCLR: the only positive for anchor i is its OTHER augmented view
        # (same underlying image). Since rows 0..B-1 are view-1 and rows
        # B..2B-1 are view-2 of the same images in the same order, the
        # desired mask is the identity rolled by B columns — this places a
        # 1 at position (i, (i+B) % 2B) for every i.
        mask = torch.eye(twoB, device=features.device).roll(B, dims=1)
    else:
        # SupCon: duplicate labels (view-1 and view-2 share their class),
        # then pairs with matching labels become positives. Zero the
        # diagonal so an anchor is never its own positive.
        labels = torch.cat([labels, labels])
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)

    # Pairwise cosine similarities scaled by temperature. Because features
    # are L2-normalized, matmul is equivalent to pairwise cosine. Shape:
    # [2B, 2B]. logits[i, j] = sim(anchor_i, sample_j) / temp.
    logits = features @ features.T / temp

    # Numerical stability: subtract per-row max from each logit before
    # exponentiating. This doesn't change the softmax (constants cancel)
    # but keeps exp() from overflowing in fp16 / large-temp regimes.
    # .detach() so the subtracted max isn't part of the backward graph.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Denominator for log-softmax must EXCLUDE the self-similarity (which is
    # always 1/temp after normalization — trivially the largest logit and
    # would dominate the softmax). `not_self` is a [2B, 2B] matrix with 1s
    # everywhere except the diagonal.
    not_self = 1 - torch.eye(twoB, device=features.device)

    # log_prob[i, j] = log P(sample j | anchor i) under the softmax over all
    # non-self samples. The 1e-12 is a guard against log(0) if the entire
    # row of exponentials happens to be zero (effectively never, but safe).
    log_prob = logits - torch.log((logits.exp() * not_self).sum(dim=1, keepdim=True) + 1e-12)

    # For each anchor i: average log_prob over its positives P(i).
    #   - (mask * log_prob).sum(dim=1): sum of log-probs over positive j.
    #   - mask.sum(dim=1): |P(i)|, the number of positives for anchor i.
    #   - .clamp(min=1): edge case — if an anchor happens to have no
    #     positives in this batch (rare with balanced sampling) divide by 1
    #     instead of 0. Since the numerator is also 0 for such anchors,
    #     their contribution to the mean becomes 0, not NaN.
    # Final: negate (SupCon maximizes log-prob so loss minimizes -log-prob)
    # and average over all 2B anchors.
    return -(mask * log_prob).sum(dim=1).div(mask.sum(dim=1).clamp(min=1)).mean()


class TwoViews:
    """Return two independently-augmented versions of the same input image.

    Wrapping the torchvision transform in this class is the standard SimCLR /
    SupCon trick: it makes DataLoader yield batches shaped as
        ((view1_batch, view2_batch), label_batch)
    so we get two stochastic views of every image to use as a known positive
    pair in the contrastive loss.
    """
    def __init__(self, t): self.t = t
    def __call__(self, x): return (self.t(x), self.t(x))


def main(argv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Input shape is fixed (32x32, constant batch size thanks to drop_last),
    # so let cuDNN benchmark kernels at startup and pick the fastest for
    # each conv. Free ~5-15% speedup on conv-heavy backbones.
    torch.backends.cudnn.benchmark = True
    spec = DATASET_SPECS[FLAGS.dataset]

    # --- Augmentation pipeline ----------------------------------------------
    # SimCLR-style augmentations. They have to be strong enough that two
    # views of the same image look meaningfully different (otherwise the
    # encoder just learns a trivial identity-ish mapping), but not so strong
    # that class-identifying content is destroyed. Built conditionally from
    # the dataset spec (SVHN skips horizontal flip).
    aug_list = [
        # Random crop + resize: forces spatial invariance. scale=(0.2, 1.0)
        # means crops can be as small as 20% of the original area.
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    ]
    if spec['hflip']:
        aug_list.append(transforms.RandomHorizontalFlip())
    aug_list += [
        # ColorJitter with p=0.8: aggressive color perturbation. Critical
        # for preventing the encoder from using color shortcuts.
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # 20% chance of converting to grayscale — another anti-color-shortcut.
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # Per-channel normalization that matches paper/utils/datasets.py so
        # downstream Memory Wrap training and eval see the same feature scale.
        transforms.Normalize(spec['mean'], spec['std']),
    ]
    aug = transforms.Compose(aug_list)

    # Dataset returns ((view1, view2), label) per sample thanks to TwoViews.
    ds = spec['cls'](FLAGS.data_dir, download=True,
                     transform=TwoViews(aug), **spec['split_kwargs'])
    # drop_last=True: SupCon needs a predictable 2B batch shape; dropping
    # the incomplete final batch avoids per-epoch shape edge cases.
    # persistent_workers=True: don't tear down and respawn worker processes
    #   between epochs (saves ~1-2s of Python startup per epoch).
    # prefetch_factor=4: each worker keeps 4 batches queued ahead of the
    #   GPU, hiding CPU augmentation latency behind GPU compute.
    loader = torch.utils.data.DataLoader(ds, batch_size=FLAGS.batch_size,
        shuffle=True, drop_last=True, pin_memory=True,
        num_workers=FLAGS.num_workers, persistent_workers=FLAGS.num_workers > 0,
        prefetch_factor=4 if FLAGS.num_workers > 0 else None)

    # --- Model --------------------------------------------------------------
    # We instantiate the 'encoder_memory' variant (= real Memory Wrap) so we
    # get access to `forward_encoder`, which returns the [B, d] feature
    # vector that Memory Wrap attends over. The self.mw head is PRESENT on
    # the model but never called during pretraining — it stays at random
    # init and receives no gradient updates, so its parameters persist
    # untouched into the saved checkpoint. Downstream train.py reinitializes
    # it anyway (or ignores those keys if the modality differs).
    model = utils.get_model(FLAGS.model, 10, model_type='encoder_memory').to(device)

    # SGD + momentum + weight decay matches the SupCon paper's recipe for
    # CIFAR-10. Nesterov gives a small convergence boost on this setup.
    opt = torch.optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=0.9,
                          weight_decay=1e-4, nesterov=True)
    # Cosine schedule: smoothly anneals LR from `lr` to 0 over all epochs.
    # Empirically better than step schedules for contrastive pretraining.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FLAGS.epochs)
    # Automatic mixed precision: roughly 2x training speedup on modern GPUs
    # with negligible accuracy impact. GradScaler handles the dynamic loss
    # scaling needed to prevent fp16 gradient underflow.
    scaler = torch.cuda.amp.GradScaler()

    # --- Training loop ------------------------------------------------------
    model.train()
    for ep in range(1, FLAGS.epochs + 1):
        for (v1, v2), y in loader:
            # Stack both views into a single tensor of shape [2B, 3, 32, 32].
            # Encoding both views in the same forward pass keeps BatchNorm
            # statistics consistent across views (important! — separate
            # forward passes would compute different running means/stds for
            # the two views and degrade the contrastive signal).
            imgs = torch.cat([v1, v2]).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                # forward_encoder returns [2B, d] raw features. F.normalize
                # projects them onto the unit hypersphere so matmul later =
                # cosine similarity.
                feat = F.normalize(model.forward_encoder(imgs), dim=1)
                # SupCon: pass labels. SimCLR: pass None. Hybrid: both,
                # combined as alpha * supcon + (1 - alpha) * simclr.
                if FLAGS.loss == 'supcon':
                    loss = contrastive_loss(feat, y, FLAGS.temperature)
                elif FLAGS.loss == 'simclr':
                    loss = contrastive_loss(feat, None, FLAGS.temperature)
                else:  # hybrid
                    l_sup = contrastive_loss(feat, y, FLAGS.temperature)
                    l_sim = contrastive_loss(feat, None, FLAGS.temperature)
                    loss = FLAGS.hybrid_alpha * l_sup + (1 - FLAGS.hybrid_alpha) * l_sim

            # scaler.scale: multiplies loss by dynamic scale factor to keep
            # fp16 gradients in representable range.
            scaler.scale(loss).backward()
            # scaler.step: unscales grads and calls optimizer.step(), but
            # skips the step if inf/NaN gradients are detected.
            scaler.step(opt)
            # scaler.update: adjusts the scale factor for next iteration.
            scaler.update()

        sched.step()  # Cosine schedule steps once per epoch, not per batch.
        print(f'Epoch {ep}/{FLAGS.epochs}  loss={loss.item():.4f}')

    # --- Save checkpoint ----------------------------------------------------
    # Checkpoint format mirrors what train.py saves: model_name and
    # dataset_name are used by downstream scripts (generate_memory_images.py,
    # generate_heatmaps.py) to reconstruct the correct architecture. The
    # 'modality' key is informational — train.py reads state_dict only.
    # Save path includes both dataset and loss so runs don't clobber each
    # other when pilot-comparing across datasets or objectives.
    out = f'models/{FLAGS.dataset}/{FLAGS.loss}/{FLAGS.model}/1.pt'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'model_name': FLAGS.model,
                'num_classes': 10, 'modality': f'{FLAGS.loss}_pretrained',
                'dataset_name': FLAGS.dataset}, out)
    print(f'Saved {out}')


if __name__ == '__main__':
    absl.app.run(main)
