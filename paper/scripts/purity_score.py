"""PIP-Net-inspired purity score for Memory Wrap checkpoints.

Two metrics per query, both in [0, 1]:
  soft  = sum of sparsemax attention weights on same-class memory items
  topK  = fraction of the K nearest memory items (cosine) that match class

Averaged over `--num_redraws` random memory draws per checkpoint, then
mean/std across checkpoints in the directory. Mirrors paper/eval.py.
"""
import sys
sys.path.append('..')

import os
import time

import absl.app
import absl.flags
import numpy as np
import torch

import utils.datasets as datasets
import utils.utils as utils

absl.flags.DEFINE_string("path", None, "Dir of .pt files or single .pt path.")
absl.flags.DEFINE_string("dir_dataset", '../datasets/', "Datasets directory.")
absl.flags.DEFINE_integer("num_redraws", 5, "Random memory draws per checkpoint.")
absl.flags.DEFINE_string("topk", "1,5,10", "Comma-separated K values.")
absl.flags.mark_flag_as_required("path")
FLAGS = absl.flags.FLAGS


def purity(model, test_loader, mem_loader, topk, num_redraws, device):
    model.eval()
    soft, hard = [], {K: [] for K in topk}
    with torch.no_grad():
        for _ in range(num_redraws):
            mem_iter = iter(mem_loader)
            s_sum, h_sum, n = 0.0, {K: 0.0 for K in topk}, 0
            for q_imgs, q_lbl in test_loader:
                try:
                    m_imgs, m_lbl = next(mem_iter)
                except StopIteration:
                    mem_iter = iter(mem_loader); m_imgs, m_lbl = next(mem_iter)
                q_imgs, q_lbl = q_imgs.to(device), q_lbl.to(device)
                m_imgs, m_lbl = m_imgs.to(device), m_lbl.to(device)

                q_feat = model.forward_encoder(q_imgs)
                m_feat = model.forward_encoder(m_imgs)
                match = (q_lbl.unsqueeze(1) == m_lbl.unsqueeze(0)).float()

                _, w = model.mw(q_feat, m_feat, return_weights=True)
                s_sum += (w * match).sum().item()

                dist = utils.vector_distance(q_feat, m_feat, 'cosine')
                for K in topk:
                    idx = dist.topk(min(K, m_feat.size(0)), dim=1, largest=False).indices
                    h_sum[K] += match.gather(1, idx).mean(dim=1).sum().item()
                n += q_imgs.size(0)

            soft.append(s_sum / n)
            for K in topk: hard[K].append(h_sum[K] / n)
    return soft, hard


def run_experiment(path, dataset_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    topk = sorted({int(x) for x in FLAGS.topk.split(',')})

    if os.path.isdir(path):
        models, base = sorted(f for f in os.listdir(path) if f.endswith('.pt')), path
    else:
        models, base = [os.path.basename(path)], os.path.dirname(path)

    run_soft, run_hard = [], {K: [] for K in topk}
    for name in models:
        run = int(name.split('.')[0]) - 1
        utils.set_seed(run)
        ckpt = torch.load(os.path.join(base, name), map_location=device)
        model = utils.get_model(ckpt['model_name'], ckpt['num_classes'],
                                model_type=ckpt['modality'])
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)

        _, _, test_loader, mem_loader = getattr(datasets, 'get_' + ckpt['dataset_name'])(
            dataset_dir, batch_size_train=128, batch_size_test=500,
            batch_size_memory=ckpt['mem_examples'],
            size_train=ckpt['train_examples'], seed=run,
        )

        t0 = time.time()
        soft, hard = purity(model, test_loader, mem_loader, topk,
                            FLAGS.num_redraws, device)
        run_soft.append(float(np.mean(soft)))
        for K in topk: run_hard[K].append(float(np.mean(hard[K])))

        log = f"Run:{run+1} | Soft:{run_soft[-1]:.4f}"
        for K in topk: log += f" | Top{K}:{run_hard[K][-1]:.4f}"
        log += f"  E:{(time.time()-t0)/60:.2f}min"
        print(log, flush=True)

    summary = f"SUMMARY (n={len(run_soft)}) | Soft: {np.mean(run_soft):.4f} +/- {np.std(run_soft):.4f}"
    for K in topk:
        summary += f" | Top{K}: {np.mean(run_hard[K]):.4f} +/- {np.std(run_hard[K]):.4f}"
    print(summary, flush=True)


def main(argv=None):
    run_experiment(FLAGS.path, FLAGS.dir_dataset)


if __name__ == '__main__':
    absl.app.run(main)
