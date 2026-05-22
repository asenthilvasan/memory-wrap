#!/bin/bash
# ============================================================================
# CINIC-10 ablation sweep — mirrors run_svhn_ablation.sh
#   6 cells: scratch/supcon x linear/MW x frozen/finetune
#   Uses 300 epochs for both pretrain and downstream (per CINIC10 block in
#   config/train.yaml; pretrain epochs passed via CLI).
#
# PREREQUISITES (run these once before launching this script):
#   1. Symlink CINIC-10 dataset to fast storage (config/setup_cinic.sh)
#   2. Pretrain the SupCon encoder (300 epochs). Pick one of:
#
#      # Option A: canonical w/ projection head + BN  (~45-90 min on 4090)
#      python -u pretrain_supcon.py \
#          --dataset=CINIC10 --loss=supcon --model=mobilenet \
#          --train_examples=2000 --seed=1 --epochs=300 \
#          --projection_dim=128 --projection_bn=True \
#          --warmup_epochs=10 --temperature=0.07 --lr=0.1 \
#          2>&1 | tee /root/cinic_run/logs/00_pretrain_cinic_canonical.txt
#
#      # Option B: verified-working feature-space SupCon (no projection head)
#      python -u pretrain_supcon.py \
#          --dataset=CINIC10 --loss=supcon --model=mobilenet \
#          --train_examples=2000 --seed=1 --epochs=300 \
#          --projection_dim=0 --temperature=0.07 --lr=0.5 \
#          2>&1 | tee /root/cinic_run/logs/00_pretrain_cinic_no_proj.txt
#
# WARNING: at 300 epochs and 15 runs per cell, this sweep takes considerably
#   longer than the SVHN 40-epoch version. Rough estimate on a single 4090:
#     - each downstream cell:  ~1.5-3 hours
#     - full 6-cell sweep:     ~10-18 hours
#   Run inside tmux and consider attaching an auto-stop (see end of file).
# ============================================================================

set -e
set -o pipefail  # make `python ... | tee ...` propagate python's exit status
cd /workspace/memory-wrap-SENN/paper

ENC=models/CINIC10/supcon/mobilenet/2000/1.pt
LOG=/root/cinic_run/logs
YAML=config/train.yaml

mkdir -p "$LOG"

# Verify encoder exists before launching downstream sweep
if [ ! -f "$ENC" ]; then
    echo "ERROR: pretrained CINIC10 encoder not found at $ENC." >&2
    echo "Run pretrain_supcon.py first (see header of this script)." >&2
    exit 1
fi

# Flip yaml dataset_name to CINIC10 (back up original so we can restore on exit)
cp "$YAML" "${YAML}.bak"
sed -i 's/^dataset_name:.*/dataset_name: CINIC10/' "$YAML"
echo "Set $YAML -> dataset_name: CINIC10"
grep "^dataset_name:" "$YAML"

# Restore yaml on exit (success, failure, or Ctrl-C)
restore_yaml() {
    if [ -f "${YAML}.bak" ]; then
        mv "${YAML}.bak" "$YAML"
        echo "Restored $YAML from backup."
    fi
}
trap restore_yaml EXIT

echo "===== Cell 1: Scratch + Linear ====="
python -u train.py --modality=std \
    2>&1 | tee $LOG/01_cinic_scratch_linear.txt

echo "===== Cell 2: Scratch + MW (skip if already done) ====="
python -u train.py --modality=encoder_memory \
    2>&1 | tee $LOG/02_cinic_scratch_mw.txt

echo "===== Cell 3: SupCon + Linear (frozen) ====="
python -u train.py --modality=std --pretrained_encoder=$ENC --freeze_encoder=True \
    2>&1 | tee $LOG/03_cinic_supcon_linear_frozen.txt

echo "===== Cell 4: SupCon + Linear (fine-tune) ====="
python -u train.py --modality=std --pretrained_encoder=$ENC \
    2>&1 | tee $LOG/04_cinic_supcon_linear_finetune.txt

echo "===== Cell 5: SupCon + MW (frozen) ====="
python -u train.py --modality=encoder_memory --pretrained_encoder=$ENC --freeze_encoder=True \
    2>&1 | tee $LOG/05_cinic_supcon_mw_frozen.txt

echo "===== Cell 6: SupCon + MW (fine-tune) ====="
python -u train.py --modality=encoder_memory --pretrained_encoder=$ENC \
    2>&1 | tee $LOG/06_cinic_supcon_mw_finetune.txt

echo "===== ALL DONE ====="

# ----------------------------------------------------------------------------
# Optional: auto-stop the RunPod pod when the sweep finishes, so you stop
# being billed for idle GPU time while you sleep. Uncomment if desired.
#
# runpodctl stop pod "$RUNPOD_POD_ID"
# ----------------------------------------------------------------------------
