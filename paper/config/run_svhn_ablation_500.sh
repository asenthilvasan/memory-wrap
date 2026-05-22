#!/bin/bash
# ============================================================================
# SVHN ablation sweep at train_examples=500 (lower-budget variant of
# run_svhn_ablation.sh's 2000-sample sweep).
#
# Cells run: 1 (Scratch+Linear), 2 (Scratch+MW), 3 (SupCon+Linear frozen),
# 5 (SupCon+MW frozen). Cells 4 and 6 (fine-tune) are intentionally skipped
# to save compute; the frozen cells alone answer "does SupCon's lift over MW
# survive at half the original 2000-budget data?".
#
# PREREQUISITES (run once before this script):
#   1. Pretrain a SupCon encoder on the SAME 500-image subset downstream
#      will see (same-data-budget). Batch 64 because batch 256 + drop_last
#      on 500 samples gives only 1 SGD step/epoch; batch 64 gives 7
#      steps/epoch, matching the 7 steps/epoch the 2000-budget recipe got
#      at batch 256, so total SGD updates are conserved at the original
#      280 (= 7 * 40).
#
#      LR follows the SupCon/SimCLR linear scaling rule (lr = base_lr *
#      batch / 256). The 2000-budget run used batch=256, lr=0.5, so at
#      batch=64 we use lr=0.125.
#
#      python -u pretrain_supcon.py \
#          --dataset=SVHN --loss=supcon --model=mobilenet \
#          --train_examples=500 --seed=1 \
#          --epochs=40 --batch_size=64 \
#          --lr=0.125 --temperature=0.07 --projection_dim=0 \
#          2>&1 | tee /root/svhn_run_500/logs/00_pretrain_svhn_500.txt
#
#      Output: models/SVHN/supcon/mobilenet/500/1.pt
# ============================================================================

set -e
set -o pipefail  # make `python ... | tee ...` propagate python's exit status
cd /workspace/memory-wrap-SENN/paper

ENC=models/SVHN/supcon/mobilenet/500/1.pt
LOG=/root/svhn_run_500/logs
YAML=config/train.yaml

mkdir -p "$LOG"

# Verify encoder exists before launching downstream sweep
if [ ! -f "$ENC" ]; then
    echo "ERROR: pretrained SVHN-500 encoder not found at $ENC." >&2
    echo "Run pretrain_supcon.py first (see header of this script)." >&2
    exit 1
fi

# Flip yaml: dataset_name=SVHN, train_examples=500. Back up original first.
cp "$YAML" "${YAML}.bak"
sed -i 's/^dataset_name:.*/dataset_name: SVHN/' "$YAML"
sed -i 's/^train_examples:.*/train_examples: 500/' "$YAML"
sed -i 's/^batch_size_train:.*/batch_size_train: 64/' "$YAML"
echo "Set $YAML -> dataset_name: SVHN, train_examples: 500, batch_size_train: 64"
grep -E "^(dataset_name|train_examples|batch_size_train):" "$YAML"

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
    2>&1 | tee $LOG/01_scratch_linear.txt

echo "===== Cell 2: Scratch + MW ====="
python -u train.py --modality=encoder_memory \
    2>&1 | tee $LOG/02_scratch_mw.txt

echo "===== Cell 3: SupCon + Linear (frozen) ====="
python -u train.py --modality=std --pretrained_encoder=$ENC --freeze_encoder=True \
    2>&1 | tee $LOG/03_supcon_linear_frozen.txt

# Cell 4 (SupCon + Linear fine-tune) intentionally skipped.

echo "===== Cell 5: SupCon + MW (frozen) ====="
python -u train.py --modality=encoder_memory --pretrained_encoder=$ENC --freeze_encoder=True \
    2>&1 | tee $LOG/05_supcon_mw_frozen.txt

# Cell 6 (SupCon + MW fine-tune) intentionally skipped.

echo "===== ALL DONE ====="
