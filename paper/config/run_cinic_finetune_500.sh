#!/bin/bash
# ============================================================================
# CINIC-10 SupCon fine-tune at train_examples=500. Runs only the two cells
# (4 and 6) intentionally skipped by run_cinic_ablation_500.sh. Encoder is
# UNFROZEN so the SupCon-pretrained features can co-adapt with the
# downstream head.
#
# Prereqs (already satisfied in this workspace):
#   - paper/models/CINIC10 -> /root/cinic_run/models (symlink)
#   - SupCon encoder at models/CINIC10/supcon/mobilenet/500/1.pt
#   - CINIC10 dataset visible at datasets/CINIC10/train (setup_cinic.sh)
#
# LR note: train.yaml ships with lr=1e-1 (from-scratch). For fine-tune we
# drop to 1e-2 so the SupCon initialization isn't wrecked in the first epoch.
# Override --learning_rate=... below if you want a different value.
# ============================================================================

set -e
set -o pipefail
cd /workspace/memory-wrap-SENN/paper

ENC=models/CINIC10/supcon/mobilenet/500/1.pt
LOG=/root/cinic_run_500/logs
YAML=config/train.yaml

mkdir -p "$LOG"

[ -f "$ENC" ] || { echo "ERROR: missing encoder $ENC" >&2; exit 1; }
[ -d datasets/CINIC10/train ] || { echo "ERROR: missing datasets/CINIC10/train; run config/setup_cinic.sh" >&2; exit 1; }

cp "$YAML" "${YAML}.bak"
trap 'mv "${YAML}.bak" "$YAML" 2>/dev/null || true; echo "Restored $YAML."' EXIT

sed -i 's/^dataset_name:.*/dataset_name: CINIC10/'    "$YAML"
sed -i 's/^train_examples:.*/train_examples: 500/'    "$YAML"
sed -i 's/^batch_size_train:.*/batch_size_train: 64/' "$YAML"
sed -i 's/^  learning_rate:.*/  learning_rate: 1e-2/' "$YAML"
# Fine-tune needs far fewer epochs than scratch: encoder adapts quickly and
# 300 epochs overfits 500 samples (loss hits near-zero by ~epoch 75).
# 100 epochs with milestones at [60, 80] gives a sensible LR schedule.
python - <<'PY'
import yaml, re, sys
with open("config/train.yaml") as f:
    raw = f.read()
raw = re.sub(r'(CINIC10:.*?opt_milestones\s*:)[^\n]*', r'\1 [60, 80]', raw, flags=re.DOTALL)
raw = re.sub(r'(CINIC10:.*?num_epochs\s*:)[^\n]*', r'\1 100', raw, flags=re.DOTALL)
with open("config/train.yaml", "w") as f:
    f.write(raw)
print("Set CINIC10 num_epochs=100, opt_milestones=[60,80]")
PY
echo "YAML patched:"
grep -E "^(dataset_name|train_examples|batch_size_train|  learning_rate):" "$YAML"
grep -A3 "^CINIC10:" "$YAML" | grep -E "num_epochs|opt_milestones"

echo "===== Cell 4: SupCon + Linear (fine-tune, encoder UNFROZEN) ====="
python -u train.py \
    --modality=std \
    --pretrained_encoder=$ENC \
    --freeze_encoder=False \
    2>&1 | tee $LOG/04_supcon_linear_finetune.txt

echo "===== Cell 6: SupCon + MW (fine-tune, encoder UNFROZEN) ====="
python -u train.py \
    --modality=encoder_memory \
    --pretrained_encoder=$ENC \
    --freeze_encoder=False \
    2>&1 | tee $LOG/06_supcon_mw_finetune.txt

echo "===== ALL DONE ====="
