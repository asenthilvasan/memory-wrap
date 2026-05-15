#!/bin/bash
set -e
cd /workspace/memory-wrap-SENN/paper

ENC=models/SVHN/supcon/mobilenet/2000/1.pt
LOG=/root/cinic_run/logs

# Verify encoder exists before launching downstream sweep
if [ ! -f "$ENC" ]; then
    echo "ERROR: pretrained encoder not found at $ENC. Run pretrain_supcon.py first." >&2
    exit 1
fi

echo "===== Cell 1: Scratch + Linear ====="
python -u train.py --modality=std \
    2>&1 | tee $LOG/01_scratch_linear.txt

echo "===== Cell 2: Scratch + MW (skip if already done) ====="
if [ ! -f models/SVHN/encoder_memory/mobilenet/2000/conf.p ]; then
    python -u train.py --modality=encoder_memory \
        2>&1 | tee $LOG/02_scratch_mw.txt
else
    echo "Already done, skipping."
fi

echo "===== Cell 3: SupCon + Linear (frozen) ====="
python -u train.py --modality=std --pretrained_encoder=$ENC --freeze_encoder=True \
    2>&1 | tee $LOG/03_supcon_linear_frozen.txt

echo "===== Cell 4: SupCon + Linear (fine-tune) ====="
python -u train.py --modality=std --pretrained_encoder=$ENC \
    2>&1 | tee $LOG/04_supcon_linear_finetune.txt

echo "===== Cell 5: SupCon + MW (frozen) ====="
python -u train.py --modality=encoder_memory --pretrained_encoder=$ENC --freeze_encoder=True \
    2>&1 | tee $LOG/05_supcon_mw_frozen.txt

echo "===== Cell 6: SupCon + MW (fine-tune) ====="
python -u train.py --modality=encoder_memory --pretrained_encoder=$ENC \
    2>&1 | tee $LOG/06_supcon_mw_finetune.txt

echo "===== ALL DONE ====="