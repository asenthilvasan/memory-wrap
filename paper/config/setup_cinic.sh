#!/bin/bash
set -e

# === 1. Dataset on fast storage (RAM disk) ===
if [ ! -d /dev/shm/CINIC10/train ]; then
    df -h /dev/shm
    # If too small, uncomment the next line
    # mount -o remount,size=8G /dev/shm
    mkdir -p /dev/shm/CINIC10
    cd /dev/shm/CINIC10
    wget -q https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
    tar xzf CINIC-10.tar.gz
    rm CINIC-10.tar.gz
fi

# === 2. Outputs on fast NVMe (persists during pod uptime) ===
mkdir -p /root/cinic_run/models /root/cinic_run/images /root/cinic_run/logs

# === 3. Symlinks into the codebase's expected paths ===
cd /workspace/memory-wrap-SENN/paper

# Dataset
mkdir -p datasets
rm -rf datasets/CINIC10
ln -s /dev/shm/CINIC10 datasets/CINIC10

# Models output
mkdir -p models
rm -rf models/CINIC10
ln -s /root/cinic_run/models models/CINIC10

# Memory image output
mkdir -p images/mem_images
rm -rf images/mem_images/CINIC10
ln -s /root/cinic_run/images images/mem_images/CINIC10

# === 4. Verify ===
echo "--- Verification ---"
ls -L datasets/CINIC10/train | wc -l        # 10
ls -L datasets/CINIC10/train/airplane | wc -l # 9000
echo "Datasets:" && ls -lh datasets/CINIC10  | head -3
echo "Models out:" && ls -ld models/CINIC10
echo "Images out:" && ls -ld images/mem_images/CINIC10