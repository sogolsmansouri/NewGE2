#!/bin/bash
#SBATCH --job-name=ge2_fb_inventory
#SBATCH --partition=a6000
#SBATCH --nodelist=c31
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=00:05:00
#SBATCH --output=ge2_fb_inventory_%j.out
#SBATCH --error=ge2_fb_inventory_%j.err
set -eu
hostname
nvidia-smi
df -h /mnt/local "$HOME"
find /mnt/local/smansou2 -maxdepth 8 -type f -name dataset.yaml -print
find /mnt/local/smansou2 -maxdepth 5 -path '*/bin/python' -print
