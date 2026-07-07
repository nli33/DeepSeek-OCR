#!/usr/bin/env bash
#SBATCH --job-name=deepseek-ocr-probe
#SBATCH --partition=JIMMY
#SBATCH --account=jimmy_group
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:jimmygpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/u2/n262li/ocr/DeepSeek-OCR/outputs/slurm-probe-%j.out
#SBATCH --error=/u2/n262li/ocr/DeepSeek-OCR/outputs/slurm-probe-%j.err

set -euo pipefail

cd /u2/n262li/ocr/DeepSeek-OCR
mkdir -p outputs

# M0 encoder probe: vision-token embeddings per resolution tier (see RETRIEVAL_PLAN.md).
/u2/n262li/ocr/.conda/deepseek-ocr/bin/python scripts/encode_page.py \
  --output-dir /u2/n262li/ocr/DeepSeek-OCR/outputs/probe
