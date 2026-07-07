#!/usr/bin/env bash
#SBATCH --job-name=deepseek-ocr-remaining
#SBATCH --partition=JIMMY
#SBATCH --account=jimmy_group
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:jimmygpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/u2/n262li/ocr/DeepSeek-OCR/outputs/slurm-remaining-demos-%j.out
#SBATCH --error=/u2/n262li/ocr/DeepSeek-OCR/outputs/slurm-remaining-demos-%j.err

set -euo pipefail

cd /u2/n262li/ocr/DeepSeek-OCR
mkdir -p outputs

/u2/n262li/ocr/.conda/deepseek-ocr/bin/python scripts/run_hf_all_asset_demos.py \
  --output-root /u2/n262li/ocr/DeepSeek-OCR/outputs/hf_remaining_assets \
  --max-new-tokens 2048 \
  show3.jpg show4.jpg
