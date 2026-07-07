#!/usr/bin/env bash
#SBATCH --job-name=deepseek-ocr-demo
#SBATCH --partition=JIMMY
#SBATCH --account=jimmy_group
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:jimmygpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/u2/n262li/ocr/DeepSeek-OCR/outputs/slurm-%j.out
#SBATCH --error=/u2/n262li/ocr/DeepSeek-OCR/outputs/slurm-%j.err

set -euo pipefail

cd /u2/n262li/ocr/DeepSeek-OCR
mkdir -p outputs

/u2/n262li/ocr/.conda/deepseek-ocr/bin/python scripts/run_hf_asset_demo.py \
  --image /u2/n262li/ocr/DeepSeek-OCR/assets/show1.jpg \
  --output /u2/n262li/ocr/DeepSeek-OCR/outputs/hf_show1
