#!/usr/bin/env bash
#SBATCH --job-name=dsocr-index-dev
#SBATCH --partition=JIMMY
#SBATCH --account=jimmy_group
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:jimmygpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/u2/n262li/ocr/DeepSeek-OCR/outputs/slurm-index-dev-%j.out
#SBATCH --error=/u2/n262li/ocr/DeepSeek-OCR/outputs/slurm-index-dev-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

cd /u2/n262li/ocr/DeepSeek-OCR
mkdir -p outputs

# Cheap pipeline validation: encode the first 120 contiguous corpus pages (streaming
# head -> no scattered download), all tiers. Validates encoder -> cache -> eval.
/u2/n262li/ocr/.conda/deepseek-ocr/bin/python -u scripts/build_corpus_index.py \
  --streaming --all --limit 120 \
  --output-root outputs/index/colpali_dev \
  --log-every 20
