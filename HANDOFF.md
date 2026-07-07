# HANDOFF

## Project Overview

DeepSeek-OCR is a research/demo repository for OCR and document parsing with the released `deepseek-ai/DeepSeek-OCR` Hugging Face model. It converts images/PDFs into compact visual tokens, feeds them to an LLM decoder with a prompt, and emits markdown-like OCR/layout output plus optional bounding-box visualizations and cropped image regions.

Main stack:

- Python 3.12.9 in a project-local conda env: `/u2/n262li/ocr/.conda/deepseek-ocr`
- PyTorch `2.6.0+cu118`, Transformers `4.46.3`, tokenizers `0.20.3`
- Optional/upstream vLLM path expects CUDA 11.8, vLLM 0.8.5, and `flash-attn==2.7.3`
- Slurm is used on this cluster for GPU execution

Important paths:

- `README.md`: upstream setup/inference instructions.
- `requirements.txt`: upstream Python dependency pins for the Transformers path.
- `DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py`: upstream one-image Hugging Face demo; hard-coded placeholder paths.
- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/`: upstream vLLM implementation and scripts.
- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`: vLLM input/output/model/prompt settings; currently has empty `INPUT_PATH` and `OUTPUT_PATH`.
- `assets/show1.jpg` ... `assets/show4.jpg`: bundled demo collage images.
- `scripts/run_hf_asset_demo.py`: local helper for one HF demo image.
- `scripts/run_hf_all_asset_demos.py`: local helper that loads the HF model once and runs multiple assets; supports `--max-new-tokens`.
- `scripts/*.sbatch`: local Slurm wrappers for the helper scripts.
- `outputs/`: generated demo outputs and Slurm logs; do not treat as source.
- `LOCAL_DEMO_NOTES.md`: short local run notes from prior work.

## Current State

Working:

- Repository is cloned at `/u2/n262li/ocr/DeepSeek-OCR`.
- Project-local conda env exists and imports PyTorch/Transformers successfully.
- `python -m py_compile scripts/run_hf_asset_demo.py scripts/run_hf_all_asset_demos.py` passes.
- Hugging Face inference completed end-to-end for bundled `show1` through `show4` via Slurm.
- Verified outputs are under `outputs/hf_all_assets/show*/` with `result.mmd`, `result_with_boxes.jpg`, and cropped `images/`.

Observed demo token summaries from `outputs/slurm-all-demos-1464334.out`:

| Demo | Visual tokens | Output text tokens | Ratio |
| --- | ---: | ---: | ---: |
| `show1` | 625 | 1414 | 2.26 |
| `show2` | 802 | 1096 | 1.37 |
| `show3` | 802 | 3937 | 4.91 |
| `show4` | 650 | 1128 | 1.74 |

Quality notes:

- `show1`, `show2`, and `show4` demonstrate the pipeline successfully, but these images are collages from the README, so the model also OCRs embedded prompt/result panels.
- `show3` is a weak qualitative result: it repeats author-bio text and shows the model's long-generation failure mode on a collage.
- The repo does not contain a full benchmark dataset. Benchmark-style runs require external datasets such as Fox or OmniDocBench.

Incomplete/risky:

- vLLM path has not been reproduced locally. Upstream requires a vLLM 0.8.5 CUDA 11.8 wheel and `flash-attn`.
- `flash-attn==2.7.3` failed to build on the login/non-toolkit shell because `nvcc`/`CUDA_HOME` were unavailable.
- Direct Python inference in the current shell reported `torch.cuda.is_available() == False`; Slurm batch jobs did get GPU access.
- `outputs/` and `scripts/__pycache__/` are untracked/generated.

## Setup and Commands

From repo root:

```bash
cd /u2/n262li/ocr/DeepSeek-OCR
```

Use the existing local env:

```bash
conda activate /u2/n262li/ocr/.conda/deepseek-ocr
```

The env was created with:

```bash
conda create -p /u2/n262li/ocr/.conda/deepseek-ocr python=3.12.9 -y
/u2/n262li/ocr/.conda/deepseek-ocr/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
/u2/n262li/ocr/.conda/deepseek-ocr/bin/python -m pip install -r requirements.txt
```

Run one bundled demo via Slurm:

```bash
sbatch scripts/run_hf_asset_demo.sbatch
```

Run all bundled demos via Slurm:

```bash
sbatch scripts/run_hf_all_asset_demos.sbatch
```

Run selected demos with bounded generation via Slurm:

```bash
sbatch scripts/run_hf_remaining_asset_demos.sbatch
```

Run directly only if the current process can see a CUDA GPU:

```bash
/u2/n262li/ocr/.conda/deepseek-ocr/bin/python scripts/run_hf_all_asset_demos.py --max-new-tokens 2048 show3.jpg show4.jpg
```

Validation commands verified safe:

```bash
/u2/n262li/ocr/.conda/deepseek-ocr/bin/python -m py_compile scripts/run_hf_asset_demo.py scripts/run_hf_all_asset_demos.py
/u2/n262li/ocr/.conda/deepseek-ocr/bin/python - <<'PY'
import torch, transformers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('cuda_available', torch.cuda.is_available())
PY
```

No test, lint, format, type-check, build, migration, or package manifest was found beyond the Python scripts and `requirements.txt`.

## Architecture Notes

Transformers data flow:

1. A prompt such as `<image>\n<|grounding|>Convert the document to markdown. ` is tokenized.
2. The image is loaded by the remote Hugging Face model code (`trust_remote_code=True`).
3. With `crop_mode=True`, the image is encoded as a global/base view plus optional local crops.
4. Visual embeddings are projected to the LLM hidden size and inserted where the `<image>` token appears.
5. The LLM decoder generates text/layout tokens.
6. `model.infer(..., save_results=True, test_compress=True)` writes `result.mmd`, `result_with_boxes.jpg`, and cropped regions.

Visual-token details from the remote model code:

- Base 1024 view contributes up to 256 visual-token vectors.
- Each 640 crop contributes 100 visual-token vectors.
- Printed `valid image tokens` can differ from raw `BASE + PATCHES` count because the code scales the base count by image aspect ratio.
- Text tokens are output tokenizer units generated by the LLM; they are not one-to-one with visual tokens.

Local helper behavior:

- `scripts/run_hf_all_asset_demos.py` uses `_attn_implementation="eager"` by default because `flash-attn` is not installed here.
- `--max-new-tokens` monkey-patches `model.generate` in the loaded model instance so runaway collage generations can be bounded without editing the Hugging Face cache.
- Slurm scripts are cluster-specific: partition `JIMMY`, account `jimmy_group`, GRES `gpu:jimmygpu:1`.

External services/APIs:

- First model load downloads model code/weights from Hugging Face: `deepseek-ai/DeepSeek-OCR`.
- No repo secrets or environment variable values are required. `CUDA_VISIBLE_DEVICES` is set to `0` by helper scripts if unset.

## Recent Work and Next Steps

Recent local changes, all untracked as of the last `git status --short`:

- Added local demo runners under `scripts/`.
- Added `LOCAL_DEMO_NOTES.md`.
- Generated demo outputs under `outputs/`.
- Added this `HANDOFF.md`.

Likely next tasks:

1. Decide whether to keep local helper scripts in version control; if so, add an ignore rule for `outputs/` and `__pycache__/`.
2. Run on clean, non-collage input documents to judge OCR quality more fairly.
3. If vLLM performance matters, set up a CUDA-toolkit environment with vLLM/flash-attn and test `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py`.
4. For benchmark reproduction, obtain external datasets and adapt `run_dpsk_ocr_eval_batch.py`; this repo does not ship a dataset.
5. Consider adding a small README section or script comments explaining visual-token and output-token counts.

Files to inspect first:

- `README.md`
- `LOCAL_DEMO_NOTES.md`
- `scripts/run_hf_all_asset_demos.py`
- `outputs/slurm-all-demos-1464334.out`
- `outputs/hf_all_assets/show*/result.mmd`
- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`

## Warnings for the Next Agent

- Do not assume the current shell has GPU access. Check `torch.cuda.is_available()` before direct runs; use Slurm if false.
- Do not edit files in `/u2/n262li/.cache/huggingface/...` manually. Those are downloaded remote-code cache files.
- Do not treat `outputs/` as source; it is generated and can be large/noisy.
- `show*.jpg` are demo collages, not clean OCR benchmark pages. Prompt artifacts and repeated text in `.mmd` are expected failure/noise modes.
- `flash-attn` needs CUDA build tooling. Installing it on a shell without `nvcc`/`CUDA_HOME` will fail.
- vLLM scripts rely on `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`; set `INPUT_PATH` and `OUTPUT_PATH` before running them.
- The model is loaded with `trust_remote_code=True`; pin a model revision if reproducibility/security review matters.
