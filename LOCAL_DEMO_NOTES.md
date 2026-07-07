# Local Demo Notes

Repository clone:

```bash
cd /u2/n262li/ocr/DeepSeek-OCR
```

Project-local environment:

```bash
conda activate /u2/n262li/ocr/.conda/deepseek-ocr
```

Run the Hugging Face sample-image demo on a GPU allocation:

```bash
sbatch scripts/run_hf_asset_demo.sbatch
```

Or run directly from an interactive GPU shell:

```bash
/u2/n262li/ocr/.conda/deepseek-ocr/bin/python scripts/run_hf_asset_demo.py \
  --image /u2/n262li/ocr/DeepSeek-OCR/assets/show1.jpg \
  --output /u2/n262li/ocr/DeepSeek-OCR/outputs/hf_show1
```

The helper defaults to `--attn-implementation eager` because this machine does not expose
`nvcc`/`CUDA_HOME` on the login shell, so `flash-attn==2.7.3` cannot be built there.
On a machine with `flash-attn` installed, add:

```bash
--attn-implementation flash_attention_2
```

The completed sample run wrote:

- `/u2/n262li/ocr/DeepSeek-OCR/outputs/hf_show1/result.mmd`
- `/u2/n262li/ocr/DeepSeek-OCR/outputs/hf_show1/result_with_boxes.jpg`
- cropped image regions under `/u2/n262li/ocr/DeepSeek-OCR/outputs/hf_show1/images/`
