#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR on a sample image.")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--image", default=str(repo_root / "assets" / "show1.jpg"))
    parser.add_argument("--output", default=str(repo_root / "outputs" / "hf_show1"))
    parser.add_argument(
        "--prompt",
        default="<image>\n<|grounding|>Convert the document to markdown. ",
    )
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--crop-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Use flash_attention_2 on machines where flash-attn is installed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image = Path(args.image)
    output = Path(args.output)

    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DeepSeek-OCR inference needs a CUDA GPU. "
            "Run this script inside a GPU allocation, for example with scripts/run_hf_asset_demo.sbatch."
        )

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        _attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)

    result = model.infer(
        tokenizer,
        prompt=args.prompt,
        image_file=str(image),
        output_path=str(output),
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        save_results=True,
        test_compress=True,
    )
    print(result)
    print(f"Saved outputs under {output}")


if __name__ == "__main__":
    main()
