#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR on bundled demo images.")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--assets-dir", default=str(repo_root / "assets"))
    parser.add_argument("--output-root", default=str(repo_root / "outputs" / "hf_all_assets"))
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the model infer() default of 8192 generated tokens.",
    )
    parser.add_argument("images", nargs="*", default=["show1.jpg", "show2.jpg", "show3.jpg", "show4.jpg"])
    return parser.parse_args()


def main():
    args = parse_args()
    assets_dir = Path(args.assets_dir)
    output_root = Path(args.output_root)

    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-OCR inference needs a CUDA GPU visible to this process.")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        _attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)
    if args.max_new_tokens is not None:
        original_generate = model.generate

        def bounded_generate(*generate_args, **generate_kwargs):
            generate_kwargs["max_new_tokens"] = args.max_new_tokens
            return original_generate(*generate_args, **generate_kwargs)

        model.generate = bounded_generate

    for image_name in args.images:
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = assets_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        demo_name = image_path.stem
        output_path = output_root / demo_name
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n===== {demo_name}: {image_path} =====", flush=True)
        result = model.infer(
            tokenizer,
            prompt=args.prompt,
            image_file=str(image_path),
            output_path=str(output_path),
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=args.crop_mode,
            save_results=True,
            test_compress=True,
        )
        print(result)
        print(f"Saved outputs under {output_path}", flush=True)


if __name__ == "__main__":
    main()
