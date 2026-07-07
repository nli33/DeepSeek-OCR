#!/usr/bin/env python
"""M0 encoder probe for the retrieval project (see RETRIEVAL_PLAN.md).

Taps DeepSeek-OCR's vision encoder (shared code in scripts/ds_encoder.py) and
reports, per (image, resolution-tier):
  - raw vision-token count the LLM would see (content + newline/separator rows)
  - content-token count (real page tokens, separators excluded)
  - the model's own `valid image tokens` count (the infer() formula)
  - a pooled + L2-normalized page vector (mean over content tokens)

Then prints a cosine-similarity matrix over all pooled vectors as a sanity check.
Per-token + pooled vectors saved as .npz. GPU required (use Slurm; see HANDOFF.md).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ds_encoder import TIERS, load_model, encode_page


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Probe DeepSeek-OCR vision-token embeddings per tier.")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-OCR")
    p.add_argument("--assets-dir", default=str(repo_root / "assets"))
    p.add_argument("--output-dir", default=str(repo_root / "outputs" / "probe"))
    p.add_argument("--tiers", nargs="+", default=list(TIERS), choices=list(TIERS))
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument("images", nargs="*", default=["show1.jpg", "show2.jpg", "show3.jpg", "show4.jpg"])
    return p.parse_args()


def main():
    args = parse_args()
    assets_dir = Path(args.assets_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _tok, mod = load_model(args.model, args.attn_implementation)

    labels, vectors, summary = [], [], []
    for image_name in args.images:
        image_path = assets_dir / image_name if not Path(image_name).is_absolute() else Path(image_name)
        image = Image.open(image_path).convert("RGB")
        for tier in args.tiers:
            cfg = TIERS[tier]
            label = f"{Path(image_name).stem}:{tier}"
            try:
                vec, content, meta = encode_page(model, mod, image, cfg)
                np.savez(out_dir / f"{label.replace(':', '_')}.npz", content_tokens=content.numpy(), pooled=vec)
                row = dict(label=label, image=image_name, tier=tier, expected=cfg["expected"],
                           model_valid_tokens=meta["valid_img_tokens"], content_tokens=meta["content_tokens"],
                           full_tokens=meta["full_tokens"], dim=meta["dim"], n_crops=meta["n_crops"],
                           ratio=round(meta["ratio"], 3), ok=True)
                labels.append(label)
                vectors.append(vec)
            except Exception as e:
                row = dict(label=label, image=image_name, tier=tier, ok=False, error=f"{type(e).__name__}: {e}")
            summary.append(row)
            print(json.dumps(row))

    sim = None
    if len(vectors) > 1:
        V = np.stack(vectors)
        sim = (V @ V.T).tolist()
        print("\n=== cosine similarity (pooled page vectors) ===")
        w = max(len(l) for l in labels) + 1
        print(" " * w + " ".join(f"{l[:7]:>7}" for l in labels))
        for i, l in enumerate(labels):
            print(f"{l:<{w}}" + " ".join(f"{sim[i][j]:7.3f}" for j in range(len(labels))))

    (out_dir / "summary.json").write_text(json.dumps(dict(summary=summary, labels=labels, cosine=sim), indent=2))
    print(f"\nSaved vectors + summary under {out_dir}")


if __name__ == "__main__":
    main()
