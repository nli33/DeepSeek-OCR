#!/usr/bin/env python
"""M1 corpus pre-encoding: page images -> per-tier pooled-vector cache.

Encodes document page images from an HF image corpus (default
Tevatron/colpali-corpus, docid->image) with the DeepSeek-OCR vision encoder at a
chosen resolution tier, and caches one L2-normalized pooled vector per page.

This is the substrate for the tier sweep that produces the headline
tokens-vs-retrieval curve (RETRIEVAL_PLAN.md). Run once per tier.

Outputs under --output-root/<tier>/:
  vectors.npy   float32 [N, D]   L2-normalized pooled page vectors
  ids.json      [docid, ...]     row i of vectors <-> ids[i]
  manifest.json tier config, N, D, mean content-token count, provenance

By default only encodes the docids in --needed-docids (from prepare_colpali.py),
so a small subset stays small. Use --limit to cap, or --all to encode everything.
GPU required (use Slurm; see HANDOFF.md).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

from ds_encoder import TIERS, load_model, encode_page


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Encode a page-image corpus into a per-tier vector cache.")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-OCR")
    p.add_argument("--corpus", default="Tevatron/colpali-corpus")
    p.add_argument("--split", default="train")
    p.add_argument("--id-field", default="docid")
    p.add_argument("--image-field", default="image")
    p.add_argument("--tiers", nargs="+", default=["tiny", "small", "base", "large", "gundam"], choices=list(TIERS))
    p.add_argument("--needed-docids", default=str(repo_root / "outputs" / "colpali_subset" / "needed_docids.json"),
                   help="JSON list of docids to encode; omit/--all to encode the whole split.")
    p.add_argument("--all", action="store_true", help="Encode every doc in the split (ignore --needed-docids).")
    p.add_argument("--limit", type=int, default=None, help="Cap number of docs encoded (after filtering).")
    p.add_argument("--output-root", default=str(repo_root / "outputs" / "index" / "colpali"))
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--streaming", action="store_true",
                   help="Stream the corpus instead of a cached random-access load. Only sensible with "
                        "--all/--limit over the first rows; scattered --needed-docids should NOT stream.")
    return p.parse_args()


def iter_docs(args, needed):
    """Yield (docid, PIL image) for the docs to encode.

    Random-access (default): cached non-streaming load + docid->row map, so we
    touch only the needed docs (one-time full download, reused across tiers/runs).
    Streaming (--streaming): single forward pass; early-stops once all needed
    docids are seen (or --limit reached). Good only for contiguous head-of-corpus
    validation, not scattered ids.
    """
    if args.streaming:
        ds = load_dataset(args.corpus, split=args.split, streaming=True)
        found, n = set(), 0
        for row in ds:
            docid = str(row[args.id_field])
            if needed is not None and docid not in needed:
                continue
            yield docid, row[args.image_field].convert("RGB")
            n += 1
            if needed is not None:
                found.add(docid)
                if len(found) >= len(needed):
                    break
            if args.limit is not None and n >= args.limit:
                break
        return

    ds = load_dataset(args.corpus, split=args.split)  # cached, random access
    if needed is None:
        order = range(len(ds)) if args.limit is None else range(min(args.limit, len(ds)))
        for i in order:
            row = ds[int(i)]
            yield str(row[args.id_field]), row[args.image_field].convert("RGB")
    else:
        id_to_row = {str(d): i for i, d in enumerate(ds[args.id_field])}
        want = [d for d in needed if d in id_to_row]
        missing = len(needed) - len(want)
        if missing:
            print(f"WARNING: {missing} needed docids not in corpus", flush=True)
        if args.limit is not None:
            want = want[: args.limit]
        for d in want:
            row = ds[id_to_row[d]]
            yield d, row[args.image_field].convert("RGB")


def main():
    args = parse_args()
    tiers = {t: TIERS[t] for t in args.tiers}

    needed = None
    if not args.all and args.needed_docids and Path(args.needed_docids).exists():
        needed = set(json.loads(Path(args.needed_docids).read_text()))
        print(f"Filtering corpus to {len(needed)} needed docids.")
    else:
        print("Encoding the whole split (no docid filter).")

    model, _tok, mod = load_model(args.model, args.attn_implementation)

    # accumulate per tier
    vecs = {t: [] for t in tiers}
    ids = {t: [] for t in tiers}
    tok_counts = {t: [] for t in tiers}

    seen = 0
    t0 = time.time()
    for docid, image in iter_docs(args, needed):
        for tname, cfg in tiers.items():
            try:
                vec, _content, meta = encode_page(model, mod, image, cfg)
                vecs[tname].append(vec)
                ids[tname].append(docid)
                tok_counts[tname].append(meta["content_tokens"])
            except Exception as e:
                print(json.dumps(dict(docid=docid, tier=tname, ok=False, error=f"{type(e).__name__}: {e}")), flush=True)
        seen += 1
        if seen % args.log_every == 0:
            rate = seen / (time.time() - t0)
            print(f"encoded {seen} docs  ({rate:.1f} docs/s)", flush=True)

    out_root = Path(args.output_root)
    for tname, cfg in tiers.items():
        d = out_root / tname
        d.mkdir(parents=True, exist_ok=True)
        V = np.stack(vecs[tname]).astype(np.float32) if vecs[tname] else np.zeros((0, 0), np.float32)
        np.save(d / "vectors.npy", V)
        (d / "ids.json").write_text(json.dumps(ids[tname]))
        manifest = dict(corpus=args.corpus, split=args.split, tier=tname, tier_cfg=cfg,
                        num_docs=int(V.shape[0]), dim=int(V.shape[1]) if V.ndim == 2 else 0,
                        mean_content_tokens=float(np.mean(tok_counts[tname])) if tok_counts[tname] else 0.0,
                        model=args.model)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest))

    print(f"\nEncoded {seen} docs across {len(tiers)} tiers in {time.time() - t0:.0f}s -> {out_root}")


if __name__ == "__main__":
    main()
