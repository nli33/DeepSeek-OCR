#!/usr/bin/env python
"""M1 retrieval-eval harness (see RETRIEVAL_PLAN.md).

Given a per-tier corpus vector cache (from build_corpus_index.py) and query
vectors, computes nDCG@10, Recall@{1,5,10}, MRR@10 against qrels. This is the
scoring code that turns encoded tiers into the tokens-vs-retrieval curve.

The query tower is trained in M2; until then this runs a --self-retrieval sanity
mode (pseudo-query = each doc's own vector; a correct index must rank it #1) to
validate the metric code and the cache end-to-end.

CPU only (vectors are precomputed). No GPU needed.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_index(tier_dir):
    V = np.load(Path(tier_dir) / "vectors.npy")
    ids = json.loads((Path(tier_dir) / "ids.json").read_text())
    return V, ids


def dcg(rels):
    return float(np.sum((2 ** np.asarray(rels) - 1) / np.log2(np.arange(2, len(rels) + 2))))


def eval_queries(qvecs, qids, V, ids, qrels, ks=(1, 5, 10), ndcg_k=10):
    """qvecs [Q,D] vs corpus V [N,D]; both L2-normalized. Returns metric dict."""
    id_to_row = {d: i for i, d in enumerate(ids)}
    sims = qvecs @ V.T  # cosine (normalized)
    order = np.argsort(-sims, axis=1)
    recall = {k: [] for k in ks}
    mrr, ndcg, n_eval = [], [], 0
    for qi, qid in enumerate(qids):
        rel = qrels.get(qid, {})
        gold = {d for d, r in rel.items() if r > 0 and d in id_to_row}
        if not gold:
            continue
        n_eval += 1
        ranked = [ids[r] for r in order[qi]]
        for k in ks:
            recall[k].append(len(gold & set(ranked[:k])) / len(gold))
        rr = 0.0
        for rank, d in enumerate(ranked[: max(ks)], start=1):
            if d in gold:
                rr = 1.0 / rank
                break
        mrr.append(rr)
        gains = [rel.get(d, 0) for d in ranked[:ndcg_k]]
        ideal = sorted(rel.values(), reverse=True)[:ndcg_k]
        ndcg.append(dcg(gains) / dcg(ideal) if ideal and dcg(ideal) > 0 else 0.0)
    out = {f"recall@{k}": float(np.mean(recall[k])) if recall[k] else 0.0 for k in ks}
    out[f"ndcg@{ndcg_k}"] = float(np.mean(ndcg)) if ndcg else 0.0
    out["mrr@10"] = float(np.mean(mrr)) if mrr else 0.0
    out["num_queries_evaluated"] = n_eval
    return out


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Evaluate retrieval over per-tier corpus caches.")
    p.add_argument("--index-root", default=str(repo_root / "outputs" / "index" / "colpali"))
    p.add_argument("--tiers", nargs="+", default=None, help="Default: every tier dir found under index-root.")
    p.add_argument("--qrels", default=str(repo_root / "outputs" / "colpali_subset" / "qrels.json"))
    p.add_argument("--query-vectors", default=None,
                   help="npz with arrays 'qids' (str) and 'vectors' [Q,D]. Omit to use --self-retrieval.")
    p.add_argument("--self-retrieval", action="store_true",
                   help="Sanity: pseudo-query = each doc vector; expects recall@1==1.0.")
    p.add_argument("--out", default=str(repo_root / "outputs" / "index" / "colpali" / "eval.json"))
    return p.parse_args()


def main():
    args = parse_args()
    index_root = Path(args.index_root)
    tiers = args.tiers or sorted(d.name for d in index_root.iterdir() if (d / "vectors.npy").exists())

    results = {}
    for tier in tiers:
        V, ids = load_index(index_root / tier)
        if V.size == 0:
            results[tier] = dict(error="empty index")
            continue
        if args.self_retrieval or not args.query_vectors:
            qvecs, qids = V, ids
            qrels = {d: {d: 1} for d in ids}  # each doc is its own gold
            mode = "self-retrieval"
        else:
            z = np.load(args.query_vectors, allow_pickle=True)
            qvecs, qids = z["vectors"].astype(np.float32), [str(x) for x in z["qids"]]
            qrels = json.loads(Path(args.qrels).read_text())
            mode = "query-vectors"
        m = eval_queries(qvecs.astype(np.float32), qids, V.astype(np.float32), ids, qrels)
        m["mode"] = mode
        m["num_docs"] = int(V.shape[0])
        m["mean_content_tokens"] = json.loads((index_root / tier / "manifest.json").read_text()).get("mean_content_tokens")
        results[tier] = m
        print(json.dumps({tier: m}))

    Path(args.out).write_text(json.dumps(results, indent=2))
    # token-vs-performance table (the curve, as text)
    print("\n=== tokens vs retrieval ===")
    print(f"{'tier':<8}{'tokens':>8}{'ndcg@10':>10}{'recall@1':>10}{'recall@5':>10}{'mrr@10':>9}")
    for tier in tiers:
        m = results.get(tier, {})
        if "ndcg@10" in m:
            print(f"{tier:<8}{m['mean_content_tokens'] or 0:>8.0f}{m['ndcg@10']:>10.3f}"
                  f"{m['recall@1']:>10.3f}{m['recall@5']:>10.3f}{m['mrr@10']:>9.3f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
