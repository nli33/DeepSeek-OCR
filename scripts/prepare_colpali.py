#!/usr/bin/env python
"""M1 data prep: build a small self-contained retrieval subset from Tevatron/colpali.

Picks the first --num-queries queries and collects the union of their positive +
negative document ids. That candidate-doc set is what build_corpus_index.py will
encode from Tevatron/colpali-corpus, so we never touch all 118k pages.

Writes to --output-dir:
  queries.jsonl     one row/query: {query_id, query_text, positives:[docid], negatives:[docid]}
  qrels.json        {query_id: {docid: 1}}  (relevance judgments)
  needed_docids.json  sorted list of unique candidate docids to encode
  meta.json         counts + provenance

CPU/network only; no GPU. Runs on the login node.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Prepare a small colpali retrieval subset.")
    p.add_argument("--dataset", default="Tevatron/colpali")
    p.add_argument("--split", default="train")
    p.add_argument("--num-queries", type=int, default=500)
    p.add_argument("--num-negatives", type=int, default=20,
                   help="Negatives per query to keep as candidates (<=20 available).")
    p.add_argument("--output-dir", default=str(repo_root / "outputs" / "colpali_subset"))
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    queries, qrels, needed = [], {}, set()
    for i, row in enumerate(ds):
        if i >= args.num_queries:
            break
        qid = str(row["query_id"])
        pos = [str(d) for d in row["positive_document_ids"]]
        neg = [str(d) for d in row["negative_document_ids"]][: args.num_negatives]
        queries.append(dict(query_id=qid, query_text=row["query_text"], positives=pos,
                            negatives=neg, answer=row.get("answer"), source=row.get("source")))
        qrels[qid] = {d: 1 for d in pos}
        needed.update(pos)
        needed.update(neg)

    with (out / "queries.jsonl").open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    (out / "qrels.json").write_text(json.dumps(qrels))
    needed_sorted = sorted(needed, key=lambda x: (len(x), x))
    (out / "needed_docids.json").write_text(json.dumps(needed_sorted))
    meta = dict(dataset=args.dataset, split=args.split, num_queries=len(queries),
                num_candidate_docs=len(needed_sorted), num_negatives_kept=args.num_negatives)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"Wrote subset to {out}")


if __name__ == "__main__":
    main()
