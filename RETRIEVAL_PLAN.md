# Project Plan: DeepSeek-OCR as a Document Retriever — Optical Compression vs. Retrieval Performance

## 1. Motivation & Thesis

DeepSeek-OCR compresses a document page into a small, **tunable** number of vision
tokens (64 → 400 depending on resolution tier) before its decoder expands them back
to text. That per-page token budget is an explicit **optical-compression knob** that
existing visual-document retrievers do not expose.

This project repurposes the DeepSeek-OCR encoder as a **retrieval encoder**: a PDF
page goes in, one (or a set of) dense vector(s) come out, and we retrieve pages by
query–page similarity — no OCR text, no parsing. The training recipe follows
**DSE — "Unifying Multimodal Retrieval via Document Screenshot Embedding"**
(Ma et al., EMNLP 2024, `aclanthology.org/2024.emnlp-main.373`): a contrastively
trained bi-encoder over document screenshots.

**Central hypothesis.** Retrieval is a *coarser* task than exact OCR. Ranking a page
against a query needs enough signal to discriminate, not enough to transcribe every
glyph. So retrieval quality should degrade **more gracefully** under optical
compression than OCR fidelity does. If true, we can index at ~100 tokens/page and
retain most of the accuracy of a ~1000-token/page encoder (DSE, ColPali) at roughly
an order of magnitude smaller index and faster encoding.

**Headline deliverable.** A curve of **retrieval performance (nDCG@10 / Recall@k)
vs. vision tokens per page**, swept across DeepSeek-OCR's resolution tiers, with
index size and encode latency on secondary axes. This curve *is* the answer to "how
does optical compression affect retrieval."

## 2. Research Questions

- **RQ1 (the curve).** How does retrieval quality vary with vision-token budget
  (64 / 100 / 256 / 400 / multi-crop)? Where is the knee of the curve?
- **RQ2 (optical vs. text pipeline).** Does keeping the page in the compressed
  *optical* space beat decoding it to OCR text and embedding that text — using the
  **same** DeepSeek-OCR backbone on both sides?
- **RQ3 (single- vs. multi-vector).** Does ColPali-style late interaction over the
  compressed tokens recover accuracy that single-vector pooling loses, and at what
  index-size cost?
- **RQ4 (where to tap).** Encoder-only tap (compressed vision tokens) vs. decoder
  last-hidden-state tap (DSE-faithful, contextualized but more expensive)?
- **RQ5 (alignment gap).** How much does contrastive fine-tuning matter vs. frozen
  features? (ablation)

## 3. Background Anchors

### DSE (the training template)
Bi-encoder. A document **screenshot** is encoded by a vision-language model
(Phi-3-vision) and pooled to one dense vector; the text query goes through the same
LM to a vector in the same space. Trained with **InfoNCE / in-batch negatives**,
cosine similarity, DPR-style. Corpus: **Wiki-SS** (1.3M Wikipedia screenshots),
evaluated on **NQ** + slide retrieval; beats BM25 and OCR-then-embed. We reuse its
objective, its data (for comparability), and its "pixels → vector" framing.

### DeepSeek-OCR (our backbone)
`DeepEncoder` = SAM (local/window attention) + a 16× convolutional compressor +
CLIP-large (global attention), producing the **compressed vision tokens**; then a
DeepSeek-3B-MoE decoder expands them to text. Compression happens **at the encoder,
before the decoder** — so for retrieval we pool the encoder output and **never run
the decoder** (running it per document would erase the efficiency win).

Resolution tiers (from `DeepSeek-OCR-vllm/config.py`):

| Tier   | base_size | image_size | crop_mode | Vision tokens/page |
| ------ | --------: | ---------: | :-------: | -----------------: |
| Tiny   | 512       | 512        | False     | 64                 |
| Small  | 640       | 640        | False     | 100                |
| Base   | 1024      | 1024       | False     | 256                |
| Large  | 1280      | 1280       | False     | 400                |
| Gundam | 1024      | 640        | True      | 256 + 100·n_crops  |

These five tiers are the x-axis of the headline curve.

## 4. System Design

### 4.1 Encoding a document
- A PDF is rasterized to **one image per page** (pdf2image / PyMuPDF, ~144–200 DPI).
- Retrieval granularity is **per page** (as in DSE and ColPali). A whole-document
  vector, if needed, is a pooled aggregate over page vectors — but page-level is the
  primary unit; report doc-level only as a secondary aggregation.

### 4.2 Two representation axes (prototype both)

**A. Where to tap**
- **Encoder tap (primary):** pool the `DeepEncoder`'s output vision tokens. Cheapest;
  the representation the compression story is about. Decoder skipped.
- **Decoder tap (DSE-faithful):** feed a short fixed prompt (or last position)
  through the MoE decoder, take the final hidden state. More contextualized, more
  expensive; used for RQ4.

**B. Single- vs. multi-vector**
- **Single-vector (DSE-style):** mean- or attention-pool → one vector/page →
  projection head → cosine ANN. Small index.
- **Multi-vector (ColPali/ColBERT-style):** keep per-token vectors, score by MaxSim
  late interaction. Index size ∝ tokens/page — so DeepSeek-OCR's compression yields a
  direct, large index shrink vs. ColPali's ~1030 tokens. This is the strongest angle
  for RQ3.

### 4.3 Query encoder
Text query → LM/text tower → projection head → vector in the shared space. Start with
the **shared-backbone** setup (query text through DeepSeek-OCR's own LM), matching
DSE; keep a dual-encoder-with-separate-text-tower variant as a fallback if joint
alignment is hard.

## 5. Training (mapped from DSE)

- **Objective:** InfoNCE with in-batch negatives + **mined hard negatives**
  (BM25 or first-stage retriever). Temperature-scaled cosine.
- **Heads:** small projection to retrieval dim — ~128 for late-interaction tokens,
  ~768–1024 for single-vector.
- **Parameter-efficient:** LoRA on the encoder (and decoder only if using the decoder
  tap); freeze the backbone otherwise. Keeps it tractable on the cluster GPUs.
- **Corpus pre-encoding:** encode the corpus once per tier, cache vectors to disk;
  training/eval then reads cached vectors. Encoding dominates cost, so cache
  aggressively.
- **The sweep (RQ1):** run the full train→eval loop **once per resolution tier**.
  For multi-vector, also sweep n_crops in Gundam mode.

## 6. Datasets

- **Wiki-SS + NQ** (from DSE) — for direct comparability to the reference paper.
- **ViDoRe** (ColPali's benchmark suite) — the standard for visual document
  retrieval; multi-domain (industrial PDFs, slides, tables, figures).
- (Optional) an in-house PDF set if a domain-specific story is wanted.

## 7. Baselines

Because DeepSeek-OCR also emits OCR text, several baselines come from one backbone —
a clean, controlled comparison:

| System                                   | Page representation        | Tests                    |
| ---------------------------------------- | -------------------------- | ------------------------ |
| BM25 on DeepSeek-OCR text                | sparse                     | lexical floor            |
| Dense text-embed on DeepSeek-OCR text    | dense text vector          | OCR-then-embed pipeline  |
| **DeepSeek-OCR optical vector (per tier)** | compressed vision tokens | **our method (RQ1/RQ2)** |
| DSE (Phi-3-vision)                        | ~1000 vision tokens        | screenshot-embed ref     |
| ColPali                                   | ~1030 tokens, late interaction | multi-vector visual ref |

The optical-vector vs. OCR-then-embed rows isolate RQ2 with the backbone held fixed.

## 8. Metrics & Deliverables

- **Retrieval quality:** nDCG@10, Recall@{1,5,10}, MRR.
- **Efficiency:** vision tokens/page, index size (bytes), corpus-encode throughput
  (pages/s), query latency.
- **Primary deliverable — the curve:** retrieval metric (y) vs. tokens/page (x) across
  tiers, one line per method (single-vector, multi-vector, decoder-tap), with a
  second panel of index-size / latency vs. tokens/page. Annotate the knee.
- **Secondary:** RQ2 bar chart (optical vs. OCR-then-embed), RQ5 frozen-vs-finetuned
  ablation, per-domain breakdown on ViDoRe.

## 9. Milestones

1. **M0 — Encoder probe. ✅ DONE** (`scripts/encode_page.py` + `.sbatch`, job 1468242).
   Taps the `DeepEncoder` at a selectable tier and dumps pooled + per-token vectors
   without running the decoder. Verified content-token counts match the table exactly
   (64/100/256/400; Gundam = 256 + 100·n_crops), all five tiers run (incl. Tiny/Small
   at 512/640), embedding dim = 1280. Findings: (a) `model_valid_tokens` < content
   tokens at Base/Large due to aspect-ratio scaling (`int(256·ratio)`); (b) untrained
   mean-pooled features are only weakly discriminative — cross-page cosine sits at
   0.6–0.95 and *tier* often dominates *content* (e.g. show3:tiny–show2:tiny = 0.955 >
   show3:tiny–show3:large = 0.811). This quantifies the alignment gap and confirms
   contrastive fine-tuning (M2/RQ5) is doing the real work. Vectors + cosine matrix in
   `outputs/probe/`.
2. **M1 — Data pipeline. ✅ DONE.** Small starter chosen over Wiki-SS (1.3M):
   **`Tevatron/colpali`** (118k text queries, 1 pos + 20 neg docids each) +
   **`Tevatron/colpali-corpus`** (118k page images, `docid`→`image`; also a 500-doc
   `test` split). Scripts (all reuse `scripts/ds_encoder.py`):
   - `scripts/prepare_colpali.py` — pick first N queries, collect their candidate
     docid union → `queries.jsonl`, `qrels.json`, `needed_docids.json` (CPU/login).
   - `scripts/build_corpus_index.py` (+`.sbatch`) — encode page images per tier →
     `vectors.npy` [N,1280] + `ids.json` + `manifest.json`. Random-access-by-docid
     via a cached (non-streaming) load so only needed pages are touched (one-time
     corpus download, reused across tiers/runs); `--streaming --all --limit` fast-path
     for head-of-corpus validation.
   - `scripts/eval_retrieval.py` — nDCG@10 / Recall@{1,5,10} / MRR@10 from a per-tier
     cache + query vectors; prints the tokens-vs-retrieval table (the curve). Includes
     a `--self-retrieval` sanity mode.
   Validated: 120 head docs × 5 tiers encoded in 69s (tiny 64 → gundam ~838 mean
   tokens); self-retrieval gives recall@1=1.0 on every tier, confirming the full
   encoder→cache→metrics path. Env: `datasets==5.0.0` added; `pymupdf` already present
   for PDF→page rasterization. **Open item:** the real curve needs the query tower
   (M2) — until then eval only runs in self-retrieval mode.
3. **M2 — Single-vector DSE-style training.** InfoNCE + hard negatives, LoRA; evaluate
   at Base tier; reproduce a DSE-like number as a smoke test.
4. **M3 — The tier sweep (RQ1).** Train/eval at all five tiers → **first version of the
   headline curve.**
5. **M4 — Multi-vector late interaction (RQ3)** + baselines (RQ2) + ablations (RQ4/RQ5).
6. **M5 — Writeup:** final curves, tables, analysis of where optical compression helps
   vs. hurts.

## 10. Risks & Notes

- **Alignment gap:** untuned DeepSeek-OCR features won't retrieve well zero-shot;
  the contrastive fine-tune does the real work. Report frozen-vs-finetuned (RQ5).
- **Don't decode:** the efficiency argument dies if the MoE decoder runs per document.
  Encoder tap by default.
- **GPU access:** direct runs need a visible CUDA GPU — use Slurm (see `HANDOFF.md`);
  the login shell reports `cuda_available == False`.
- **Compute:** the sweep is 5× the train/eval cost; pre-encode + cache the corpus, and
  LoRA everything to stay tractable.
- **Page vs. document granularity:** start page-level (DSE/ColPali standard); treat a
  single whole-PDF vector as a lossy secondary aggregation, not the primary unit.

## 11. Immediate Next Step

Prototype `scripts/encode_page.py` (M0) to extract encoder vision-token embeddings at
a selectable resolution tier and confirm the tokens/page counts in §3, before
committing to any training run.
