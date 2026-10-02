# Shared-context prefill switch for multi-question requests (2026-10-02)

Worker cba71646, `track=shared-ctx`; branch `xunzhuo/decision-2-shared-ctx` (from the integration branch). User
decision (COORDINATION 2026-10-02 18:50): an opt-in switch that computes the shared input of a multi-question System
One request once; off is today's exact path and the default. Nothing was uploaded and no released package changed;
shipping comes later through the parity rollout, with the switch off. Measured results (speed per N and size,
decision changes and accuracy on the four scored panels, the private multi-question sample and the private Index
delta) are in the private report. Times are UTC.

## Why

The runtime puts every question of a request in its own row, `[state + question + options]`, and runs the rows as
one padded batch, so a request with N questions recomputes the shared state N times and its latency grows linearly
in N. The open-jev-fast study's prefix-state hand-off prototype (`xunzhuo/decision-2-ojf-study`) showed the
approach works but was slower at small sizes (two eager forwards, per-question gathers).

## API

```python
model = Decision2.from_pretrained(package_dir)                     # switch off (default): exact path
model.system_one(state=..., questions=..., share_context=True)    # this request shares its prefix
model = Decision2.from_pretrained(package_dir, share_context=True) # runtime default; a request can pass False
model.system_one(state=..., questions=..., share_context={"tau": 0.01, "align": 64})  # explicit policy
```

`share_context` is `None` (use the runtime default), `False`, `True` (the default policy) or a
`shared_ctx.SharePolicy` / dict of its fields. The response schema is unchanged (`model`, `answers`, `usage`;
`usage.input_tokens` still counts every question's full prompt). Profiles without the switch (`kai-native`) answer
exactly whatever it says. `backend.share_stats` describes the last request (shared or not and why, prefix tokens,
mode, answers re-scored).

| Policy field | Default | Meaning |
| --- | --- | --- |
| `mode` | `"tree"` | `tree`: one packed forward; `cache`: prefix with a cache, then the suffixes as a padded batch (also the fallback when the packed row exceeds the forward token budget) |
| `min_questions` | 2 | fewer questions run exactly |
| `min_shared_tokens` | `None` | shared tokens a request must save, `(questions - 1) * prefix`; `None` takes the backbone's measured break-even (`auto_shared_tokens`) |
| `align` | 1 | the prefix length is a multiple of it (64 aligns the split with the attention key blocks and gated-delta chunks) |
| `tau` | 0.0 | a request with an answer whose top-two probability margin (Noul `\|2p − 1\|`) is below it falls back to the exact path |
| `fallback` | `"request"` | `request`: the whole request runs exactly (its answers are the exact path's); `rows`: only those answers are re-scored in their exact-path batch shape (cheaper, but a re-scored near-tie can flip again) |
| `max_buckets`, `bucket_tokens` | 1, 2048 | `cache` mode only: split suffix rows by length |

## Design (`v2/release/runtime/shared_ctx.py`; hooks in `qwen.py` and `api.py`)

- **Shared prefix.** The longest token prefix of all encoded questions, cut before the first option endpoint (the
  heads read option endpoints and the query token, which stay in the suffix), rounded down to `align`.
- **Tokenization once.** A request's encodings reuse the tokens of the shared context (`PrefixTokenizer`): the cut is
  a token and pre-token boundary with ASCII on both sides and no added-token text near it, and the first reuse is
  checked against a full tokenization (any mismatch turns reuse off). Equal to the tokenizer on 17,928 encodings of
  panel states, adversarial tails and random Unicode for both tokenizer families.
- **Tree mode.** The prefix and every suffix run as one packed row with the exact path's positions. Full-attention
  layers: the prefix attends causally to itself; every suffix query attends to the prefix keys (one kernel for all
  queries) and to its own suffix (one causal kernel over the padded suffix rows), and the two results are merged by
  their log-sum-exp (`softmax` over the union). Gated-delta layers: the prefix runs from a zero state; the suffixes run
  as variable-length sequences (FLA `cu_seqlens`) from the prefix-end recurrent state, each convolution window
  starting with the prefix's last inputs. Transformers' attention interface carries the tree to the attention layers;
  the gated-delta modules' forwards are swapped only for the call. The model's own forward and head run unchanged on
  the suffix rows, so every head variant (shared, typed, residual, label-token) works.
- **Cache mode.** The prefix runs once with a Transformers cache; a read-only suffix cache hands every row the prefix
  keys / values (cast once to the SDPA dtype) and the prefix-end conv / recurrent states, storing nothing the suffix
  computes, so memory stays one layer at a time.
- **Exactness tricks kept.** Exact positions; the exact path's mask regime for the prefix (explicit mask when the
  exact batch is padded); BF16 casts where autocast casts anyway; FP32 log-sum-exp merge; host-sync-free gathers.
- **Fallback.** `fallback="request"` (default when `tau` > 0) runs a request whose answers include a near-tie on
  the exact path, so its answers are the exact path's bit for bit. `fallback="rows"` (`exact_rows`) re-scores only
  those answers in their exact-path micro-batch's padded length and mask regime; that is not bit-identical to the
  full exact batch, because the exact path itself is not batch-invariant (GEMM kernels depend on the row count).
- **Fast path (phase A).** Tree mode runs the original decoder-layer forwards while it runs (the fused forwards read
  tensor masks only); HIP graphs fall back to eager for the shared calls. Checked on a trial merge with
  `xunzhuo/decision-2-runtime-a` (`xunzhuo/decision-2-shared-ctx-on-runtime-a-trial`, not for integration).

## Results (summary; numbers in the private report)

Measured on one MI325X with the four released sizes (Kai 0.6B, Eos 0.8B, Nox 4B, Vega 27B), 7.4 GPU-h:

- **Switch off = today's path, bit for bit:** the released runtime and this branch's runtime (switch off, phase A
  fast path merged) give byte-identical answers on every scored-panel prompt (typed FINAL, CSS15, public 231,
  mlx-diag): 10,653 / 10,653 prompts for Kai, Eos and Nox; 1,731 / 1,731 for Vega (500 per panel).
- **Speed:** latency grows far more slowly with the number of questions; several-fold faster at 64–128 questions
  for every size, with or without the phase A fast path on the exact side. Below a backbone's break-even (small
  backbones, few questions, short shared context) the default policy keeps the exact path.
- **Accuracy:** decisions that change are near-ties, about as many as asking each question alone instead of in one
  batch changes on the exact path; accuracy deltas on the four panels and on the private evaluation are
  indistinguishable from zero.
- **Trade-off levers:** exactness tricks are built in (aligning the prefix to 64 does not help consistently);
  `tau` with the whole-request fallback trades speed for exact-path answers monotonically; re-scoring only the
  low-margin answers does not reduce changes; break-even thresholds are measured per backbone, eager and with HIP
  graphs.
- **Recommended default:** `share_context=True` (tree mode, align 1, tau 0, automatic break-even); strict profile
  `{"tau": 0.005}`.

## Tests

`v2/release/tests/test_shared_ctx.py` (CPU, FP32, tiny random Qwen3 and Qwen3.5 decision models and a trained
byte-level BPE with Qwen2's pre-tokenizer): with the switch off the runtime never imports the module and runs the
batches it always ran; on (tree and cache) equals the exact path within float rounding at 2–16 questions; runtime
default and per-request override; single question and invalid questions; long prefixes and the break-even
threshold; the over-budget fallback to cache mode; both fallbacks; policy parsing; the shared prefix; buckets; the
prefix tokenizer; the break-even thresholds. The builder ships `shared_ctx.py` with the Qwen runtime
(`test_vendor_source`). 92 release runtime tests pass on the branch merged with phase A.
