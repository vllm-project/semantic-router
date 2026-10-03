# Pure-Go keyword rules and model selectors against the bindings

The router's keyword rules (BM25, n-gram) and model selectors (KNN, KMeans,
SVM, MLP) moved from `nlp-binding`, `ml-binding` and candle to Go. This
record compares them with the bindings on the same inputs and states the
parity evidence.

- **Date:** 2026-10-04.
- **Machine:** node B, AMD EPYC 9575F, CPU only.
- **Commits:** Go side at `75c1aa5e0` (exact mirror). Bindings: `nlp-binding`
  built from the same mirror; `ml-binding` and candle from the fixture
  recording mirror `01430de48`.
- **Go measurements:** `GOMAXPROCS=1`, pinned to one core
  (`taskset -c 8`), median of three runs.
- **Binding measurements:**
  - Keyword rules: the bindings ran in the same benchmark binary, pinned to
    the same core. The side-by-side harness is the temporary one from
    `e85d51dbd`, added to a scratch copy of the mirror.
  - Selectors: the fixture recorder measured each binding's mean select
    latency over the same queries in a warm process, not pinned (KNN rebuilds
    its ball tree on every select).

## Parity

- **Keyword rules:** the pure-Go rules reproduce `nlp-binding` (bm25 2.3.2,
  ngrammatic 0.7.0) on 29 recorded cases: 58 rules over 7,714 texts (2,394
  BM25, 5,320 n-gram). Matched keywords, scores (float32-exact) and counts all
  agree; keywords with equal scores are compared as sets.
- **Snowball English stemmer:** matches a 4,955-word sample of the reference
  vocabulary output, and the full vocabulary when
  `SNOWBALL_ENGLISH_VOCABULARY` is set.
- **Selectors:** recorded binding selections match on:
  - 18 synthetic artifacts × 64 queries each (KNN at k 1, 3, 7 and 100 and
    with equal votes, KMeans, one-vs-one SVC (RBF and linear, binary and
    multi-class), legacy one-vs-rest, MLPs including argmax ties);
  - the four published artifacts (`abdallah1008/semantic-router-ml-models`
    at `7c83fa3e`) × 400 queries each.
- **Mutation checks:** parity fails when an exactness detail is perturbed
  (BM25 k1 1.25: 364 failures; n-gram results capped at 2: 411; selector
  ties or vote order changed: 64 and 128).

## Model selection on the published artifacts (µs per select)

The artifacts have 1,038 features: KNN 1,573 samples, SVM RBF 2,475 support
vectors and 4 classes, MLP 1038-256-128-4. "AVX2" is the default build on this
CPU; "portable" is the same code built with `-tags purego`, which uses Go
loops with four accumulators.

| Selector | Binding | Go, AVX2 | Speed-up | Go, portable | Speed-up |
| --- | --- | --- | --- | --- | --- |
| KNN | 2,010,082 | 137.8 | 14,583× | 438.5 | 4,584× |
| KMeans | 10.45 | 1.37 | 7.6× | 2.71 | 3.9× |
| SVM | 1,467.7 | 333.1 | 4.4× | 719.0 | 2.0× |
| MLP | 22.30 | 15.81 | 1.41× | 64.80 | 0.34× |

The MLP is why the AVX2 kernels exist. Candle runs its float32 GEMM with SIMD,
and the portable loop is 2.9× slower than candle. The kernels are one Go
assembly file (`pkg/embedding/vecmath`), selected at run time when the CPU has
both AVX2 and FMA. Their portable fallbacks are tested against them on the
same inputs.

## Kernels (ns per call, one core)

| Kernel | Length | AVX2 + FMA | Portable | Ratio |
| --- | --- | --- | --- | --- |
| `Dot` (float32) | 384 | 11.66 | 79.53 | 6.8× |
| `Dot` (float32) | 768 | 21.52 | 159.1 | 7.4× |
| `Dot` (float32) | 1,024 | 27.74 | 210.2 | 7.6× |
| `SquaredDistance64` | 1,038 | 56.50 | 268.7 | 4.8× |

The semantic cache's HNSW, the embedding signal, memory, KNN, KMeans and SVM
use these kernels.

## Keyword rules (µs per evaluation)

These ran at `c6dd12193` on node D, which has the same CPU model. They
replace the node B run at `75c1aa5e0`, made before the allocation work
below.

Each evaluation finds the first match and then all matches. That is two
binding calls, and two Go passes, each with its own text analysis.

| Scenario | Binding | Go | Speed-up | Go allocations |
| --- | --- | --- | --- | --- |
| BM25, 30 rules, long Chinese prompt | 10,398.9 | 428.7 | 24.3× | 246 |
| BM25, 5 rules, long Chinese prompt | 1,755.5 | 195.8 | 9.0× | 246 |
| BM25, 30 rules, short English prompt | 81.91 | 5.07 | 16.2× | 42 |
| BM25, code / medical rules | 9.24 | 5.85 | 1.58× | 58 |
| BM25, 30 rules, long English prompt | 3,378.8 | 100.6 | 33.6× | 64 |
| n-gram, urgent rule, typos | 43.93 | 14.73 | 3.0× | 176 |
| n-gram, code / medical rules | 76.02 | 20.73 | 3.7× | 346 |
| n-gram, 30 rules, long English prompt | 10,389.1 | 1,248.9 | 8.3× | 330 |

Where the speed comes from:

- **BM25:** posting lists from each query token straight to the keywords that
  contain it, with IDF times weight precomputed.
- **n-gram:** the text's words and n-grams are computed once and shared by all
  rules of an evaluation.
- **Both:** ASCII fast paths for lowercasing and word breaking.

Allocations per evaluation (long Chinese prompt: 1,568 → 246; long English:
710 → 64) came down three ways:

- each distinct word of a text is stemmed once;
- the Snowball stemmer rewrites its word in one buffer;
- a BM25 rule that no token reaches allocates nothing.

No keyword or selector scenario is slower than its binding.
