# Pure-Go keyword rules and model selectors against the bindings

The router's keyword rules (BM25, n-gram) and model selectors (KNN, KMeans,
SVM, MLP) moved from `nlp-binding`, `ml-binding` and candle to Go. This
record compares them with the bindings on the same inputs and states the
parity evidence.

- **Date:** 2026-10-05 (timing); the parity evidence is from 2026-10-04.
- **Machine:** node B, AMD EPYC 9575F, CPU only. Each run had its own cgroup
  cpuset scope on the same 16 cores (`systemd-run --scope -p
  AllowedCPUs=48-63`); the effective cpuset, read inside each of the 12
  scopes, was 48–63. `GOMAXPROCS`, `RAYON_NUM_THREADS` and `OMP_NUM_THREADS`
  were 16. Other timed runs used other cores; the host's 1-minute load was
  49–79 of 160.
- **Commits:**
  - Go: `f5d35999c`.
  - Legacy: the base tree `1c6d372ec`, with `nlp-binding`, `ml-binding` and
    candle built from it and called as the router called them.

  Both are exact mirrors; a temporary harness was added to scratch copies.
- **Inputs:**
  - Keyword rules: the eight scenarios below, with the rules and texts of the
    side-by-side benchmark in `e85d51dbd`.
  - Selectors: the four published artifacts
    (`abdallah1008/semantic-router-ml-models` at `7c83fa3e`, sha256-checked)
    and their fixture queries, 400 each, regenerated from the recorded seeds.
    KNN uses the first 16: the binding rebuilds its ball tree on every select
    (about 2 s).

## Method

Six rounds. A round runs the legacy binary and the Go binary once each, in
fresh processes and their own scopes, legacy first in odd rounds and Go first
in even ones. Every call is timed on its own:

- a keyword evaluation (the first match, then all matches, as the keyword
  signal calls them), after 50 warm-up calls, for at least 2 s and 200 calls;
- a select over the query set, after one warm-up pass, for at least 3 s and
  one pass.

The tables show means over the six rounds. The intervals pair each round's Go
value with legacy's from the same round: a 95% t interval (5 degrees of
freedom) over the six differences. Throughput is one caller's operations per
second (1 / the mean latency).

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
  - the four published artifacts × 400 queries each.
- **Mutation checks:** parity fails when an exactness detail is perturbed
  (BM25 k1 1.25: 364 failures; n-gram results capped at 2: 411; selector
  ties or vote order changed: 64 and 128).

## Model selection on the published artifacts (µs per select)

The artifacts have 1,038 features: KNN 1,573 samples, SVM RBF 2,475 support
vectors and 4 classes, MLP 1038-256-128-4. Go runs its AVX2 + FMA kernels on
this CPU.

| Selector | Legacy p50 / p95 | Go p50 / p95 | Go − legacy, p50 | Go − legacy, p95 | Selects/s, legacy → Go | Go − legacy, selects/s |
| --- | --- | --- | --- | --- | --- | --- |
| KNN | 2,021,863 / 2,054,309 | 123 / 132 | −2,021,740 [−2,032,125, −2,011,355] | −2,054,176 [−2,082,061, −2,026,292] | 0.49 → 8,118 | +8,118 [+7,888, +8,347] |
| KMeans | 9.36 / 11.0 | 1.08 / 1.20 | −8.28 [−8.38, −8.19] | −9.85 [−10.3, −9.43] | 101,983 → 907,047 | +805,064 [+794,806, +815,321] |
| SVM | 1,213 / 1,344 | 253 / 284 | −959 [−1,099, −819] | −1,060 [−1,304, −816] | 825 → 3,901 | +3,076 [+2,914, +3,237] |
| MLP | 19.4 / 22.8 | 12.9 / 15.0 | −6.50 [−6.88, −6.11] | −7.81 [−10.7, −4.87] | 49,920 → 74,552 | +24,632 [+22,727, +26,537] |

The MLP is why the kernels exist: candle runs its float32 GEMM with SIMD.
They are Go assembly in `pkg/embedding/vecmath`, selected at run time when
the CPU has both AVX2 and FMA. Their portable fallbacks are tested against
them on the same inputs.

## arm64

The CPU router image is also published for linux/arm64. There the kernels
are NEON assembly (`f5d35999c`), with the same structure as the AVX2 ones:
eight accumulators, a fixed reduction tree, then the scalar tail. Without
them, arm64 would run the portable loop. On this CPU that loop
(`-tags purego`, 2026-10-04 at `75c1aa5e0`) took the MLP select to 64.8 µs
against candle's 22.3 µs (0.34×), the only selector it left behind its
binding (KNN, KMeans and SVM stayed 2.0–4,584× ahead).

Under qemu-aarch64 the vecmath tests pass, and they fail when a kernel
instruction is mutated. Every recorded selection matches there too,
including the published artifacts' 1,600. No arm64 host was available, so
the arm64 kernels are not timed here.

## Kernels (ns per call, one core)

AVX2 + FMA against the portable loop on this CPU, from 2026-10-04 at
`75c1aa5e0`. This is not a comparison with a binding.

| Kernel | Length | AVX2 + FMA | Portable | Ratio |
| --- | --- | --- | --- | --- |
| `Dot` (float32) | 384 | 11.66 | 79.53 | 6.8× |
| `Dot` (float32) | 768 | 21.52 | 159.1 | 7.4× |
| `Dot` (float32) | 1,024 | 27.74 | 210.2 | 7.6× |
| `SquaredDistance64` | 1,038 | 56.50 | 268.7 | 4.8× |

The semantic cache's HNSW, the embedding signal, memory, KNN, KMeans and SVM
use these kernels.

## Keyword rules (µs per evaluation)

Each evaluation finds the first match and then all matches. That is two
binding calls, and two Go passes, each with its own text analysis.

| Scenario | Legacy p50 / p95 | Go p50 / p95 | Go − legacy, p50 | Go − legacy, p95 | Evaluations/s, legacy → Go | Go − legacy, evaluations/s |
| --- | --- | --- | --- | --- | --- | --- |
| BM25, 30 rules, long Chinese prompt | 10,223 / 10,311 | 400 / 681 | −9,823 [−9,927, −9,718] | −9,631 [−9,799, −9,463] | 97.7 → 2,286 | +2,188 [+2,021, +2,355] |
| BM25, 5 rules, long Chinese prompt | 1,710 / 1,730 | 173 / 303 | −1,537 [−1,557, −1,517] | −1,427 [−1,463, −1,391] | 583 → 5,258 | +4,675 [+4,425, +4,926] |
| BM25, 30 rules, short English prompt | 74.6 / 81.7 | 4.88 / 6.78 | −69.7 [−71.2, −68.2] | −74.9 [−78.0, −71.9] | 13,263 → 191,190 | +177,927 [+171,310, +184,544] |
| BM25, code / medical rules | 8.89 / 10.4 | 5.49 / 6.95 | −3.40 [−3.46, −3.34] | −3.46 [−4.64, −2.29] | 109,578 → 168,377 | +58,798 [+56,162, +61,434] |
| BM25, 30 rules, long English prompt | 3,311 / 3,390 | 96.0 / 159 | −3,215 [−3,237, −3,193] | −3,231 [−3,323, −3,139] | 301 → 9,688 | +9,388 [+9,187, +9,588] |
| n-gram, urgent rule, typos | 41.4 / 47.6 | 14.5 / 27.6 | −26.9 [−27.4, −26.4] | −20.0 [−24.6, −15.5] | 23,830 → 54,462 | +30,633 [+28,976, +32,290] |
| n-gram, code / medical rules | 72.8 / 83.0 | 20.6 / 38.7 | −52.1 [−52.7, −51.6] | −44.3 [−54.4, −34.2] | 13,431 → 38,545 | +25,114 [+24,148, +26,080] |
| n-gram, 30 rules, long English prompt | 10,006 / 10,046 | 1,251 / 1,653 | −8,755 [−8,799, −8,710] | −8,393 [−8,749, −8,038] | 99.9 → 773 | +673 [+653, +694] |

Where the speed comes from:

- **BM25:** posting lists from each query token straight to the keywords that
  contain it, with IDF times weight precomputed.
- **n-gram:** the text's words and n-grams are computed once and shared by all
  rules of an evaluation.
- **Both:** ASCII fast paths for lowercasing and word breaking.

Allocations per evaluation (2026-10-04, at `c6dd12193`; the keyword code is
unchanged since) came down three ways, from 1,568 to 246 for the long Chinese
prompt and from 710 to 64 for the long English one:

- each distinct word of a text is stemmed once;
- the Snowball stemmer rewrites its word in one buffer;
- a BM25 rule that no token reaches allocates nothing.

Every interval above is on Go's side of zero: no keyword or selector row
regressed, at p50, at p95 or in throughput. The narrowest margins are the
BM25 code / medical rules (3.40 µs at p50) and the MLP select (6.50 µs at
p50, 1.5× the binding's throughput).
