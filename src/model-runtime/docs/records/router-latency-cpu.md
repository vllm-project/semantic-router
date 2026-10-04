# Router latency on CPU: built-in model runtime against the in-process bindings

The router's model-backed signals moved from the in-process bindings (candle)
to the built-in model runtime. This record measures end-to-end router latency
per request with both, on the same CPU cores, inputs and signal set, and
states the design change that the measurement forced.

- **Date:** 2026-10-04.
- **Machine:** AMD EPYC 9575F, CPU only. Other workstreams' pinned jobs ran
  on other cores; the 1-minute load stayed between 28 and 55 (the program
  voids timings taken above 120).
- **Commits:**
  - Legacy: the router built from the exact mirror of `61aa7eb2d` with the
    candle CPU bindings (`--no-default-features`, as the CPU image).
  - Runtime: the router built from the exact tree of `bf630c5eb` (the
    canonical parser; the router downloads no model artifacts); the model
    runtime installed from the same tree into a Python 3.12 environment with
    PyTorch 2.10.0 (CPU) and ONNX Runtime 1.30.
- **Models:** Vela 1.0 Domain, Guard, PII, FactCheck and Feedback at their
  pinned registry revisions. The legacy router reads local artifacts at those
  revisions; the runtime loads the same revisions from its Hugging Face cache.
- **Configuration:** the default configuration's model-backed request signals
  (domain, prompt guard, PII, fact-check, feedback) plus one keyword signal,
  with decisions that read every signal. The runtime router serves
  [`router-latency-cpu.yaml`](router-latency-cpu.yaml) through five implicit
  `model_runtime` deployments. The legacy router serves
  [`router-latency-cpu-legacy.yaml`](router-latency-cpu-legacy.yaml): the same
  signals, decisions and models plus the backend selectors candle needs
  (`variant`, `use_mmbert_32k`), which the current parser refuses.
- **Inputs:** the 539 distinct user prompts of the repository's E2E test data
  plus edge cases (median 59 characters, 95th percentile 456, longest 4,920),
  written by [`tools/router_latency.py`](../../tools/router_latency.py)
  `corpus` (the same corpus file for every run in this record).
- **Method:** `POST /api/v1/routing/preview` (every signal and the decision, no
  upstream call), measured by the client. Each pass sends the corpus three
  times after 20 warm-up requests: sequentially, then at concurrency 4 and 16.
  Both result caches of the runtime path are off (the router's and the
  runtime's), so every request computes. Each router and its runtime processes
  are pinned to the same 16 cores (`taskset -c 48-63`); the driver runs on four
  other cores. Three rounds, alternating legacy and runtime. Raw summaries per
  round: [`router-latency-cpu.json`](router-latency-cpu.json).

## Result

The runtime router makes the same routing decision as the legacy router on
539 of 539 inputs in every round.

Median of three rounds (ms, and requests per second):

| Pass | Metric | Legacy (candle) | Runtime | Change |
| --- | --- | --- | --- | --- |
| Sequential | p50 | 52.0 | 14.8 | −72 % |
| | p95 | 231.9 | 48.1 | −79 % |
| | p99 | 447.2 | 89.3 | −80 % |
| | req/s | 11.1 | 44.4 | ×4.0 |
| Concurrency 4 | p50 | 165.8 | 52.1 | −69 % |
| | p95 | 611.3 | 136.6 | −78 % |
| | p99 | 918.1 | 552.2 | −40 % |
| | req/s | 17.3 | 54.8 | ×3.2 |
| Concurrency 16 | p50 | 702.3 | 211.6 | −70 % |
| | p95 | 2,234.5 | 719.6 | −68 % |
| | p99 | 3,440.6 | 760.3 | −78 % |
| | req/s | 17.7 | 66.4 | ×3.8 |

Per round, sequential p50 was 53.1 / 52.0 / 50.9 ms (legacy) and
15.4 / 14.8 / 13.9 ms (runtime).

The runtime router beats the legacy router on every percentile and on
throughput in every pass. Long inputs gain the most: the longest prompt
(4,920 characters) takes 2.55 s with candle and 0.51 s with the runtime; Guard
and PII scan it in windows and dominate both.

Footprint, after loading: the legacy router holds 8.5 GB resident; the runtime
router holds 0.1 GB plus 1.54 GB per runtime process, 7.8 GB for the five
models. The legacy router is ready 12 s after start, the runtime router 8–9 s.

## Earlier measurement

The first measurement, at `32a45d331` (before the parser flip, same machine
model, same method), had the runtime router at 30.6 ms sequential p50
(legacy 51.8) and one slower tail: p99 at concurrency 4 was 1,207 ms against
candle's 907 ms, because short requests queued behind long windowed forwards
inside a model's process. The runtime at `bf630c5eb` carries the IP1 engine
work (the exact profile on oneDNN packed linears and length-grouped packed
attention), which halves the sequential p50 again and removes that tail. Its
processes held 1.08 GB each then.

## The design change behind the result

A runtime process runs every model's forward on one device thread (PyTorch
keeps one OpenMP team per calling thread, so concurrent forwards would
oversubscribe the cores). With all five models in one CPU process, the default
the design started from, a request's five forwards ran one after another, each
split across all 16 cores. For short inputs that split is inefficient, and the
router lost to candle, which runs the models concurrently. Measured at
`32a45d331`:

| Process plan (first round of each run, same setup) | Sequential p50 | p95 | req/s | req/s at 16 |
| --- | --- | --- | --- | --- |
| Legacy (candle) | 58.7 | 264.6 | 9.9 | 16.7 |
| One process, 16 threads | 70.1 | 168.6 | 11.1 | 11.4 |
| One process per model, pinned to 4/3/3/3/3 cores | 37.8 | 152.2 | 15.6 | 16.7 |
| One process per model, 4 threads each, unpinned | 30.4 | 115.6 | 20.3 | 22.6 |

So the router plans CPU processes itself (`pkg/modelservice`): CPU models
without a `process` key spread over one process per model, at most one per two
cores (`VLLM_SR_RUNTIME_CPU_PROCESSES` caps it). Each CPU process runs
`ceil(cores / CPU processes)` threads, where cores is the router's
`GOMAXPROCS` (its affinity limited by the container quota). Pinning each process
to a disjoint share measured slower: the share of a rarely used model (the
feedback detector runs on follow-up turns only) then idles, while unpinned
threads let the busy processes use it. Every request stage still sends one
`/v1/bundle` per process; the stage's bundles run in parallel. GPU devices
keep one process per device.

## Reproduce

On a node with the runtime installed, with both router binaries and the models
downloaded:

```bash
python3 tools/router_latency.py corpus --repo . --out corpus.json
# start the legacy router with router-latency-cpu-legacy.yaml, then:
python3 tools/router_latency.py run --url http://127.0.0.1:8080 \
  --corpus corpus.json --out legacy.json --label legacy
# start the runtime router with router-latency-cpu.yaml, run again to runtime.json, then:
python3 tools/router_latency.py compare --base legacy.json --new runtime.json --corpus corpus.json
```

For the runtime router, set `VLLM_SR_RUNTIME_RESULT_CACHE=0` and start the
runtime with `--result-cache-entries 0` (through `VLLM_SR_RUNTIME_COMMAND`) to
measure without caches.
