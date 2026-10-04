# Router latency on CPU: built-in model runtime against the in-process bindings

The router's model-backed signals moved from the in-process bindings (candle)
to the built-in model runtime. This record measures end-to-end router latency
per request with both, on the same CPU cores, configuration and inputs, and
states the design change that the measurement forced.

- **Date:** 2026-10-04.
- **Machine:** node A, AMD EPYC 9575F, CPU only, otherwise idle.
- **Commits:**
  - Legacy: the router built from the exact mirror of `61aa7eb2d` with the
    candle CPU bindings (`--no-default-features`, as the CPU image).
  - Runtime: the router built from the exact mirror of `32a45d331`; the model
    runtime installed from the same tree into a Python 3.12 environment with
    PyTorch 2.10.0 (CPU) and ONNX Runtime 1.30.
- **Models:** Vela 1.0 Domain, Guard, PII, FactCheck and Feedback at their
  pinned registry revisions. Both routers read the same local artifacts; the
  runtime loads the same revisions from its Hugging Face cache.
- **Configuration:** [`router-latency-cpu.yaml`](router-latency-cpu.yaml):
  the default configuration's model-backed request signals (domain, prompt
  guard, PII, fact-check, feedback) plus one keyword signal, with decisions that
  read every signal. Each router serves it unchanged, so the legacy router runs
  the five models through candle and the new router through five implicit
  `model_runtime` deployments.
- **Inputs:** the 539 distinct user prompts of the repository's E2E test data
  plus edge cases (median 59 characters, 95th percentile 456, longest 4,920),
  written by [`tools/router_latency.py`](../../tools/router_latency.py)
  `corpus`.
- **Method:** `POST /api/v1/routing/preview` (every signal and the decision, no
  upstream call), measured by the client. Each pass sends the corpus three
  times after 20 warm-up requests: sequentially, then at concurrency 4 and 16.
  Both result caches of the runtime path are off (the router's and the
  runtime's), so every request computes. Each router and its runtime processes
  are pinned to the same 16 cores (`taskset -c 48-63`); the driver runs on four
  other cores. Three rounds, alternating legacy and runtime.

## Result

The runtime router makes the same routing decision as the legacy router on
539 of 539 inputs in every round.

Median of three rounds (ms, and requests per second):

| Pass | Metric | Legacy (candle) | Runtime | Change |
| --- | --- | --- | --- | --- |
| Sequential | p50 | 51.8 | 30.6 | −41 % |
| | p95 | 233.0 | 115.6 | −50 % |
| | p99 | 459.2 | 223.2 | −51 % |
| | req/s | 11.1 | 20.3 | +83 % |
| Concurrency 4 | p50 | 166.5 | 139.0 | −17 % |
| | p95 | 609.4 | 314.2 | −48 % |
| | p99 | 906.6 | 1,207.0 | +33 % |
| | req/s | 17.2 | 22.4 | +30 % |
| Concurrency 16 | p50 | 712.9 | 659.8 | −7 % |
| | p95 | 2,265.9 | 1,719.6 | −24 % |
| | p99 | 3,549.4 | 1,909.2 | −46 % |
| | req/s | 17.4 | 22.6 | +30 % |

Per round, sequential p50 was 50.8 / 51.8 / 57.1 ms (legacy) and
30.4 / 30.6 / 40.6 ms (runtime); the third round ran slower for both.

The runtime router matches or beats the legacy router on p50 and p95 in every
pass, and on throughput. One tail is slower: p99 at concurrency 4 (see
[Remaining gap](#remaining-gap)).

Long inputs gain the most. The longest prompt (4,920 characters) takes
2.5–2.6 s with candle and 1.06–1.13 s with the runtime; Guard and PII scan it in
windows and dominate both.

Footprint, after loading: the legacy router holds 8.5 GB resident; the runtime
router holds 0.1 GB plus 1.1 GB per runtime process, 5.5 GB for the five models.
Both are ready 11 s after start.

## The design change behind the result

A runtime process runs every model's forward on one device thread (PyTorch
keeps one OpenMP team per calling thread, so concurrent forwards would
oversubscribe the cores). With all five models in one CPU process, the default
the design started from, a request's five forwards ran one after another, each
split across all 16 cores. For short inputs that split is inefficient, and the
router lost to candle, which runs the models concurrently:

| Process plan (first round of each run, same setup) | Sequential p50 | p95 | req/s | req/s at 16 |
| --- | --- | --- | --- | --- |
| Legacy (candle) | 58.7 | 264.6 | 9.9 | 16.7 |
| One process, 16 threads | 70.1 | 168.6 | 11.1 | 11.4 |
| One process per model, pinned to 4/3/3/3/3 cores | 37.8 | 152.2 | 15.6 | 16.7 |
| One process per model, 4 threads each, unpinned | 30.4 | 115.6 | 20.3 | 22.6 |

So the router now plans CPU processes itself (`pkg/modelservice`): CPU models
without a `process` key spread over one process per model, at most one per two
cores (`VLLM_SR_RUNTIME_CPU_PROCESSES` caps it). Each CPU process runs
`ceil(cores / CPU processes)` threads, where cores is the router's
`GOMAXPROCS` (its affinity limited by the container quota). Pinning each process
to a disjoint share measured slower: the share of a rarely used model (the
feedback detector runs on follow-up turns only) then idles, while unpinned
threads let the busy processes use it. Every request stage still sends one
`/v1/bundle` per process; the stage's bundles now run in parallel. GPU devices
keep one process per device.

## Remaining gap

At concurrency 4, p99 is higher with the runtime than with candle. A model's
process batches the requests queued for it, so a short request queued behind a
long, windowed one waits for that forward; candle ran each request's forward on
its own thread. Throughput is still 1.3 times candle's at this concurrency, and
p95 is half. Ordering a model's queue by length, or a separate lane for long
windowed inputs, is a runtime scheduler change.

## Reproduce

On a node with the runtime installed, with both router binaries and the models
downloaded:

```bash
python3 tools/router_latency.py corpus --repo . --out corpus.json
# start a router with router-latency-cpu.yaml, then:
python3 tools/router_latency.py run --url http://127.0.0.1:8080 \
  --corpus corpus.json --out legacy.json --label legacy
python3 tools/router_latency.py compare --base legacy.json --new runtime.json --corpus corpus.json
```

For the runtime router, set `VLLM_SR_RUNTIME_RESULT_CACHE=0` and start the
runtime with `--result-cache-entries 0` (through `VLLM_SR_RUNTIME_COMMAND`) to
measure without caches.
