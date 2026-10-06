# Router latency on CPU: built-in model runtime against the in-process bindings

The router's model-backed signals moved from the in-process bindings (candle)
to the built-in model runtime. This record measures end-to-end router latency
per request with both, on the same CPU cores, inputs and signal set, and
states the design change that the measurement forced.

- **Date:** 2026-10-05.
- **Machine:** AMD EPYC 9575F, CPU only. Other workstreams' jobs ran on other
  cores; the 1-minute load stayed between 41 and 86 (the program voids timings
  taken above 120).
- **Commits:**
  - Legacy: the router built from the exact mirror of `61aa7eb2d` with the
    candle CPU bindings (`--no-default-features`, as the CPU image).
  - Runtime: the router built from `7409aaac7` (the IP3a′ head `1a153479a`,
    plus the bundle cap read from `/v1/models` and the fusion of a stage's
    classify calls); the model runtime installed from the same tree into a
    Python 3.12 environment with the router image's pins, PyTorch 2.10.0 (CPU)
    and ONNX Runtime 1.30.0.
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
  upstream call), measured by the client. Each pass sends 20 warm-up requests,
  then the corpus three times: sequentially, then at concurrency 4 and 16.
  Both result caches of the runtime path are off (the router's and the
  runtime's), so every request computes.
  - Each router and its runtime processes run in their own
    `systemd-run --scope -p AllowedCPUs=32-43` (12 vCPUs); the effective
    cpuset of every scope read 32–43. The driver is pinned to 4 other vCPUs.
  - Five rounds alternate legacy and runtime, and the order rotates each round.
  - Per metric, the table gives the median of the five rounds and the paired
    difference runtime − legacy per round with a 95% t interval
    (`router_latency.py rounds`).
  - Raw summaries per round, with each scope's load and cpuset:
    [`router-latency-cpu.json`](router-latency-cpu.json).

## Result

The runtime router makes the same routing decision as the legacy router on
539 of 539 inputs in every round, and no request of the 30 passes failed.

Median of five rounds (ms, and requests per second), and the difference with
its 95% interval:

| Pass | Metric | Legacy (candle) | Runtime | Runtime − legacy [95% CI] |
| --- | --- | --- | --- | --- |
| Sequential | p50 | 53.4 | 15.5 | −38.7 [−42.1, −35.3] |
| | p95 | 245.1 | 55.2 | −194.6 [−209.9, −179.3] |
| | p99 | 495.5 | 105.9 | −386.3 [−393.6, −378.9] |
| | req/s | 10.6 | 40.1 | +27.8 [+22.9, +32.6] |
| Concurrency 4 | p50 | 194.1 | 60.4 | −139.5 [−157.9, −121.1] |
| | p95 | 709.2 | 231.1 | −492.4 [−536.2, −448.5] |
| | p99 | 1,091.4 | 303.7 | −802.2 [−859.4, −745.0] |
| | req/s | 14.7 | 48.1 | +32.3 [+28.6, +35.9] |
| Concurrency 16 | p50 | 815.5 | 291.5 | −536.3 [−584.0, −488.7] |
| | p95 | 2,691.5 | 688.5 | −2,027.2 [−2,189.7, −1,864.6] |
| | p99 | 4,328.8 | 1,090.9 | −3,316.9 [−3,624.9, −3,008.8] |
| | req/s | 14.9 | 50.5 | +34.3 [+29.9, +38.6] |

Every interval lies wholly on the runtime's better side, so every row passes
the no-regression standard. Per round, sequential p50 was 67.0 / 53.4 / 53.4 /
52.7 / 52.7 ms (legacy) and 23.5 / 15.5 / 15.5 / 15.5 / 15.6 ms (runtime); the
first round is slower on both sides. Long inputs gain the most: the longest
prompt (4,920 characters) takes a median 2.64 s with candle and 0.63 s with the
runtime; Guard and PII scan it in windows and dominate both.

Both routers are ready within 8 s (runtime) and 10–11 s (legacy) of start.

## Earlier measurements

- **`bf630c5eb` (2026-10-04, superseded by this record):** three rounds
  alternating legacy and runtime with `taskset` on 16 cores, no intervals.
  The runtime led on every percentile and on throughput (sequential p50 14.8
  against 52.0 ms; 66.4 against 17.7 req/s at concurrency 16). After loading,
  the legacy router held 8.5 GB resident, and the runtime router 0.1 GB plus
  1.54 GB per runtime process (7.8 GB for the five models).
- **`32a45d331` (before the parser flip, same method):** the runtime router
  was at 30.6 ms sequential p50 (legacy 51.8), with one slower tail: p99 at
  concurrency 4 was 1,207 ms against candle's 907 ms, because short requests
  queued behind long windowed forwards inside a model's process. The IP1 engine
  work (the exact profile on oneDNN packed linears and length-grouped packed
  attention) halved the sequential p50 again and removed that tail.

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
cores (`VLLM_SRUN_CPU_PROCESSES` caps it). Each CPU process runs
`ceil(cores / CPU processes)` threads, where cores is the router's
`GOMAXPROCS` (its affinity limited by the container quota). Pinning each process
to a disjoint share measured slower: the share of a rarely used model (the
feedback detector runs on follow-up turns only) then idles, while unpinned
threads let the busy processes use it. Every request stage still sends one
`/v1/bundle` per process; the stage's bundles run in parallel. GPU devices
keep one process per device.

## Reproduce

On a node with the runtime installed, with both router binaries and the models
downloaded, start each router in its own cgroup scope and drive it from other
cores:

```bash
python3 tools/router_latency.py corpus --repo . --out corpus.json
# Each round, in the rotated order: start a router, run, stop it. For example:
systemd-run --scope -p AllowedCPUs=32-43 env LD_LIBRARY_PATH=<candle libraries> \
  router-candle -config router-latency-cpu-legacy.yaml -api-port 18080 &
taskset -c 44-47 python3 tools/router_latency.py run --url http://127.0.0.1:18080 \
  --corpus corpus.json --out rounds/r1-legacy.json --label legacy-r1
# ... the same with the runtime router and router-latency-cpu.yaml to rounds/r1-runtime.json ...
python3 tools/router_latency.py rounds --dir rounds --base legacy --new runtime
```

For the runtime router, set `VLLM_SRUN_RESULT_CACHE=0` and start the
runtime with `--result-cache-entries 0` (through `VLLM_SRUN_COMMAND`) to
measure without caches.
