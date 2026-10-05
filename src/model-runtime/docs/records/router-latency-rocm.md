# Router latency on ROCm: the model runtime on one GPU against the same runtime on CPU

The router's model-backed signals run in the built-in model runtime, which can
place them on a GPU. The legacy router had no ROCm request path for these
signals, so this record measures the runtime router with its models on one
AMD Instinct MI325X against the same router with the models on CPU, on the
same host cores, inputs and image. The model-level ROCm comparison with the
legacy AMD recipe (ONNX Runtime MIGraphX / ROCm EP) is in
[`vela1-performance.md`](vela1-performance.md); the CPU comparison with the
legacy router is in [`router-latency-cpu.md`](router-latency-cpu.md).

- **Date:** 2026-10-05.
- **Machine:** a VM with AMD EPYC 9575F vCPUs and AMD Instinct MI325X GPUs
  (gfx942). The router container (the router and its runtime processes) is
  pinned to 12 vCPUs with `docker run --cpuset-cpus`, and the effective cpuset
  read the same 12 in every pass; the ROCm side also gets one GPU. The driver
  is pinned to 4 other vCPUs. Other workstreams' jobs ran on other cores: the
  1-minute load stayed between 60 and 83 (the program voids timings taken above
  120), and processes outside the container used 0.2–0.6 of the router's 12
  vCPUs during a pass.
- **Commit:** `efc68a81c`, a trial tree: the router IP3 head `a85c57f6e` plus
  `f3c09121d` (encoder graphs capture in thread-local mode). Its runtime package
  equals the IP3 staging's (`a0d396565`) except the decoder-only `fast.py`, which
  these encoders don't run.
- **Image:** `tools/docker/Dockerfile.extproc` (target `extproc`,
  `ACCELERATOR=rocm`, `TORCH_ROCM_INDEX` at the rocm7.2 index), built from the
  exact mirror: PyTorch 2.12.0+rocm7.2, HIP 7.2.53211, Triton 3.7.0, `fla-core`
  0.5.2. It has no `causal-conv1d`. The shipped image adds it for the decoders'
  causal convolution, which the Vela 1.0 (ModernBERT) encoders don't have, so
  it doesn't change these rows.
- **Models:** Vela 1.0 Domain, Guard, PII, FactCheck and Feedback at their
  pinned registry revisions, as in `router-latency-cpu.md`.
- **Configuration:**
  - CPU side: [`router-latency-cpu.yaml`](router-latency-cpu.yaml), the CPU
    record's file. The router runs one CPU process per model (five), each with
    its share of the 12 cores.
  - ROCm side: the same file with `use_cpu: false` on the five models. The
    router places them in one runtime process on `rocm:0` (`exact` profile,
    native engine). That process runs one model's device work at a time (the
    per-device lock). Each model replays a HIP graph per length bucket,
    captured in thread-local mode on the bucket's second use; long rows run
    eagerly.
- **Inputs:** the CPU record's corpus file (539 prompts, sha256
  `dc81026ad0c9…`). Since `551436917` changed three cache-polarity test
  prompts, `router_latency.py corpus` at this commit writes a file that differs
  in those three; both records use the earlier file.
- **Method:** the CPU record's. `POST /api/v1/routing/preview` (every signal and
  the decision, no upstream call), measured by the client. Each pass sends 20
  warm-up requests, then the corpus three times: sequentially, then at
  concurrency 4 and 16. Both result caches are off, so every request computes.
  - Every pass starts a fresh router container.
  - Five rounds alternate ROCm and CPU, and the order rotates each round.
  - Per metric, the table gives the median of the five rounds and the paired
    difference ROCm − CPU per round with a 95% t interval
    (`router_latency.py rounds`).
  - Raw summaries per round, with each pass's load, cpuset, GPU memory, runtime
    restarts and device failures:
    [`router-latency-rocm.json`](router-latency-rocm.json).

## Result

The ROCm router makes the same routing decision, from the same model, as the
CPU router on 539 of 539 inputs in every round. No request of the 30 passes
failed; no runtime process restarted, and none logged a device failure. The
one GPU process held 13.2–13.4 GiB of GPU memory with all five models loaded.

Median of five rounds (ms, and requests per second), and the difference with
its 95% interval:

| Pass | Metric | CPU | ROCm | ROCm − CPU [95% CI] | ROCm speed-up |
| --- | --- | --- | --- | --- | --- |
| Sequential | p50 | 30.8 | 9.8 | −20.2 [−25.1, −15.4] | 3.1× |
| | p95 | 81.4 | 16.3 | −66.3 [−73.4, −59.2] | 5.0× |
| | p99 | 167.3 | 31.8 | −133.2 [−147.8, −118.7] | 5.3× |
| | req/s | 23.4 | 90.4 | +67.7 [+63.8, +71.6] | 3.9× |
| Concurrency 4 | p50 | 116.6 | 31.0 | −83.7 [−104.1, −63.3] | 3.8× |
| | p95 | 314.2 | 67.2 | −248.1 [−298.2, −198.0] | 4.7× |
| | p99 | 692.9 | 187.3 | −526.0 [−628.5, −423.6] | 3.7× |
| | req/s | 25.9 | 109.4 | +79.2 [+62.6, +95.7] | 4.2× |
| Concurrency 16 | p50 | 551.6 | 135.5 | −418.1 [−494.2, −342.1] | 4.1× |
| | p95 | 1,216.6 | 200.3 | −985.5 [−1,220.2, −750.9] | 6.1× |
| | p99 | 1,472.2 | 364.4 | −1,110.2 [−1,387.6, −832.8] | 4.0× |
| | req/s | 26.1 | 107.8 | +82.5 [+77.5, +87.5] | 4.1× |

Every interval lies wholly on the ROCm side. Per round, sequential p50 was
30.8 / 35.1 / 31.8 / 23.7 / 27.9 ms on CPU and 10.1 / 9.8 / 10.0 / 9.0 / 9.1 ms
on ROCm: the GPU rows barely move with the node's load, and the CPU rows do.
ROCm throughput levels off at about 108 requests per second from 4 callers
on: the one GPU process runs the five models' device work one at a time, so
more callers only queue.

## What the CPU side measures

The CPU side is this image's PyTorch (2.12.0+rocm7.2) running the models on
CPU: what a model gets in the ROCm image when its deployment sets
`use_cpu: true`. It is not the CPU image, and two things set it apart from
`router-latency-cpu.md`, which stays the reference for CPU deployments:

- **Cross-request batching.** On CPU, `exact` batches concurrent requests only
  for a model whose rows the load-time probe finds identical alone and inside
  a mixed batch (design §9 and §11). With this image's PyTorch, Domain, Guard,
  FactCheck and Feedback fail the probe and PII passes, so four of the five
  models answer one request at a time under load. With the CPU image built
  from the same tree (PyTorch 2.10.0+cpu) all five pass.
- **The node.** The CPU rows move with the node's load. In two diagnostic
  rounds on the same cores, alternating this side with the CPU image (order
  rotated), the CPU image's sequential p50 was 30.3 and 24.0 ms and its rate at
  16 callers 31.8 and 45.3 requests per second, against 37.7 and 50.5 ms and
  17.7 and 15.5 here. The node's load was higher during this side's passes
  (87–90 against 69–72), so the two rounds don't separate the image from the
  load; they are in the JSON (`cpu_side`).

## Reproduce

On a ROCm node with the image built and the five models in the runtime's
cache, run each pass in a fresh container pinned to the router's cores and
drive it from other cores:

```bash
python3 tools/router_latency.py corpus --repo . --out corpus.json  # three prompts differ from the recorded file
sed 's/^\(\s*\)use_cpu: true$/\1use_cpu: false/' router-latency-cpu.yaml > router-latency-rocm.yaml
# Each round, in the rotated order: start a router container, run, remove it.
# The ROCm side, for example (the CPU side drops the device flags and serves
# router-latency-cpu.yaml):
docker run -d --name router-rocm --network host --cpuset-cpus 128-139 \
  --device /dev/kfd --device /dev/dri --group-add video -e HIP_VISIBLE_DEVICES=6 \
  -v "$PWD":/work:ro --entrypoint /usr/local/bin/router <image> \
  --config=/work/router-latency-rocm.yaml --api-port 18080
taskset -c 140-143 python3 tools/router_latency.py run --url http://127.0.0.1:18080 \
  --corpus corpus.json --out rounds/r1-rocm.json --label rocm-r1
# ... the same with the CPU side to rounds/r1-cpu.json ...
python3 tools/router_latency.py rounds --dir rounds --base cpu --new rocm
```

As in the CPU record, set `VLLM_SR_RUNTIME_RESULT_CACHE=0` and start the
runtime with `--result-cache-entries 0` (through `VLLM_SR_RUNTIME_COMMAND`) to
measure without caches.
