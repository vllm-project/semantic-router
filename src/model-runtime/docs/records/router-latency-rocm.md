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
  is pinned to 4 other vCPUs. Other workstreams' jobs ran on other cores; each
  result gives the node's 1-minute load (the program voids timings taken above
  120).
- **Images**, both `tools/docker/Dockerfile.extproc` (target `extproc`,
  `ACCELERATOR=rocm`) built from exact mirrors:
  - **Shipped:** staging `a580be6b9`. Its ROCm stage takes PyTorch
    2.12.0+git6bbd260 (AOTriton 0.13.50) and the ROCm 7.2.3 userspace from
    `vllm/vllm-openai-rocm@sha256:1fd21abe…`, the release image's base, plus
    Triton 3.7.0, `fla-core` 0.5.2 and `causal-conv1d` 1.7.0. The runtime and
    router are at `a580be6b9`.
  - **Stack B, measured first:** the official PyTorch 2.12.0+rocm7.2 wheel
    (`TORCH_ROCM_INDEX` at the rocm7.2 index), HIP 7.2.53211, Triton 3.7.0 and
    `fla-core` 0.5.2, with no `causal-conv1d`, at `efc68a81c`: the router IP3
    head `a85c57f6e` plus `f3c09121d` (encoder graphs capture in thread-local
    mode).
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
  prompts, `router_latency.py corpus` at these commits writes a file that
  differs in those three; both records use the earlier file.
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

## Result on the shipped image

The ROCm router makes the same routing decision, from the same model, as the
CPU router on 539 of 539 inputs in every round. No request of the 30 passes
failed; no runtime process restarted, and none logged a device failure. The
one GPU process held 12.9–13.2 GiB of GPU memory with all five models loaded.
The node was quiet: its 1-minute load stayed between 5 and 24, and processes
outside the container used 0.2–0.4 of the router's 12 vCPUs during a pass.

Median of five rounds (ms, and requests per second), and the difference with
its 95% interval:

| Pass | Metric | CPU | ROCm | ROCm − CPU [95% CI] | ROCm speed-up |
| --- | --- | --- | --- | --- | --- |
| Sequential | p50 | 22.9 | 8.7 | −14.3 [−14.7, −13.9] | 2.6× |
| | p95 | 90.1 | 13.7 | −77.8 [−81.2, −74.4] | 6.6× |
| | p99 | 218.3 | 32.6 | −197.0 [−224.4, −169.5] | 6.7× |
| | req/s | 23.3 | 100.5 | +77.4 [+76.8, +78.1] | 4.3× |
| Concurrency 4 | p50 | 105.0 | 29.8 | −75.8 [−77.7, −74.0] | 3.5× |
| | p95 | 400.8 | 60.6 | −348.2 [−372.2, −324.2] | 6.6× |
| | p99 | 1,448.6 | 163.1 | −1,281.2 [−1,332.8, −1,229.6] | 8.9× |
| | req/s | 24.1 | 118.3 | +94.7 [+93.6, +95.8] | 4.9× |
| Concurrency 16 | p50 | 551.9 | 125.5 | −427.1 [−445.6, −408.6] | 4.4× |
| | p95 | 1,667.1 | 184.6 | −1,484.0 [−1,501.6, −1,466.4] | 9.0× |
| | p99 | 2,563.6 | 345.4 | −2,170.8 [−2,313.2, −2,028.4] | 7.4× |
| | req/s | 24.0 | 117.2 | +93.0 [+92.2, +93.8] | 4.9× |

Every interval lies wholly on the ROCm side. Per round, sequential p50 was
8.65–8.72 ms on ROCm and 22.8–23.6 ms on CPU. The JSON keeps every pass
(`shipped_image`).

## Result on the stack B image

The same five interleaved rounds, measured first, on the stack B image.
Decisions matched on 539 of 539 inputs in every round, no request of the 30
passes failed, no runtime process restarted and none logged a device failure;
the GPU process held 13.2–13.4 GiB. The 1-minute load stayed between 60 and
83, and processes outside the container used 0.2–0.6 of the router's vCPUs.
The JSON keeps every pass (`stack_b_image`).

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

## Stack B with `causal-conv1d`, and the node's other GPUs

After the stack B rounds, the same router served the same corpus from the
stack B image with `causal-conv1d` 1.7.0 and MIOpen's databases in the model
volume (staging `af71d5e82`), alternating with the stack B image for three
rounds on ROCm. Decisions matched on 539 of 539 inputs between the two images
in every round, and every point estimate favoured `af71d5e82`, most intervals
straddling zero. Its undisturbed passes match the stack B rounds: sequential
p50 10.3–10.9 ms and 104–105 requests per second at 16 callers.

By then four other GPUs of the node ran jobs whose host threads shared this
NUMA node, and both images showed an intermittent slow phase. In some
sub-passes the sequential p95 rose to 35–199 ms (15–18 ms otherwise), or the
rate at 16 callers fell to 29–70 requests per second, while the router
container spent 1.5–3.5 times its usual CPU time per request; the cause was
not isolated further. Four more passes of the stack B image after those jobs
paused were back at its rounds' values (sequential p50 10.4–10.9 ms, 101–104
requests per second at 16 callers), and a diagnostic hook in the runtime
logged 41 graph captures in each, none failed. So both sets of five rounds
show the GPU path on a node whose other GPUs were mostly idle; a router that
shares its host with other GPU jobs should expect that variance. The passes
are in the JSON (`stack_b_with_causal_conv1d`).

## What the CPU side measures

The CPU side is the ROCm image's PyTorch running the models on CPU
(2.12.0+git6bbd260 in the shipped image, 2.12.0+rocm7.2 in stack B): what a
model gets in the ROCm image when its deployment sets `use_cpu: true`. It is
not the CPU image, and two things set it apart from `router-latency-cpu.md`,
which stays the reference for CPU deployments:

- **Cross-request batching.** On CPU, `exact` batches concurrent requests only
  for a model whose rows the load-time probe finds identical alone and inside
  a mixed batch (design §9 and §11). With either ROCm image's PyTorch, Domain,
  Guard, FactCheck and Feedback fail the probe and PII passes, so four of the
  five models answer one request at a time under load. With the CPU image
  built from the same tree (PyTorch 2.10.0+cpu) all five pass.
- **The node.** The CPU rows move with the node's load. On the quiet node, the
  shipped image's CPU side ran at a sequential p50 of 22.8–23.6 ms. In two
  diagnostic rounds on the same cores, alternating stack B's CPU side with the
  CPU image (order rotated), the CPU image's sequential p50 was 30.3 and
  24.0 ms and its rate at 16 callers 31.8 and 45.3 requests per second, against
  37.7 and 50.5 ms and 17.7 and 15.5 for stack B. The node's load was higher
  during stack B's passes (87–90 against 69–72), so the two rounds don't
  separate the image from the load; they are in the JSON (`cpu_side`).

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

As in the CPU record, set `VLLM_SRUN_RESULT_CACHE=0` and start the
runtime with `--result-cache-entries 0` (through `VLLM_SRUN_COMMAND`) to
measure without caches.
