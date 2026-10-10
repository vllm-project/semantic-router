# CPU forward thread scaling on Arm hosts

Before optimizing the built-in signals' CPU cost (#4668), this record measures
where a signal forward's time actually goes on aarch64 hosts, and how the
intra-op thread count behaves across input sizes and host generations. Three
findings shape the runtime's CPU path:

1. A short classify spends ~99% of its time in `forward` itself; the request
   pipeline is not the bottleneck.
2. Most of that `forward` is per-op overhead — allocation, atomics, barrier
   fork/join and Python dispatch — not GEMM compute. The overhead is
   structural: an eager OMP wait policy makes it far worse, not better.
3. Responses are bit-identical across intra-op thread counts for both Vela
   1.0 encoders and the Vela 2.0 0.3B decoder, so adapting the thread count
   per input cannot change `exact`-profile answers. The optimal count,
   however, differs between host generations, so a static default is not
   optimal everywhere.

This is a measurement record only; it changes no behavior.

- **Date:** 2026-10-10.
- **Machines:** two shared Arm hosts.
  - Kunpeng 950 7590, 2 sockets, 384 cores @ 2.3 GHz, 4 NUMA nodes
    (`bf16`/`i8mm`/`sve`/`sve2`), 1.5 TiB. One-minute load varied between
    45 and 200 during the measurements (other tenants).
  - Kunpeng 920, 192 cores, 8 NUMA nodes. Load stayed between 11 and 15.
- **Runtime:** model runtime from main `738bf4839`, PyTorch 2.10.0 (CPU,
  oneDNN with the ACL backend), Python 3.11. Standalone
  `vllm-srun serve` over a Unix socket, `device cpu`, profile `exact`,
  the whole stack under `taskset -c 0-31` (32 cores spanning two NUMA
  nodes on the 950, two on the 920).
- **Models:** `Vela-1.0-Encoder-307M-Domain` (local package) for
  `/v1/classify`; `Vela-2.0-0.3B` for the four-signal `/v1/decisions` call
  (domain, prompt guard, PII presence and categories, fact-check — the
  built-in questions compile to a 473-token schema; with a ~190-token
  state the request is 662 tokens).
- **Inputs:** synthetic passages made unique per request — both result
  caches are on, so a repeated text answers from cache and measures nothing.
  Long-input caveat: the Router bounds what a signal scans; these
  standalone measurements give the runtime the full input, so only the
  router-attached numbers state production latency.
- **Method:**
  - Phases from the runtime's `Server-Timing` header
    (`parse;tokenize;queue;forward;post;serialize;total`).
  - Thread counts via `--threads N` (the runtime sets its intra-op threads
    from it); one server process per count, one lazy-load warm-up request
    retried until `forward > 0`, then unique requests.
  - Bit-invariance: byte-compare the response bodies of the same input
    across thread counts.
  - Attribution: `py-spy record --native` sampling (150 Hz) while requests
    run; `torch.profiler` was not usable here — it recorded no events and
    its per-op tracing inflated the forward it measured from ~190 ms to
    1,583 ms.
  - Raw medians per run:
    [`cpu-forward-thread-scaling-arm.json`](cpu-forward-thread-scaling-arm.json).

## Result

### A short classify is forward, and forward is not GEMM

On the 950, a 29-token `classify` request reports (medians, `Server-Timing`):
parse/tokenize/queue/post/serialize are each below 1 ms, and `forward` is
~99% of the total. The model's 88 GEMMs, measured in isolation at the same
thread count, sum to ~15 ms. The real `forward` takes 87–157 ms depending on
the thread count. py-spy leaf frames agree: the top self-time frames are the
allocator (`malloc`, `madvise`, mimalloc pages), `syscall`,
`__aarch64_ldadd*` atomics, `pthread_create`/`clone` still occurring during
steady-state forwards, and Python dispatch
(`FunctionSignature::parse`, `THPVariable_Wrap`, `Module._call_impl`); GEMM
kernels barely appear. A 22-layer encoder's short-input forward is hundreds
of tiny parallel regions, and each region pays allocation, barrier and
dispatch costs that dwarf its arithmetic.

### Thread scaling: the crossover is host- and input-dependent

Medians; every server under the same 32-core cpuset.

| Workload | Host | 1 thread | 8 threads | 16 threads |
| --- | --- | --- | --- | --- |
| Vela 1.0, 29 tokens | 950 | **87.7 ms** | 191 ms | 157 ms |
| Vela 1.0, 29 tokens | 920 | 251 ms | — | **199 ms** |
| Vela 1.0, ~1,200 tokens | 950 | 1852 ms | — | **783 ms** |
| Vela 2.0, 662 tokens, 4 signals | 950 | — | **870 ms** | 1359 ms |

On the 950, one thread beats sixteen on the short input (the sweep's single
samples ranged 76–203 ms on a loaded host, so treat the 16-thread median as
directional), while the long input is monotonic in threads. On the quieter
920, sixteen threads are mildly better even at 29 tokens. On the 950, the
Vela 2.0 default's four-signal call at its production input size is 1.56×
faster at 8 threads than at the 16-thread default. No static thread count is
optimal across hosts and input sizes; the crossover moves with both.

### Responses are bit-identical across thread counts

Byte-comparing the same inputs across thread counts: Vela 1.0 at 29 and
~1,200 tokens for counts {1, 4, 8, 16}, and the Vela 2.0 four-signal
decisions call for counts {4, 8, 16} — every response pair is identical
(max float difference 0.0). GEMM parallelizes over output rows and the
attention and elementwise kernels over independent elements, so the thread
count does not reorder any output element's summation. Adapting the thread
count per request therefore cannot change an `exact`-profile answer.

### An eager OMP wait policy is not a fix

`OMP_WAIT_POLICY=ACTIVE` on the 920 turns the 16-thread short-input forward
from 190 ms into 4,977 ms (8 threads: 235 → 5,093 ms). Spinning siblings
hammer the barrier's atomics and interfere with the serial sections
(dispatch, allocation) between regions. The per-region cost is structural;
neither wait policy removes it. Fewer regions (fused or batched ops) or
fewer participating threads are the levers.

### Related: the packed linear on aarch64

Both `torch.ops.mkldnn` prepack operators exist on aarch64 builds; enabling
the `float32-packed` path there (the gate is only `platform.machine()`)
produces responses bit-identical to `F.linear` across 28 shape/length cells
on the 950, with no long-input regression on that host. The end-to-end
effect stays within ±8% because the small-input budget is dominated by the
overhead above, not by GEMM.

## Implications

- The request pipeline and the tokenizer are not worth optimizing for small
  inputs; the forward's per-op overhead is.
- An input-adaptive intra-op thread count is exact-safe and would have
  saved 1.56× on the Vela 2.0 default at its production input size on the
  950 — but the optimum is host-dependent, so an adaptive mechanism must
  calibrate on the host rather than ship a fixed table.
- Schema tokens (473 of 662 in the four-signal call) remain the largest
  single lever for the Vela 2.0 default's CPU cost.

Reproduction: the commands above with any local copy of the two models;
unique text per request (both caches on), one server per thread count,
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` on hosts without a usable NPU stack.
