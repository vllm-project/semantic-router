# Decision 3.0 computes in BF16 on a CPU class with its own BF16 references

Since the [FP32 record](d3-cpu-fp32.md), `--dtype auto` computes every
Decision 3.0 size in FP32 on a CPU, because one CPU class's BF16 answers fail
another's references (#5273). That record also measures what FP32 costs where
the CPU computes BF16 natively: 1.2 to 3.5 times the time per request and
about 1.7 times the memory. This change gives that back where it is checked.
The CPU key now names the x86 vendor, and `--dtype auto` computes BF16 on a
CPU that computes BF16 natively and whose key has a BF16 reference for every
golden request. Today that is an AMD CPU with AVX-512 BF16, for all eight
sizes. Every other CPU keeps FP32, so no CPU is checked against another
class's BF16 answers.

BF16 is Decision 3.0's published precision, the one it computes in on GPUs and
NPUs. On the typed-decisions TEST set it decides 99.35% to 99.65% of decisions
as FP32 does, with the same accuracy, and serves 1.8 to 3.6 times faster in
0.55 to 0.70 times the memory.

- **Date:** 2026-11-18.
- **Hosts:** AMD EPYC 9755 (Zen 5, Turin) and AMD EPYC 9654 (Zen 4, Genoa),
  AVX-512 with BF16, no AMX; one check of this change on an AMD EPYC 9454
  (Zen 4).
- **Image:** Python 3.12 with PyTorch 2.14.1+cpu, the version the CPU images
  pin, and the runtime's lock. tcmalloc is preloaded.
- **Runtime:** this change's base, served at `--dtype bfloat16` (BF16) and
  `--dtype auto` (FP32). On an AMD CPU this change's `--dtype auto` holds the
  same BF16 weights and kernels: d3-edge's golden answers at `auto` with this
  change equal those at `--dtype bfloat16` without it, bit for bit (EPYC 9454,
  8 threads).
- **Packages:** the eight revisions `registry/tables/decision3.py` pins.
- **Threads:** 32 pinned cores per run unless stated.

## The key names the vendor

`CPUAccelerator.golden_reference` appends the x86 vendor read from
`/proc/cpuinfo` (`amd`, `intel`), for example
`cpu:x86_64:avx512:bf16_native:amd`. The BF16 references of the
[PyTorch 2.14 record](d3-cpu-goldens-pytorch-2.14.md) under
`cpu:x86_64:avx512:bf16_native@bfloat16` were recorded on an AMD EPYC 9575F,
so they move to `…:amd@bfloat16` with their values unchanged. A host checks
its vendor's key, then the key without a vendor, then `cpu`, so references
recorded before this change still check the hosts they checked.

## Which CPUs compute BF16 at `auto`

| Host | Key | `auto` computes | Checked against |
| --- | --- | --- | --- |
| AMD, AVX-512 BF16 | `cpu:x86_64:avx512:bf16_native:amd` | BF16 | `…:amd@bfloat16` |
| Intel, AMX | `cpu:x86_64:avx512:bf16_native:intel` | FP32 (no reference yet) | `cpu` |
| AVX2 (`ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2`) | `cpu:x86_64:avx2:bf16_emulated:amd` | FP32 (BF16 emulated) | `cpu` |
| Any CPU, `--dtype float32` | | FP32 | `cpu` |
| Any CPU, `--dtype bfloat16` | | BF16 | its class's BF16 reference |

The AVX2 references at `@bfloat16` stay recorded, but an AVX2 CPU emulates BF16,
where FP32 is 3.0 to 5.8 times faster, so `auto` keeps FP32 there.

## Decisions

The typed-decisions TEST set (2,000 decisions: 600 Choice, 600 Noul, 800
Score) at each precision on Turin, one request at a time. A decision is a
question's most probable answer; accuracy counts decisions that match the
labelled one.

| Size | Changed | Accuracy FP32 | Accuracy BF16 | Right → wrong | Wrong → right | McNemar p | max \|Δp\| | p99 \|Δp\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d3-edge | 7 | 0.4210 | 0.4230 | 1 | 5 | 0.22 | 0.035 | 0.019 |
| d3-lite | 9 | 0.5225 | 0.5210 | 5 | 2 | 0.45 | 0.022 | 0.015 |
| d3-nano | 13 | 0.6185 | 0.6225 | 1 | 9 | 0.02 | 0.048 | 0.017 |
| d3-mini | 7 | 0.7320 | 0.7315 | 3 | 2 | 1.00 | 0.036 | 0.012 |

Changed decisions by question type:

| Size | Choice (of 600) | Noul (of 600) | Score (of 800) |
| --- | ---: | ---: | ---: |
| d3-edge | 5 | 0 | 2 |
| d3-lite | 2 | 1 | 6 |
| d3-nano | 2 | 1 | 10 |
| d3-mini | 3 | 0 | 4 |

No size loses accuracy in BF16. Each accuracy difference is within ±0.022,
the 95% interval of one accuracy over 2,000 decisions. The one significant
difference (d3-nano, p = 0.02) favours BF16: nine decisions become right and
one becomes wrong. Agreement per question type is at least 99.17% except
d3-nano's Score questions, at 98.75% (10 of 800).

Unlike the FP32 record's 5 golden requests and 19-request panel, where no
decision changed, the full set changes 7 to 13 decisions per size. This
change keeps the published precision rather than consenting to a reduced
copy or an approximate kernel, so the rule it answers to is the AGENTS.md
one: no accuracy loss.

## Batched as served

The same set at concurrency 8, each precision against its own answers one at a
time:

| Size | FP32 changed | BF16 changed | Accuracy FP32 | Accuracy BF16 | BF16 McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| d3-edge | 0 | 4 | 0.4210 | 0.4225 | 1.00 |
| d3-lite | 0 | 13 | 0.5225 | 0.5230 | 0.34 |
| d3-nano | 0 | 6 | 0.6185 | 0.6210 | 0.38 |
| d3-mini | 0 | 3 | 0.7320 | 0.7320 | 1.00 |

FP32 is batch-invariant. BF16 batching changes 3 to 13 decisions, with no
accuracy change. The golden check plans each request alone, so it checks the
one-at-a-time answers; this table covers the batched ones.

## One class, one answer

BF16 golden answers against the AMD class's reference (recorded as
`cpu:x86_64:avx512:bf16_native@bfloat16`, now `…:amd@bfloat16`):

| Host | Sizes | Threads | Golden check | Largest difference |
| --- | --- | --- | --- | ---: |
| EPYC 9755 (Zen 5) | d3-edge, d3-lite | 1, 8, 32 | matched | 0 |
| EPYC 9755 (Zen 5) | d3-nano, d3-mini | 32 | matched | 0 |
| EPYC 9654 (Zen 4) | d3-edge, d3-lite | 8, 32 | matched | 0 |
| EPYC 9654 (Zen 4) | d3-nano, d3-mini | 32 | matched | 0 |
| EPYC 9454 (Zen 4), this change at `auto` | d3-edge | 8 | matched | 0 |

Two EPYC generations reproduce the reference bit for bit at every thread
count tried.

With AVX2 forced on the EPYC 9755, `--dtype auto` computes FP32 and matches
`cpu` to within 1.5e-6 (d3-edge, d3-lite).

## Speed and memory

The FP32 record's protocol on the EPYC 9755: peak resident memory after
loading and the golden check, and the median of 15 runs of a short request
(a one-sentence state) and a long one (an 80-sentence state, about 700
tokens), each with the same Choice, Noul and Score questions.

| Size | Peak memory FP32 | Peak memory BF16 | Short FP32 | Short BF16 | Long FP32 | Long BF16 | BF16 speed-up (short / long) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d3-edge | 3.1 GiB | 2.1 GiB | 182 ms | 95 ms | 978 ms | 518 ms | 1.91x / 1.89x |
| d3-lite | 4.5 GiB | 3.1 GiB | 402 ms | 222 ms | 2,219 ms | 1,191 ms | 1.81x / 1.86x |
| d3-nano | 10.8 GiB | 7.3 GiB | 761 ms | 306 ms | 4,141 ms | 1,704 ms | 2.49x / 2.43x |
| d3-mini | 19.8 GiB | 12.0 GiB | 1,963 ms | 695 ms | 11,506 ms | 4,015 ms | 2.82x / 2.87x |
| d3-flash | 32.5 GiB | 17.9 GiB | 3,455 ms | 974 ms | 19,144 ms | 6,204 ms | 3.55x / 3.09x |

The larger the size, the more BF16 saves. The FP32 record measures the 26B
sizes at 98.5 GiB in FP32 and 50.6 GiB in BF16.

## Not measured

- An Intel Xeon with AMX: no reference is recorded under `…:intel@bfloat16`,
  so it computes FP32 at `auto`, as before. Recording its references with
  `tools/golden_answers.py --dtype bfloat16` on a Xeon turns BF16 on there.
- ARM CPUs (NEON, SVE): their keys name no vendor and have no BF16 reference.
- The TEST set for d3-flash and the 26B sizes, and their golden answers across
  threads and hosts.
- The Vela 3.0 d3 sizes: they define no golden requests, so they have no BF16
  reference and stay FP32.
