# Agent sessions: CPU KV tier and replica placement

Coding agents resend their whole conversation on every call. In the SemiAnalysis Claude Code sessions, 98% of a
main agent's prompt tokens are a prefix of the same session's earlier calls, so what an agent deployment can
serve depends on whether a returning session finds its prefix. These scripts measure two levers below the
Router's model choice on one GPU:

- a CPU tier for the KV the GPU prefix cache would evict (LMCache), and
- where a session's requests land when a model has several replicas (the Router's `lb_policy` against
  session-aware placement).

They back the serving recommendations in the coding agents proposal. Nothing here runs in CI except the unit
tests, because the runs need a GPU.

## What runs

| Script | Measures |
| --- | --- |
| `setup.sh` | Installs LMCache (built with HIP on ROCm), WEKA's kv-cache-tester, Qwen3-14B and the traces under `WORK` |
| `run_prefix_ttft.sh` | Time to first token for 4k to 32k-token prompts: recomputed, from the GPU prefix cache, from LMCache's CPU tier |
| `run_trace.sh` | The SemiAnalysis sessions replayed against one server, with the GPU prefix cache alone and with LMCache added |
| `run_placement.sh` | The same sessions against two replicas behind `placement_proxy.py`, one arm per placement policy |
| `report.py` | Requests, input tokens and TTFT per arm from the tester's results, and per server its KV pool, last prefix-cache hit rates and mean queue |

The tester adds simulated users while the average time to first token stays under 60 s (start 4, at most 16
or `MAX_USERS`, think-time pacing, 50% warm prefix, contexts capped at 100k tokens, seed 0) for 20 minutes per
arm, so an arm's result is the load it carried at that latency limit. The load is adaptive, so the arms run
different numbers of sessions and per-request TTFT is not comparable across arms.

Placement policies in `placement_proxy.py`: `round_robin` and `least_request` are the two the Router offers for a
model's backends. `session` hashes the `X-Session-ID` header the tester sends. `sticky` sends a new session to the
replica whose recently active sessions sent the fewest prompt bytes on their latest request, and keeps it there.

## Versions

| Component | Version |
| --- | --- |
| Traces | `semianalysisai/cc-traces-weka-with-subagents-060526`, revision `8a5df339`, Apache-2.0, 336 sessions, no text |
| Load generator | `callanjfox/kv-cache-tester`, commit `94f6046d`, Apache-2.0 |
| KV tier | LMCache `v0.5.6rc5`, Apache-2.0, built with HIP for gfx942 |
| Server | vLLM 0.29.1rc1 ROCm image (torch 2.12, ROCm 7.2) |
| Model | Qwen3-14B, revision `40c06982`, extended to 131,072 tokens with YaRN |
| Hardware | One AMD MI300X (192 GB) |

## Run it

Inside a vLLM container with a GPU:

```bash
WORK=$PWD/work bash bench/agent_kv_tier/setup.sh
WORK=$PWD/work bash bench/agent_kv_tier/run_trace.sh
WORK=$PWD/work MAX_USERS=4 bash bench/agent_kv_tier/run_placement.sh
python3 bench/agent_kv_tier/report.py arms work/trace/gpu work/trace/lmcache
```

`MODEL`, `TRACES`, `TESTER_DIR`, `DURATION`, `MAX_USERS`, `ARMS` and `LMCACHE_CPU_GB` override the defaults in
`common.sh`. The scripts start and stop their own servers by process id and use ports 8000 to 8002.

## Results

### CPU tier on one GPU (`run_trace.sh`)

| Run | Arm | Requests | Input tokens | TTFT p50 / p95 / max (tester) | Tester cache hit | Last logged prefix hit |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | GPU prefix cache | 94 | 3.98M | 4.2 / 75.2 / 342.4 s | 72.4% | 51.7% |
| 1 | + LMCache CPU 64 GB | 188 | 8.75M | 8.6 / 89.3 / 105.5 s | 79.1% | 35.0% GPU, 10.0% LMCache |
| 2 | GPU prefix cache | 101 | 4.28M | 4.4 / 111.6 / 210.0 s | 73.7% | 50.4% |
| 2 | + LMCache CPU 64 GB | 191 | 8.96M | 9.7 / 56.9 / 78.7 s | 80.2% | 32.8% GPU, 16.3% LMCache |

The CPU tier carried 2.2 times the input tokens at the same latency limit in the first run and 2.1 times in the
second, which ran from these scripts in a fresh container, and cut the worst wait from 342 s to 106 s and from
210 s to 79 s. The GPU-only server's peak working set was 1,027,968 and 1,030,336 tokens against a GPU KV pool of
801,376 tokens (0.85 of memory), so it evicts prefixes and recomputes them when a session returns, while the CPU
tier returns them.

### Raw cache speed (`run_prefix_ttft.sh`, median of 5)

| Prompt | Recomputed | GPU prefix cache | LMCache CPU tier |
| --- | --- | --- | --- |
| 4,096 | 0.247 s | 0.034 s | 0.028 s |
| 8,192 | 0.535 s | 0.053 s | 0.038 s |
| 16,384 | 1.270 s | 0.099 s | 0.058 s |
| 32,768 | 3.574 s (2.913 s on the LMCache arm's first send) | 0.169 s | 0.098 s |

### Replica placement (`run_placement.sh`)

Two replicas at 0.42 of the GPU each (264,256 KV tokens each). With up to 16 sessions, which need far more KV than
the two pools hold:

| Arm | Requests | Input tokens | TTFT p50 / p90 / max | Replica 1 prefix hit, mean waiting | Replica 2 |
| --- | --- | --- | --- | --- | --- |
| round robin | 94 | 3.89M | 26.8 / 270.1 / 378.1 s | 0.0%, 7.5 | 33.1%, 0.4 |
| least request | 83 | 3.35M | 88.7 / 141.1 / 169.6 s | 0.8%, 2.8 | 0.3%, 2.9 |
| session hash | 93 | 3.68M | 32.6 / 278.5 / 332.2 s | 0.0%, 6.2 | 46.4%, 0.9 |
| sticky | 102 | 4.24M | 50.4 / 202.8 / 315.6 s | 0.0%, 3.4 | 1.7%, 2.3 |

With at most 4 sessions, which fit the two pools:

| Arm | Requests | Input tokens | TTFT p50 / p90 / max | Prefix hit, replica 1 / 2 |
| --- | --- | --- | --- | --- |
| round robin | 121 | 7.43M | 1.6 / 9.9 / 36.4 s | 77.0% / 82.0% |
| sticky | 139 | 8.73M | 1.0 / 5.3 / 28.5 s | 91.8% / 88.0% |

When the sessions fit, keeping each on one replica after a load-aware first placement raised the prefix hit rate
from about 80% to about 90%, cut the median time to first token from 1.6 s to 1.0 s, and carried 17% more input.
Oversubscribed, sticky placement carried 9% more input than round robin through balance rather than cache hits,
and a plain hash of the session id was worse than round robin, because it ignores how full each replica is.

## Limits

One GPU and one model. The CPU tier comparison ran twice, the cache speed and placement runs once per arm. The two replicas share one GPU's compute, so the placement runs measure
placement and a split KV pool, not added compute. The tester's load is closed loop, so a faster arm also reaches
further into the traces.
