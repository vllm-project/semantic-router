# Decoder Milestone 11 — amendment 1: node F GPU layout (2026-10-01)

Written ≈07:15Z, before any M11 GPU job. Preregistration `b8c3caf22`, data lock `7c3aed8c8`.

**Finding (pre-launch check, 07:10Z).** Node F GPU4 and GPU5 each hold ≈250 GB of VRAM from a workload that is not
a docker container on the host and not under any lease entry (their lease owner files still read `dec-m10` idle from
M10); no process of ours runs there. Per the isolation rules M11 does not touch them. Node E is as expected (GPU4–5
vLLM servers, GPU6–7 the auto_map worker; GPU0–3 free).

**Change (layout only; arms, data, recipes, gates and budget unchanged).** Node F uses GPU2, GPU3, GPU6 and GPU7:

| Node | GPU | Chain |
| --- | --- | --- |
| F | GPU2 / GPU3 / GPU6 | `2b-LH` s1 / s2 / s3, then `08b-LH` s1 / s2 / s3 (GPU2 pre-warms each tier) |
| F | GPU7 | node-F readouts: the LH arms' post chains (C0 read on node F, merges, soups, readouts) |
| E | GPU0–3 | unchanged |

`m11-launch.sh` refuses node F GPU4–5. The 0.8B LH seeds start after the 2B LH seeds on the same GPUs, so stage 1 takes
longer in wall-clock; the GPU-hour caps are unchanged.
