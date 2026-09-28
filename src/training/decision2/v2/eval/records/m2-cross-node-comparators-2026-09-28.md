# Cross-node comparators: Lux1 frozen-cache check and AutoJev 27B on node B

Eval & peers track, Milestone 2 (amendment A3). Post-key same-panel evidence.

## N1 — Lux1 on node B with node A's frozen autotune cache

Node B GPU3, image `decision20-lux-runtime:latest` (`sha256:ce895822…`, package trees
byte-identical to node A's image), a copy of node A's frozen Triton autotune cache
(tree `e215f8bd…`); no new autotune entry was created (6 before, 6 after).

| Comparison | Answer-category changes (of 8,778) | Max numeric drift |
| --- | ---: | ---: |
| N1 (node B, frozen node A cache) vs r4 (node B, earlier, own autotune) | **0** | 0.0 |
| N1 vs D1 (node A) | 43 (typed 7, CSS 36, public 0) | 0.190 |

**A1's autotune hypothesis is falsified.** With identical kernel configurations node B
reproduces its own earlier numbers bit for bit and still differs from node A. Both nodes
are internally deterministic (node A: GPU6 = GPU7 across runs; node B: GPU7 = GPU3). No
difference was found in host kernel, amdgpu driver 6.19.14.31400000, GPU firmware
versions, static GPU properties (gfx942, 304 CUs, part number), CPU model and ISA flags,
image package trees or package bytes. The cause sits below what we can observe; the
rule is operational: **a candidate and its comparator must run on the same node.**
Lux1 comparators: node A 65.808 (public 183/231), node B 66.268 (public 183/231).

## N2 — AutoJev 27B on node B (27B formal node)

Package `denis-pplx/autojev-27b@6f5b557e` re-downloaded on node B, native runtime
`ee63c151`, same adapter, image `ce895822…`, persisted autotune cache; formal and dev
panels in one run (1,197 s).

| Metric | Node B (N2) | Node A (earlier peer run) |
| --- | ---: | ---: |
| v3 | 72.310 | 72.310 |
| T / H | 0.886875 / 0.589571 | identical |
| Public 231 | 200 | 200 |
| Answer-category changes vs node A | 40 (typed 11, CSS 28, public 1) | — |

The aggregates coincide although 40 answers differ. **27B formal runs must use node B**
(where the 27B track trains) with the N2 run (`/data/dev2/runs/eval/m2/n2-autojev27-nodeB`
on node A for scoring; predictions originate on node B) as the same-node comparator.

GPU time: N1 420 s (0.117 GPU-hour), N2 1,197 s (0.333 GPU-hour).
