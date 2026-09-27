# JPT-9B native JevArena v3 same-panel peer result

This is a size-matched **external peer**, not Decision 2.0 or an independent
unopened blind test. The fixed v3 labels had already been exposed for other
project arms before this prospective JPT prediction freeze. All 8,147
gold-free item predictions were sealed together before this peer was scored;
there is no same-panel Lux 1.0 result or paired interval yet.

## Identity and fixed protocol

| Field | Receipt |
| --- | --- |
| Weights | `kirp/jpt-9b@7114b0c3d9bea6b82dfa2d0691e8d5562cd26d4e`; 9,409,813,744 loaded parameters from safetensors headers |
| Native path | `llm2jev@2b252d504972764211ef172c1155ac0fedc9c3de`, native JPT adapter SHA-256 `1c1bfe9ced064e5a5175408dd76f317a8f7d88c810a8a765a37059053cdf90b4` |
| Gold-free panels | typed FINAL 1,600 items/2,000 answers, input SHA-256 `e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd`; CSS15 6,547 items, input SHA-256 `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6` |
| Joint prediction freeze | SHA-256 `0f1d9e32f38c9ebaa6aa20b726703a621ed00454e06b6cfecae08a319680cfc4`; typed prediction SHA-256 `49ebdc1f7be019babd5c058279ed76cdd7eb0681bc0b9c4c18437ad0d285a2c4`; CSS prediction SHA-256 `e97c7be9c13c10f7e87ba359a30a5d029a87b97abe32d0ce16a6a61dab36ab1d` |
| Scoring | Unchanged typed scorer SHA-256 `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`; CSS scorer SHA-256 `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca` |
| Aggregate report | Typed SHA-256 `3054aaf8b19ef7770a561c830238527a40fdaf7cfde3655a1553819c6b1f808b`; CSS SHA-256 `0623e1ef0962729f1dfb32b3409c03dc96eb8afae24cde3fa7d7817e1936fc77` |
| Compute | One isolated GPU; 32-second smoke, 132-second typed collection, 408-second CSS collection; 572 GPU-seconds or 0.1589 GPU-hour by recorded start/end times. The GPU is released. |

## Same-panel findings

| v3 component | Result |
| --- | ---: |
| Typed family macro accuracy, T | 74.875% |
| CSS15 task-median macro-F1, H | 49.687% |
| `100 × sqrt(T × H)` | **60.994** |
| Valid typed answers | 2,000/2,000 |
| Valid CSS answers | 6,547/6,547 |
| Typed Brier / ECE-10 | 0.1392 / 0.0576 |
| CSS15 task-median Brier / ECE-pmax-15 | 0.6078 / 0.1160 |

Typed per-type accuracy: Choice 715/800 (89.375%); Noul 656/800 (82.0%);
Score 227/400 (56.75%). Its four family accuracies are constraint competition
78.75%, evidence join 100%, exception stack 64%, resource ledger 56.75%.
The weakest CSS task macro-F1 scores include Tropes 10.14% and TalkLife
25.52%; the median task score is 49.687%. These failures matter despite the
high Choice/typed evidence result.

The **previously completed**, exact-model native public JevBench 231-item
result was reused, not rerun: 197/231 overall, easy 48/48, standard 68/72,
hard 81/111. Its input SHA-256 is
`642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd`
and report SHA-256 is
`5f34a80a850ef8a465e7a46e629534c964e5f7dcc2501a8387e75d5ec3dc622a`.
It is a public subset reproduction, not a closed or official ranking. The
same-panel v3 result cannot be mixed with older 1.0/card numbers to claim a
Decision 2.0 improvement or Pareto position.
