# Hard multilingual DEV r6: independent blind and post-key audit

Status: **BLOCK_FOR_INFERENCE**. This is an editorial assessment of a private
development pilot, not a model score or release benchmark.

The r6 gold-free packet SHA-256
`9b16c9ba50f51c52108732f0fdb3221580f892758e855f410839503160087824`
matches the frozen manifest. An independent reviewer assessed all 72 rows
from 18 base cases across English, Chinese, Spanish and Japanese without
access to the casebook or targets. The row, group, summary and seal SHA-256
values are respectively
`1bf6b4429177fd76df00364c962811b6f985d2c11675e9d08633cc05e215c61b`,
`4d52a64506a8bd7731780f57d77406ae41e0cd815a548e894278c0e5829147ce`,
`ed4746cb82a8a03b4becb31c5e7b44d1af416974b7e2462f48731679790e1800`
and
`d5f4aab20460b4ff3a3044e484e3a83478ad7f2f541c9d08b48b5944529ffaa5`.
The blind seal records 2026-09-26 21:09:58 UTC and `key_access=none`.

The signed local seal-first verifier source SHA-256 is
`75abef97a37491e7d7ddc0e669df1cd24712931dc90f3a3b0f5eeca9e248c21f`.
It verified all frozen hashes, row/group identities and file chronology;
wrote a separate blind-verification receipt SHA-256
`cf150e3b4f28872af55fa204e241a3fbb8598fa71794dd20c6f5a811df40fe08`;
and only then opened the private target. The private aggregate receipt SHA-256
is
`240c0217f2573e57a6a658d3a9a8046ca0979c816cd923d550495bb71da9419c`.
No raw prompt, answer key or private machine location is present here.

The reviewer marked **68/72 rows** and **17/18 base groups** as unambiguous;
all 68 resolved row answers and all 17 resolved group answers matched the
mechanical oracle. The remaining four language variants of one base group
were deliberately unresolved, not wrong predictions: an expiry day equal to
the decision day lacked an explicit inclusive/exclusive rule. All 24 Choice
native option keys matched the oracle. The reviewer found no material
translation error, but was not a native Chinese, Spanish or Japanese reviewer,
so linguistic quality is not independently certified.

A separate shortcut remains: all six Choice bases resolve to A, B or C, and
none to HOLD. A strategy that never chooses HOLD therefore evades an intended
abstention test. The Chinese mitigation-credit wording is also awkward in
two Score rows, although both use zero credit and the reviewed answers do
not change. These are reasons to revise the panel, not post-hoc adjustments
to r6. The frozen packet, key and failed judgment remain unchanged.

**Disposition:** r6 remains private DEV and `inference_eligible=false`.
No model weights were loaded, no training or GPU inference was run, and no
multilingual performance or release claim follows from this editorial audit.
