# Official Qwen3.5-9B: short-TRAIN data and synthetic backward result

**Status: technical admission only; optimizer/release HOLD.** This was the
single CPU/GPU cell specified in the previously signed [prospective protocol](qwen35-9b-short-context-admission-2026-09-28.md).
No new 9B optimizer update, SELECT choice, CAL, DEV, CSS, public or formal
evaluation was performed. The earlier 8,192-token faulted arm is unchanged.

## Complete-group TRAIN construction

The pinned source configuration/tokenizer, original TRAIN7,455/SELECT700/CAL700
hashes, and exact checked-in data builder were verified before an offline CPU
run. The builder discarded every **complete original group** containing a
native input over 4,096 tokens. It did not truncate, alter labels or change
SELECT/CAL. Reload and split-isolation checks passed.

| Field | Result |
| --- | ---: |
| Original TRAIN | 7,455 rows / 5,392 groups / 4,194,465 tokens |
| Overlength rows and removed groups | 131 and 131 |
| Filtered TRAIN | 7,324 rows / 5,261 groups / 3,579,176 tokens |
| Maximum retained native input | 4,089 tokens |
| Filtered Choice / Noul / Score | 3,824 / 2,993 / 507 |
| Filtered English / Chinese | 6,018 / 1,306 |
| Filtered TRAIN SHA-256 | `fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c` |
| Private manifest SHA-256 | `cebf07cd3db08c7e875f73e3c2fb405b6dcc5a370c55fa6f4e11c131707a9560` |

Of the 131 removed rows, 130 were from the older Stage4 composition source
and one from Stage3 replay. All 40 original family names retain at least one
TRAIN row. However, removing 1.76% of rows removed **14.67% of TRAIN tokens**,
including many long arithmetic, table and relation cases. Any later 9B gain
would conflate shorter training exposure with other effects; no broad long-
input improvement can be inferred from this subset.

## Single no-optimizer GPU cell

The exact signed code at `37ccb75b8` was mirrored by file hash. The diagnostic
runner SHA-256 remained `bf99ed3ada1ed10bd607dc726f0dc98813962c812fa63332a3d87b03637ad857`;
the official Qwen source revision was `c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
with the same four pinned weight-shard hashes as the prior 9B admission. The
existing update-64 **diagnostic** checkpoint was used read-only. One isolated
accelerator and pinned training image ran the fixed 20-iteration synthetic
cycle `[512, 1024, 2048, 4096]` with gradient checkpointing disabled.

| Gate | Observation |
| --- | --- |
| Container exit / OOM | 0 / false |
| Iterations | 20/20 finite forward/backward, no optimizer |
| Maximum allocated HBM | 107.987 GiB |
| Trainable weights after run | Bitwise unchanged |
| Container start→die | 50.858 seconds, 0.01413 GPU-hour |
| Source/checkpoint/code/data pre/post SHA | All matched |
| Private raw-log SHA-256 | `4bd657243f4a2414fdf563f1005f6f57b5e1da124f9c4017f7599a210acf092c` |

The standalone no-checkpoint cell passed; it only narrows the failure surface.
It does not prove a gradient-checkpointed full trainer will pass at 4,096
tokens, nor does it identify the 6,144-token faulting operator. GPU resources
were released. The next permitted step is a **newly frozen one-update/reload
smoke** with the filtered TRAIN and original SELECT; a full training arm needs
its own gate and budget after that. The 9B first-release state remains HOLD.
