# JevK5 native DEV/CSS pilot panel (2026-09-26)

This is development evidence for Decision 2.0 model selection, not a release
claim. The unified synthetic DEV has 1,600 questions; the CSS pilot has 1,430
human-labeled items across three tasks. Neither final split was read.

## Frozen inference contract

- Native runtime: [`allebee/jevk5`](https://github.com/allebee/jevk5) commit
  `1e5ae1b533b9eb80c0cbe3fbd010607d0b4e26ae`, clean checkout, `JevK5`
  with `graphs=False`, BF16 ROCm eager execution, released temperatures and
  option readout. The adapter calls `JevK5(local_path, graphs=False).decide(...)`
  once per typed question; model load is excluded from per-item latency.
- Collector: `inference/jevk5.py` signed source commit `1230f34`; adapter
  version `jevk5-native-eager-v1`. The runtime source digest in every receipt
  is `ce5340940b7a756da632e0095e3252d0713b73a5c130ac4cc2287dbbc46308d7`.
  Each row binds the prompt payload, model revision, release files, runtime,
  answer map, latency, and token usage. All 3,030 predictions per model have
  native typed answers and no generated output tokens.
- Model revisions and weight SHA-256:
  [`JevK5-2B`](https://huggingface.co/alibiserikbay/JevK5-2B)
  `7922d1f55df137b72ef763fced56fd09efc5e99d`,
  `847f8b2e6016f1c779cd01ff75428333ad2e9ab491853c36e9521529caf39892`;
  [`JevK5` 4B](https://huggingface.co/alibiserikbay/JevK5)
  `c4f7fdb3aeab5582336406e78d3bef11bf98833d`,
  `13824e47f2e40fe052f06943976cf742cb366ba305741a111e75a8ebae907a9c`;
  [`JevK5-9B`](https://huggingface.co/alibiserikbay/JevK5-9B)
  `d6521a18a86999190e9d775c915af3d6d6772fc4`,
  `7060dee98993bb70865817007870d557e3dd42700d36b0e7c367c847bcfd2e8a`.
  The local download metadata attest these revisions. Released SHA256SUMS
  verified 4B/9B files; 2B has no released SHA256SUMS, so the collector hashes
  its required files directly.
- A gold-free Score smoke passed for each size. The native `score` equals
  `sum(level * probability[level])` exactly in all three samples. The 4B
  sample probability sum was `1.000000067`; 9B `1.000000007`; 2B
  `1.000000045`. Thus no modal-level projection was substituted for the
  shared expected-value Score contract.
- Scorers: `typed-decision-report/2` and `css-transfer-score/2`. DEV gold
  SHA-256 `c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc`;
  CSS pilot gold SHA-256
  `9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391`.
  All three models returned 1,600/1,600 valid DEV and 1,430/1,430 valid CSS
  pilot items. CSS pilot labels are development only.

## Unified results

DEV uses point accuracy, half-scaled multiclass Brier, and 10-bin ECE. CSS
uses micro accuracy, median task macro-F1, full-sum Brier, and 15-bin pmax ECE.
The two Brier scales must not be compared across panels.

| Model | DEV acc | DEV Brier | DEV ECE | CSS pilot acc | CSS F1 | CSS Brier | CSS ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| JevK5 2B | 44.50% | .30379 | .21657 | 40.14% | .34163 | .75433 | .07836 |
| JevK5 4B | 76.94% | .13995 | .07139 | 53.36% | .54071 | .60303 | .10367 |
| JevK5 9B | 85.44% | .10162 | .02045 | 51.75% | .51639 | .59501 | .16638 |
| Decision 1.0 Lux 9B | 86.75% | .09040 | .04462 | 53.78% | .57011 | .58862 | .15691 |
| Jev 1.13 official | 88.75% | .07509 | .02323 | 59.16% | .60890 | .5470 | .1275 |

The first three rows were newly collected with this adapter. The Lux and Jev
rows are the already frozen v2 baseline panel; hosted Jev latency includes
network transit and is not compared with local runtime latency.

| Model | Attribute gate | Rule precedence | Set reconciliation | Transition table |
|---|---:|---:|---:|---:|
| JevK5 2B | 79.75% | 51.00% | 24.25% | 23.00% |
| JevK5 4B | 95.50% | 53.50% | 79.50% | 79.25% |
| JevK5 9B | 100.00% | 57.50% | 89.00% | 95.25% |
| Decision 1.0 Lux 9B | 100.00% | 64.50% | 82.75% | 99.75% |
| Jev 1.13 official | 100.00% | 66.50% | 89.25% | 99.25% |

On CSS pilot, JevK5 9B versus 4B accuracy was discourse 54.33% versus
53.52%, implicit hate 43.57% versus 42.57%, and SemEval stance 58.16%
versus 65.52%. The stance loss outweighs the other two gains in the pilot
micro score. JevK5 9B nearly matches Jev on synthetic Score/set
reconciliation, while rule precedence remains 7 percentage points behind
Lux 1.0. Its strong DEV ECE does not imply stronger CSS calibration.

## Training evidence and limits

The pinned [4B](https://huggingface.co/alibiserikbay/JevK5) and
[9B](https://huggingface.co/alibiserikbay/JevK5-9B) cards state the same
47,460 training rows: 17,408 teacher-written questions (from Qwen3.6-27B and
GPT-6 Luna) plus 30,052 public train-split replay items. They use SemIf
letter-logit readout, CE, and published temperatures. The 9B ran 1.5 epochs,
the 4B one epoch; the 2B is an older v0.2 recipe. Cross-size deltas therefore
do not isolate model capacity. The data include synthetic stance/sarcasm and
iSarcasmEval train; CSS pilot is related-domain development evidence and may
share task-family structure. Public JevBench and Decision Index outcomes
should be treated as external context, not transported to our panel.

JevK5 9B shows useful set reconciliation and probability calibration, but its
stance transfer regression illustrates a training mixture or readout failure
that scale alone did not fix. This favors selecting Decision 2.0 checkpoints
on both structured and human transfer families, fitting calibration on
nontrivial native-type CAL, and avoiding a single benchmark or single size as
the release gate. The held-out final split is required for a release claim.

## Immutable local report hashes

| Model | DEV prediction | DEV v2 score | CSS prediction | CSS v2 score |
|---|---|---|---|---|
| 2B | `f4766611dfca0aad686973790578735d4f2e9d2c701772f283c5d6d344d1547e` | `fd9c127bb62ceb686f35224664e5e6cbde230425a4e925d854d87bbebc0dac39` | `fb2673e39f46011849be74381fb33690890e1ba069c9891146d1ae85739260ad` | `2f2e46ea45c651f32851ad1c31317486ea5facc5ee5a8581f9e32e6a80845344` |
| 4B | `b440d01ed5d2827c5b7c5ba93f57c048e6b71e611369bde53e3109c97f76f81e` | `402861d10d14c6f3c430f46004317303327f43da878c964de65d8331526bb011` | `c75671d7071692857cd8d2b51a1cd3e0f7334d3478f3bddd45a29e1faddce238` | `e89a67a63a6d6cfbc6eb022646e28e6d88f38fdc16406f843e31dddfa1d0c0c5` |
| 9B | `9cbc79ccdf73001a8a333c25216c19e9078f5b6b6cd8277aa1967579dffb59b8` | `767a17986953ea50ff4afa65b920ddad39a7cb1e0628757823dcfee8c1cf9ace` | `3d008dd08b7f15eae9d463d0da5557eda2cb4478e622ca1db03ef56bd817b033` | `ac45be7d0c0c6e41ef1eaf170550c77c8bdc0452803ba132aafe448e5124f441` |

The raw receipts and scoring reports are in the task-owned experiment
workspace. Hashes are supplied here so the unified gist can cite the exact
artifacts without exposing private infrastructure paths.
