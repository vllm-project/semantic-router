# 0.6B type-separated head: matched-data development result

**Decision: HOLD.** The single prospectively registered 466-update arm finished,
but its prelaunch SELECT advancement gate failed. This is a development result,
not a JevArena, JevBench, or release result. The existing first-release 0.6B
package and its evidence are unchanged.

## Frozen comparison and integrity

The treatment replaced one shared candidate head with independent Choice,
Noul, and Score heads. Both arms start from official
`Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`,
use the same rights-clean v2 TRAIN 7,455 rows / 4,094,489 tokenizer-measured
tokens and SELECT/CAL 700 rows each, and have the same row order, one epoch,
466 updates, objective, optimizer, prompt, 8,192-token cap, and native scoring
contract. The completed shared-head arm was reused without retraining. The
treatment changes head parameter count and is not parameter matched.

Local integration is signed at commit `4361f753543675cbdbc75b6964144e7f73006368`.
The immutable technical lock SHA-256 is
`4b1a18f5492c8cc4706328651f1b5254493875d09467bdb7a2944318311b2568`.
The exact official source files passed cross-node byte verification, followed
by a CPU/no-device audit and a one-GPU technical gate. The latter passed both
zero-step source identity and one-update saved/reloaded native-output parity;
no category changed and maximum probability drift was zero in the prescribed
checks. The separately frozen full-arm launch receipt SHA-256 is
`78c31caa28779384c9197d1af69a6c535749d0957ebedf87c4ad106d0ef10b35`.
It fixed the image, code/data/source hashes, one GPU, an unshared output,
no network, the 2.0 GPU-hour watchdog, and the following SELECT gate **before
the first optimizer update**: selected checkpoint at least 562/700 correct,
family-macro accuracy at least 0.77259, all eight prescribed milestones,
finite metrics and valid native answers. No threshold was changed afterward.

## Complete SELECT trace

The registered selection order was family-macro accuracy descending,
normalized Brier ascending, then earliest step. The host validated all 700
unique, valid native answers and the checkpoint receipt at every milestone.

| Update | Correct / 700 | Six-family macro accuracy | Six-family macro Brier |
| ---: | ---: | ---: | ---: |
| 64 | 382 | 0.461873 | 0.289987 |
| 128 | 439 | 0.554494 | 0.256120 |
| 192 | 472 | 0.627657 | 0.232993 |
| 256 | 513 | 0.697913 | 0.195305 |
| 320 | 544 | 0.742685 | 0.156851 |
| 384 | 552 | 0.744259 | 0.146843 |
| **448 (selected)** | **553** | **0.745833** | **0.146686** |
| 466 | 553 | 0.742778 | 0.149820 |

At the selected step, the six SELECT families were:

| Family | Correct / items | Accuracy |
| --- | ---: | ---: |
| Human GoEmotions Choice | 155/200 | 0.775 |
| Human GoEmotions Noul | 175/200 | 0.875 |
| Narrative reading | 130/130 | 1.000 |
| Open-world abstention | 39/40 | 0.975 |
| String composition | 18/40 | 0.450 |
| Quantized median / ordinal Score | 36/90 | 0.400 |

The selected treatment misses the shared-head control by **9/700** correct
and **0.02676** family-macro accuracy, and misses both prospective advancement
thresholds. The easy narrative and abstention families are nearly saturated;
string composition and ordinal Score remain weak. These SELECT patterns are
not evidence that separate heads harmed independent transfer: no protected
or development extension panel was opened for this failed arm. The conclusion
is narrower: type separation alone does not meet the fixed development gate
under this exact data, budget and scorer.

## Resource, failure boundary and next experiment

The single run took **1,120.02 seconds / 0.31112 GPU-hours** on one GPU,
including all eight SELECT checkpoints. Training finished 466 updates and
the container exited normally. The final host receipt SHA-256 is
`0d03b313532643abd9d737253e53babae660eb0cf9b69b2255bb439f105f1e80`;
COMPLETE and BEST receipt SHA-256 values are
`9d18312510857498eedc0626fb4f746da0efdb2180d3686ec2a9ec4d4d093cc0`
and `0d1c5055fb324d60b237652e90c6187136170ed4d9a6673ee89666826fa89420`.
The host confirmed the task container was removed and the reserved GPU was
idle afterward. SELECT advancement was marked **HOLD**; there was no retry,
off-plan checkpoint choice, CAL fit, typed DEV/CSS inspection, JevBench,
JevArena, HF upload, or change to the first-release package.

The next most discriminative data experiment is the separately outlined
[shared-head Score contrast D](small06-type-separated-head-prereg-2026-09-27.md):
keep the completed official-Qwen shared-head setup, replace 384 existing
Score rows with 128 independently reviewed three-level groups, keep 7,455
TRAIN rows and match 4,094,489 training tokens within 0.5%, then use the
same frozen SELECT rule and native scorer. That isolates ordinal data coverage
from the failed head-sharing hypothesis. D is **not GPU-ready**: its original
groups, oracle, source rights, review, overlap audit and exact roster/hash
must be completed and frozen first. No larger run should be launched from
the twelve-group audit pilot or by recycling this failed treatment.
