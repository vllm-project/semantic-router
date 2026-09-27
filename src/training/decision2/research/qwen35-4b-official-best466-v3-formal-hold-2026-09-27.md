# Official-source Qwen3.5 4B: fixed v3/public-231 result

**Decision: HOLD.** The official-source BEST466 candidate passed its
[development screen](qwen35-4b-official-base-full-development-result-2026-09-27.md)
by 12.921 points, but that gain did not transfer to the locked
[JevArena v3 comparison](qwen35-4b-official-best466-v3-postkey-lock-2026-09-27.md).
It scored **53.218** against our Decision 1.0 Nox 4B's **56.470**, missing
the prospectively fixed **Nox +3.0** release gate. Do not materialize or
publish this candidate, change its checkpoint or calibration, or tune against
these formal labels.

## Identity and scoring chronology

The candidate starts directly from official `Qwen/Qwen3.5-4B-Base` revision
`1001bb4d826a52d1f399e183466143f4da7b741b` with a new decision head
and rank-16 LoRA. SELECT alone chose completed update 466. The native loaded
model has 4,240,848,384 parameters and fingerprint
`dc2a8267ec7315a48aa1b2c31e7157e79e50582c4061177bed32ca0ed2735372`.
The CAL-only temperatures were Choice `1.217267172620981`, Noul
`1.044386023961823`, Score `0.05`. The lower-bound Score temperature is
retained as fitted, despite the transport risk.

The prospective post-key lock SHA-256 was
`ab2602d5a1aa7b3284a61f207fec45c7758b2ba879c12680e5fd8308b4fbd494`.
Candidate and archived Nox/Kev typed FINAL, CSS15, and public-231 predictions
were revalidated and sealed at `2026-09-27T14:07:25Z`, before this run read
any answer key; seal SHA-256
`20d1e222c00878b3d46c1b3966327ffb5df9c19a2117c1c9e31add1046695156`.
The [prescore receipt](qwen35-4b-official-best466-v3-prescore-seal-2026-09-27.md)
contains each prompt and prediction hash. A separate scoring-start receipt,
SHA-256 `fd897aafdd432250585eb2dc8d884febcf5fd0f52b99597ef9c2d9154208a397`,
was created at `2026-09-27T14:10:17Z` after the seal. Earlier project work
had accessed these panel labels, so this is **post-key same-panel evidence**,
not a never-unsealed blind test. No candidate checkpoint or score rule was
changed after the prospective lock.

The common frozen score is `100 × sqrt(T × H)`: `T` is typed FINAL's mean
accuracy across four independent families and `H` is the median of 15
human-transfer task macro-F1 values. Typed FINAL has 1,600 items and 2,000
answer slots; CSS15 has 6,547 items. Invalid, missing and over-budget answers
fail at the full denominator. The public JevBench subset has 231 separate
items and is **not** part of the v3 score or an official closed-set rank.

| Same-panel metric | Official-source BEST466 | Own Nox 1.0 | Kev 4B |
| --- | ---: | ---: | ---: |
| Typed family macro accuracy `T` | `.570625` | `.614375` | `.719688` |
| CSS15 task median macro-F1 `H` | `.496323` | `.519046` | `.487400` |
| **JevArena v3** | **53.218** | **56.470** | **59.226** |
| Typed answer accuracy | `.648500` | `.691500` | `.775000` |
| CSS15 micro accuracy | `.551245` | `.535665` | `.533832` |
| Public-231 correct | **171/231** | 173/231 | 175/231 |
| CSS15 invalid or over-budget | **18/6,547** | 4/6,547 | 13/6,547 |

Candidate minus Nox v3 is **−3.25236 points**, with a 95% paired bootstrap
interval **[−9.19593, +2.52043]**. The typed `T` difference is `−.04375`,
interval **[−.078125, −.010000]**; CSS `H` difference is `−.022723`,
interval **[−.124222, +.080899]**. Candidate minus Kev v3 is `−6.00843`,
interval **[−11.52681, −.89454]**. These are 5,000 fixed paired draws with
seed `20260927`: typed items are resampled by their independent source group,
and CSS by task/sample structure. The result fails the +3.0 gate irrespective
of the uncertainty interval.

## Where the development gain failed to transfer

| Typed FINAL view | Candidate | Own Nox | Difference |
| --- | ---: | ---: | ---: |
| Choice accuracy, 800 slots | `.69375` | `.69000` | `+.00375` |
| Noul accuracy, 800 slots | `.72625` | `.81625` | **`−.09000`** |
| Score accuracy, 400 slots | `.40250` | `.44500` | **`−.04250`** |
| Constraint competition | `.44750` | `.38000` | `+.06750` |
| Evidence join | `.96000` | `1.00000` | `−.04000` |
| Exception stack | `.47250` | `.63250` | **`−.16000`** |
| Resource ledger / Score | `.40250` | `.44500` | `−.04250` |

All 2,000 typed answer slots were valid. Yet candidate typed Brier was
`.271960` versus Nox `.204949`, and ECE10 `.173108` versus `.091912`.
On Score specifically, candidate Brier was `.576782` versus `.356974`,
ECE10 `.563362` versus `.170040`, and expected ordinal MAE `1.126545`
versus `.825223`. Thus the CAL-selected Score temperature transported badly;
this observation is diagnostic, not permission for post-key recalibration.

| CSS15 task macro-F1 | Candidate | Own Nox | Difference | Invalid candidate / Nox |
| --- | ---: | ---: | ---: | ---: |
| conv_go_awry | `.4913` | `.4558` | `+.0355` | 0 / 0 |
| emotion | `.5885` | `.4971` | `+.0914` | 0 / 0 |
| flute | `.8911` | `.5807` | `+.3104` | 0 / 0 |
| ibc | `.4113` | `.5427` | **`−.1314`** | 0 / 0 |
| indian_english_dialect | `.3944` | `.3080` | `+.0865` | 0 / 0 |
| media_ideology | `.3790` | `.4428` | `−.0638` | 0 / 0 |
| mrf | `.7217` | `.7608` | `−.0391` | 0 / 0 |
| persuasion | `.4963` | `.5306` | `−.0343` | 0 / 0 |
| raop | `.5181` | `.5756` | `−.0574` | 0 / 0 |
| reddit_humor | `.5918` | `.4624` | `+.1294` | 0 / 0 |
| talklife | `.2800` | `.2965` | `−.0165` | 0 / 0 |
| tempowic | `.5549` | `.6213` | `−.0664` | 0 / 0 |
| tropes | `.0731` | `.0670` | `+.0061` | **18 / 4** |
| wiki_corpus | `.6108` | `.5190` | `+.0917` | 0 / 0 |
| wiki_politeness | `.3744` | `.5696` | **`−.1952`** | 0 / 0 |

All 18 candidate over-budget CSS items are in `tropes`; they remain failures.
The candidate's higher CSS micro accuracy does not offset the lower
task-median macro-F1. Public-231 split: easy **48/48**, standard **68/72**,
hard **55/111**; Nox was 48/48, 66/72 and 59/111. The public gap is mainly
on hard items. Neither public result establishes a closed JevBench ranking.

## Hashes, resources and next experiment

Fixed scorer SHA-256 values were typed
`d02a3b2bbaa08ec45928fc354532b3c5aef80e0a5d8e9ed6348ad6d30e2bcc`,
CSS `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`,
public `aec840b6497beeae8b63e9270670c22506265c086a84e889bc82f694146518cf`,
and paired comparison
`bdb966ede0018ebdcd486c77f813ba722bc8d5f78ffab0dbf673631249d2513e`.
Candidate report SHA-256 values were typed
`f55539d9b3c3e1a4a476a3d02db9d904bd503d4df3de5af5964187608b3c4ff8`,
CSS `173f1ffba1ba77dfe22f9c4e6c02f3a8c6e25b21c353eed44f4e7a7691902640`,
and public `c8ae9f012a5b18abf1a67e6ea8b7ce3aac34d428ec431f389b1598f06e5cf1da`.
Candidate-versus-Nox and candidate-versus-Kev paired report hashes were
`b4fe4273e304f6d6b6e146e83a5098679a66ac048f0058cdab80ea7196f48b8a`
and `a73a2f5cd99fde75161a78acc7725211bd3b92387160d61806b74f9d40da01b9`.
These private raw reports and predictions remain retained outside the public
model package.

Candidate training used **1.55405 GPU-hours**; development reload/readout
`.08908`, CAL/repeat `.03035`, and formal typed/CSS/public readout `.216845`,
for **1.890325 candidate GPU-hours**. The separate Nox development control
used `.04026 GPU-hours`. CPU scoring is excluded. All candidate GPU jobs
exited zero and no readout was repeated after scoring.

**Prospective ablation to review, not yet authorized:** keep the same official
Qwen source, LoRA/head architecture, 7,455-row order, 466 updates and encoded
token budget. Change only the training loss weight on the 2,240 already
audited source-disjoint human-labeled Choice rows from `1.0` to `1.5`,
normalizing each accumulation window by summed weights. This tests whether
greater genuine-label gradient mass improves *transfer* without adding data
or tailoring to typed FINAL/CSS15. Compare with this completed fixed control
on disjoint SELECT and one fixed typed DEV/CSS pilot screen, report all typed
types and every pilot task, and require no Score collapse. Before any step,
freeze the exact row IDs, code/data hashes, zero-step parity, token trace,
selection and stop thresholds. Since project formal labels have been opened,
any later release claim also needs an independently untouched validation
source rather than reusing v3 as a fresh blind selector. No GPU is reserved
or training launched by this proposal.
