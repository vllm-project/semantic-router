# Two 0.6B decision checkpoints on the frozen development panels

**Research diagnostic; HOLD for Decision 2.0 backbone selection.** This is a
same-input native rerun of two independently published checkpoints, following
[`small-decision-models-2026-09-27.md`](small-decision-models-2026-09-27.md).
No training, model upload, protected FINAL, or CSS 15-task evaluation occurred.
The public JevBench 231-item subset is a diagnostic, not the sealed official
ranking. The authors' own internal scores are not ours.

## Identity, loader, and adapter gate

| Arm | Pinned checkpoint | Measured tensors | Executable code |
| --- | --- | ---: | --- |
| A | [`thefloydd/qwen3-0.6b-rlcd`](https://huggingface.co/thefloydd/qwen3-0.6b-rlcd) at `7c28eee87fc7f7a44dacabca5e89ed9281c995c2` | 596,054,018 parameters; model safetensors `d14c8e55d74af9a37301f4e90601d637791f9ecd3973b6d514d1feeb0f74cb74` | Pinned `configuration_bouncy.py` `95bc23268a60f263f1fdbc12aefca5673e424329972d9c7c7784eb81bec061cf`; `modeling_bouncy.py` `75ef00bc7fa3ef1099815ddcf05f28138f1b0974dc5b7c2f715c5c23213d3e62` |
| B | [`anthonym21/qwen3-0.6b-rlcd-decision`](https://huggingface.co/anthonym21/qwen3-0.6b-rlcd-decision) at `b327ec5efb5fdbf8bfafa3b369720ac5f6434b05` | 596,076,544 parameters: 596,049,920 body plus 26,624 decision-head; body `ad0b65098a40026a9c2b763125c45ec312fa4e11205c07b5eb32ad10d392e47e`, head `da1328e06c64789d334350975cb3350d8d2ef7c779f133cdd5b400feb835f233` | [`eve-rlcd`](https://github.com/anthony-maio/eve-rlcd) commit `adb9a0457ef53d2dc0e106215315670f1f6da871`; exported `decision.json` `27602b53d0d389fac9129a9ed95b1efbb98d6f4af3a202d78d0d1c6004995489` |

HF CLI downloaded both exact revisions on an authorized experiment host. Before
execution, the custom A files and B loader/import path were read. A imports
only local Python/torch/Transformers code and exposes the Jev-shaped
`score(state, questions)` method. B uses the stock Qwen3 body, a local
safetensors head, and a Git-pinned loader that checks body/head hashes and
letter-token IDs. The audited B path has no top-level subprocess or network
call; conditional Hub helpers for other backends are not used by the local
decision export. Inference containers had networking disabled. A tokenizer
SHA-256 was `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506`;
B's was `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
The runtime image digest was
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`
(PyTorch `2.12.0+git6bbd260`, Transformers `5.17.0`).

A receives the original state and typed questions directly. B receives the
same state rendered as indented JSON when it is structured. Choice options
carry the original key and description; Score options carry ordered index
and level description; Noul's true/false criteria are appended explicitly to
its yes/no question. Its probabilities are mapped back by option position,
and its normalized Score mean is multiplied by `K-1`. These bridge choices
preserve the requested labels and criteria, but their text format can affect
the model. B's native defaults left-truncate state above 1,536 tokens or
question suffix above 448; preflight marks such rows invalid **before**
inference. A's packer can also truncate, so its preflight rejects that path
and its 8,192-token dense-mask cap. Native invalids and overflows have empty
answer maps and count as wrong. A's tokenizer emitted a generic Mistral-regex
warning; the pinned Qwen tokenizer bytes were used unchanged.

## Frozen panels and chronology

| Panel | Gold-free prompt SHA-256 | Private development target SHA-256 |
| --- | --- | --- |
| Typed DEV1600 | `a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a` | `c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc` |
| CSS three-task pilot1430 | `598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda` | `9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391` |
| JevBench public subset231 | `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd` | `abc17b971d13807a15b3cdb43062f4cd876aad9d7314e72365724904e88b937f` |

The 100-row smoke packet (`a8518ae10cec3fb4ba531413bac485b66711f6dbf5143a9863252ff9f330964f`)
froze 10 DEV groups (40 rows), 10 rows from each CSS pilot task, and 10
from each public tier. It selected strata from development metadata and
emitted only `id/state/questions`. Preflight finished at 22:06 UTC on
September 26; A/B smoke predictions sealed at 22:07/22:08, with 100/96
native-valid answers (B had four state truncations). Exact full prediction
files then sealed by 22:12 UTC, **before any development gold scoring at
22:13 UTC**. Each prediction was bound to the original input SHA-256 and
preflight/model manifest. The reverse-order Choice packet was frozen from
smoke prompts before scoring, then run separately; it did not use feedback to
select rows.

| Full prediction SHA-256 | A | B |
| --- | --- | --- |
| DEV1600 | `508ffafd03bab411a7aeee8f2c9cd4952d579a221d0f378a38ecdb8bfff4ecf2` | `5b94a693b1534b28818a1c3cc947bded94056d58b61f168586cdb970344a124d` |
| CSS pilot1430 | `efc67623eccfebc79b20379f5aa02581305520d0b18d4a816dac5c7117c490d2` | `e4832f302f7218f33255ce1d9e487f0c32727b4836eedcfafa8bba2d2ef4fc3a` |
| Public231 | `b60c6bf13ea2ef68201c387f5ae41274c83969e2b385d2490fe2a19bab09ff10` | `9bf039290b89f787f0d1475b61e064c549f444e65f0cecb47090557aae0f5e9b` |

The private aggregate score SHA-256 is
`0a45ac282e51d52cddbe783ff8c46b99e80018bd830be74d10c7587f74ef5b2b`;
paired Kai/Laya and option-order aggregate hashes are
`61233c3b3795b87ae59214b7ccb1ceeafd058eb1d2e731a445adfa6e06ce3c82`
and `08ac2e1b54e423596a40b1208a16e77934c514745f00100f0975f616213edb25`.
Raw per-item predictions, targets, and scores remain on the authorized host.

## Same-panel outcomes

| Development metric | A | B |
| --- | ---: | ---: |
| Typed DEV correct / valid / all | 593 / 1,600 / 1,600 | 551 / 1,600 / 1,600 |
| DEV Choice; Noul; Score correct | 250/800; 204/400; 139/400 | 258/800; 208/400; 85/400 |
| DEV 10-bin pmax ECE; Brier | .374; .479 | .202; .358 |
| CSS pilot correct / valid / all | 534 / 1,430 / 1,430 | 541 / 1,416 / 1,430 |
| CSS median task macro-F1; median 15-bin pmax ECE | .2880; .1091 | .3034; .0495 |
| Public subset correct / valid / all | 127 / 231 / 231 | 133 / 194 / 231 |
| Public easy; standard; hard correct | 48/48; 45/72; 34/111 | 47/48; 55/72; 31/111 |
| Public 15-bin pmax ECE, valid only | .1555 | .0789 |

CSS task macro-F1, A/B: discourse `.288/.282`, implicit hate
`.211/.303`, SemEval stance `.485/.550`. B's 14 CSS and 37 public state
truncations were scored as failures; there were no DEV truncations or
question-suffix overflows. All 37 public invalids were in the hard tier.
The pmax calibration metrics use declared-option probabilities, not A's
separate in-list confidence head or B's self-reported in-distribution ECE;
bin counts differ between DEV and CSS/public. DEV is a deterministic
synthetic panel and its ECE is not an estimate of deployment calibration.

With B minus A, a 2,000-replicate DEV bootstrap resampling the 400 groups
within fixed families gave **−2.625 percentage points** accuracy, 95% CI
`[−4.56, −0.75]`. CSS pilot item bootstrap within fixed tasks gave `+0.49`
points, CI `[−2.31, +3.15]`; its median task macro-F1 difference was
`+.0154`, CI `[−.0230, +.0682]`. Public item bootstrap within fixed tiers
gave `+2.60` points, CI `[−3.46, +8.66]`. These intervals describe the
fixed development/public panels, not new tasks or training overlap.

The DEV counterfactual relation held for only 111/400 A pairs and 42/400 B
pairs. In a separate 32-row Choice order reversal, A kept the same choice
on 28/32; B on 21/32, with one B input invalid by the same state cap.
Mean maximum per-label probability drift on valid pairs was `.089` for A
and `.144` for B. This is a material option-order weakness.

## Earlier 1.0 controls, kept separate by size and adapter

Exact same-row input hashes were verified for pinned [Decision-1.0-Kai-0.6B](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B)
revision `7185f514f54b8f93c55998b1e8f9c5cc67f0d029` and
[Laya typed decisions 0.4B](https://huggingface.co/convaiinnovations/laya-typed-decisions)
revision `1a793eb568e6718f15941d08f85432581df534e3` on DEV and CSS.
Their prior native receipts were rescored using this run's exact scorers;
these are historical checkpoint/adapter results, not new model inference.

| Control | DEV correct | CSS pilot correct / valid | CSS median macro-F1 | Paired DEV accuracy: A minus control; B minus control |
| --- | ---: | ---: | ---: | --- |
| Kai 1.0, 0.6B | 425/1,600 | 418/1,408 | .1748 | `+10.50` points `[+8.50,+12.44]`; `+7.88` `[+5.94,+9.88]` |
| Laya, 0.4B | 638/1,600 | 525/1,407 | .2949 | `−2.81` points `[−5.75,+.25]`; `−5.44` `[−8.19,−2.50]` |

Against Kai, CSS pilot accuracy differences were A `+8.11` points CI
`[+5.24,+10.91]` and B `+8.60` CI `[+5.80,+11.47]`. Against Laya,
they were A `+.63` CI `[−2.45,+3.50]` and B `+1.12` CI
`[−1.89,+4.06]`. The 2,000-replicate paired resampling uses DEV groups
and CSS items within task. No exact Kai/Laya public231 receipts were in
this comparison, so no public pairwise claim is made.

The separately documented [GLiNER2.5-Decide native baseline](../inference/GLINER25_BASELINE.md)
has **486,444,053 measured parameters**, 652/1,600 DEV, 561/1,430 CSS
pilot (median macro-F1 `.31035`, 1,363 valid), and 116/231 public subset
(175 valid). It is a descriptive same-panel reference; its per-row receipts
were unavailable for this paired audit. A/B are below it on DEV and CSS
pilot and above it on the public subset, where native context limits differ.
No different Kai continuation, GLiNER continuation, or author self-report
was folded into these exact-checkpoint comparisons.

The combined evidence does not support a 0.6B backbone promotion or a
release claim. B's narrower native context and both models' option-order
and counterfactual failures need resolution before they could serve as a
training start or a JevArena candidate.
