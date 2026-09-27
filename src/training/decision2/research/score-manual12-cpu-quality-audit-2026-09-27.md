# Hand-authored Score 12-group probe: CPU quality audit

**Disposition: HOLD.** This is a data-quality diagnostic, not admitted TRAIN,
SELECT or CAL data. It produced no model checkpoint, GPU run or benchmark score.
The conditional 0.8B and 2B training arms remain blocked on a larger,
independently reviewed and length-qualified dataset.

## Sealed candidate

Twelve independently composed situations yielded 36 answer-free review items:
two situations per mechanism across exception precedence, state transition,
multi-source evidence, numeric boundary, temporal precedence and insufficient
evidence. Each situation has three source variants. There are six short and six
nominally long situations, with 18 English and 18 Chinese items. The author
constructed structured oracle facts before rendering the source text. Removing
the changed source span leaves the same prompt within a three-item group; this
is a **structural ablation check**, not evidence that a human must use the span.
The private authoring source is original text, with no copied upstream passage.

The answer-free packet SHA-256 is
`5a5cfc3663c42761aa82dcc32774f2e843249ccebac3c0058b081d817250382c`;
its answer-free seal manifest SHA-256 is
`51d88e81a3887fa4043cf1890994ee48104c42fe56bff600a11af4f84db54b34`.
The private authoring-source SHA-256 is
`53644f1a58b975e91fa5178b52a8c637aacb4cb52d5cb7e74112ab5daafb48f9`.
The reviewer must receive only the answer-free packet and manifest, never the
key, grouping, author audit or this report before sealing an independent
answer/evidence/ambiguity assessment.

## Frozen overlap and native length

The CPU scan verified the prior frozen 13-source reference list
(`9964d99aedac3085133c1d59ca4bd9cab5a8cad58f475458333e641bdbf374f7`)
and 28-role protected gold-free prompt inventory
(`99f271a8a681cccc66c0e68e77228691cb6f9628ed2cf93ce4c414c93baa842c`),
then verified each listed file against its frozen digest before reading.
Across 41 roles and 31,435 reference rows, the 36 candidate prompts had no
matching IDs, no normalized exact contexts and no five-gram near/suspicious
matches at Jaccard thresholds 0.60/0.35. The largest observed five-gram
Jaccard was 0.00264. Roles include the rights-clean v2 TRAIN/SELECT/CAL,
earlier Score candidate packets, typed DEV/FINAL, human pilot/final and
public supplementary prompts. Reference roles overlap, so 31,435 is a
comparison count, not a count of independent questions. Protected gold and
predictions were not read. The private scan receipt SHA-256 is
`656963fa3c7960a03232da89b21e6a5ce3ea1815b05e48ac248dd24158f62a83`;
the scanner SHA-256 is
`c92eb4bb50b3b3151f0e19cf57f0a8f707618eb7d4280bdf8d45ed479f8612c9`.
Textual overlap checks cannot rule out paraphrase or shared semantics.

The exact tokenizer used by the candidate 0.8B and 2B Qwen-family sources has
SHA-256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.
Without a chat template, the 36 question inputs span **184–683 native tokens**
(median 459); **0 exceed 700** and **0 exceed 1,500**. Thus the six intended
long situations do not provide a meaningful long-context test. The earlier
v8.4 packet failed independent construct review despite correct labels, and
v8.5 missed its preregistered long-input quota. This new packet also fails
the long-evidence objective; no prose padding, seed reroll or post hoc
threshold change was used.

## Next action

Independent blind review may diagnose answer ambiguity, document realism,
English/Chinese naturalness, decisive evidence and single-field shortcuts on
the sealed packet. Even perfect label agreement would not admit these rows to
the planned 128-group Score training expansion. A new prospective long-source
pilot must use genuinely different document genres, require a distant source
for the answer and pass native-length, shortcut, source-ablation, overlap and
blind-review gates before any 0.8B or 2B GPU arm starts. The current data and
both model sizes remain **HOLD**.
