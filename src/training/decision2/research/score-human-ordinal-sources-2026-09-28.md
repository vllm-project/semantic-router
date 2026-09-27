# Human ordinal Score source screen: STS-B, SICK, SemEval-2017 STS

**Decision: TRAIN HOLD; CPU source audit only.** No model weights, native
predictions, protected labels or GPU runs were used. This note does not
establish a new Decision 2.0 result. Checked 2026-09-28.

The current native `Score` primitive accepts an ordered rubric with 2–10
levels and reports a probability distribution and expected level. A human
average similarity or relatedness rating is continuous, so converting it to a
hard class changes the supervision. A faithful training projection must state
the rating anchors, prespecify the discretization or soft target, retain the
source rating, and report both expected-rating error and level accuracy. These
sentence-pair sources test only one rubric family; they cannot establish
general rule, state, evidence, or long-document Score transfer.

| Source | Verified identity and label | Rights and semantic fit | Source-isolation concern |
| --- | --- | --- | --- |
| [STS-B in TFDS](https://www.tensorflow.org/datasets/catalog/glue#gluestsb) / [SemEval-2017 paper](https://aclanthology.org/S17-2001/) | English sentence-pair semantic similarity 0–5, human rated; TFDS lists 5,749 train, 1,500 validation, 1,379 test. We did not pin/download the original archive, so no row hash or grade histogram is claimed. | [GLUE's FAQ](https://gluebenchmark.com/faq/) defers to each original dataset's terms. The original STS-B site did not load during this audit. A [redistributor's copy of the upstream license text](https://github.com/PhilipMay/stsb-multi-mt/blob/main/LICENSE) states score annotations CC BY-SA 4.0 and **different text rights by component**, including Microsoft agreement and news attribution. Treat this as a clue pending direct primary-file verification, not blanket permission for text redistribution. | STS-B was selected from earlier STS tasks, including video/image captions and NLI. SICK below also uses MSR video descriptions; our control already contains SNLI-derived rows. Source-family and record-level overlap are not cleared. The open GLUE validation/test cannot be called fresh independent transfer. |
| [SICK official Task 1](https://alt.qcri.org/semeval2014/task1/index.php?id=task-guidelines) / [publisher record](https://zenodo.org/records/2787612) | Downloaded only the official **TRAIN** ZIP from the [task organizer](https://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools), ZIP SHA-256 `097295b8ad526d80cefc7a56444424d7f39020729e589618b4d40bb654195e1a`; `SICK_train.txt` SHA-256 `266cf8047149bd1d68138dd30439f3122bff30069740ac6705bb6197d9f1b48e`. It has 4,500 English pairs, ten-human-rater mean relatedness 1–5, and a separate entailment label. | The official ZIP `readme.txt`, SHA-256 `4ff14a2b07603f2214c2803a6e012048c795be5ba4e5d1d33b87d7210f633bf7`, explicitly says **CC BY-NC-SA 3.0 Unported**. Our noncommercial research can consider it with attribution and source conditions; do not place raw text in a public artifact or assume private HF storage permits redistribution. It is short-caption compositional similarity, not general evidence strength. | Built from Flickr8K and SemEval-2012 MSR video descriptions and generated variants. It is not source-independent of all STS-B components. Its public train/test split is not a new blinded benchmark, and shared sentence/expansion families must stay together. Exact comparison to private SELECT/CAL and protected gold-free evaluation prompts remains undone. |
| [SemEval-2017 STS Task 1](https://alt.qcri.org/semeval2017/task1/) / [organizer data](https://alt.qcri.org/semeval2017/task1/index.php?id=data-and-tools) | Human 0–5 semantic similarity, with Arabic, English, Spanish and cross-language tracks. The organizer says Arabic training pairs derive from prior English STS material; exact source files, revision, distributions and IDs were not downloaded. | The organizer pages provide data but no single verified blanket data license. The ACL paper's CC BY 4.0 license covers the **paper**, not necessarily every underlying text item. Rights need file-level verification. Translated/adapted pairs are not independent multilingual situations. | It shares earlier English STS provenance with STS-B, and possibly training-control SNLI/caption material. The proposed cross-language subset cannot yet serve as a source-disjoint validation of STS-B or SICK. |

## Official SICK TRAIN diagnostic (no model run)

The official source has 4,500 rows but only 4,470 distinct ordered text pairs.
Normalized shared-sentence graphing yields 939 connected components; the ten
largest have 259, 73, 64, 39, 25, 20, 19, 19, 17 and 16 rows. Therefore a
random pair split would leak altered/reused sentences. Each side's median
length is nine whitespace words, 95th percentile 17 words (maxima 28 and 32).
Nearest-integer rating buckets 1/2/3/4/5 have 304/334/1,199/1,784/879
rows. A trivial casefolded unigram-Jaccard signal has Pearson `r=0.5818`
with the continuous human rating on this same TRAIN file; that is **not** a
trained/tested model baseline, but it flags an accessible surface shortcut.
The official task ranks continuous ratings by Pearson correlation, not
five-way exact accuracy. Its organizer states ten human ratings per pair.

## Existing panel relationship and next gate

The local rights-clean v2 source inventory names internally generated items,
GoEmotions, SNLI, SQuAD, CosmosQA, FLUTE, CLINC and BANKING, but not STS-B or
SICK. This is only a **source-name screen**. CSS's 15 evaluation tasks contain
neither by task name and are Choice-only. The control's SNLI/caption ancestry,
frozen typed prompts, CSS input text and public supplements still require
record- and group-level exact/near checks before any import. A model's
pretraining exposure to these old public corpora cannot be ruled out.

SICK is the clearest licensable candidate for a bounded **future** native
Score treatment, but it has not passed admission. Before optimizer work:

1. Pin complete source files and provenance. Quarantine duplicate pairs,
   shared-sentence components and source-family near matches against all
   TRAIN/SELECT/CAL and gold-free evaluation inputs. Keep private rejection
   reasons; never inspect FINAL labels for source selection.
2. Construct a three-way contrast on official or own initializer at one size:
   reuse the archived unchanged control; substitute a fixed count of complete
   SICK TRAIN components for current Score TRAIN rows, and add a **token- and
   step-matched** non-Score/replay control. Preserve Choice/Noul rows, native
   adapter, model revision, optimizer and candidate selection rule. Run only
   after native tokenization and source isolation prove the budget match;
   do not pad misleading text merely to match tokens.
3. Freeze the five-level rubric and mean-rating-to-target mapping before
   optimization. Compare hard nearest-level CE to a soft adjacent-level target
   only in an explicitly budgeted ablation. Use a **separate source** for
   human ordinal development; SICK train/test or STS-B split alone would only
   measure same-family generalization. Include rule/evidence Score DEV as the
   required transfer check, plus Choice/Noul and CSS regression checks.
4. Benchmark an evidence-free lexical-overlap control and a swapped-sentence
   check on held connected components. If it approaches the full-input result
   or the source fails rights/overlap/token gates, keep the arm on HOLD.

No SICK/STS-B/SemEval row is admitted yet. The next most discriminating CPU
task is source-group and token-budget isolation, not another GPU experiment.
