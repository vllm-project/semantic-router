# Human-labeled evidence support as a three-level Score arm

**Status: source hypothesis only.** No new corpus has been downloaded, no
training row has been admitted, and no model has been evaluated under this
proposal. It follows the 0.6B, 0.8B and 9B development observations that
three-level Score often collapses to an extreme. The completed 4B Base
control is a counterexample: the failure is not universal across sizes or
initializations. The source and model effects require a matched ablation.

## Semantics worth testing

The native System One `Score` question accepts an ordered 2–10-level rubric
and returns probabilities plus the expected level index. A naturally
three-level rubric is **evidence refutes a claim / evidence is insufficient /
evidence supports a claim**. Mapping NLI contradiction, neutral and entailment
to those levels preserves the original three-way distinction. Unlike an
invented 0/1/2 rule template, the text and labels in the candidate sources
were produced and judged independently of this model program.

| Source | Proposed role | Grounded source facts | Admission issue |
| --- | --- | --- | --- |
| [Stanford SNLI 1.0](https://nlp.stanford.edu/projects/snli/) | English TRAIN candidates only | 570k human-written premise/hypothesis pairs, manually labeled entailment, contradiction or neutral. The source includes Flickr30k captions and is Attribution-ShareAlike. | Rights-clean v2 already includes 272 SNLI-derived Choice rows. Pin record IDs and deduplicate by original pair **and premise group** across all training and evaluation roles. Short caption style may create annotation artifacts; it cannot by itself establish long-document transfer. |
| [CLUE OCNLI](https://github.com/CLUEbenchmark/OCNLI) | Native Chinese TRAIN candidates only | About 50k Chinese training pairs, written originally in Chinese rather than translated, with the same three relations; publisher license is CC BY-NC 2.0 and some premises have separate source conditions. | Audit genre and premise IDs, filter any disallowed or uncertain source, preserve original script and label meaning, and check against every protected Chinese prompt. A private repository does not itself clear re-distribution. |
| [FEVER](https://fever.ai/) | Possible source-disjoint evidence diagnostic, not a default training source | Claim labels are supports, refutes and not enough information. | The claim label alone does not supply the evidence shown to the model; building a faithful native decision requires a fixed evidence-retrieval protocol, full source rights and an explicit treatment of missing evidence. A claim-only shortcut probe would not qualify. |

This arm would teach one specific evidence-support decision. It must not be
described as a general solution to arbitrary Score rubrics. Existing typed
Score has rule/state mechanisms; an independently held rule/state diagnostic
is needed in addition to any NLI development gain.

## Prospective CPU admission before a training lock

1. Pin publisher revisions and exact train files on an authorized SSH host.
   Keep source text and row-level labels private. Count rows and independent
   premise groups per relation, language, genre and native token-length band.
   Inspect a blinded sample for whether the proposed rubric matches the
   publisher label, especially neutral versus missing evidence.
2. Group all hypotheses sharing a premise and all translations/near
   duplicates before allocating TRAIN, SELECT and CAL. Compare exact and near
   matches against rights-clean v2, typed DEV/FINAL gold-free prompts, CSS
   pilot/15, public231 and previous Score packets. Quarantine overlaps and
   source-uncertain records with private reason codes.
3. Build separate English, Chinese and mixed 0/1/2 pilots. Test a
   hypothesis-only negative control, original premise/hypothesis order,
   rubric wording, option permutation and class/length shortcuts. If a cheap
   shortcut performs as well as full evidence, hold the arm.
4. Freeze a source-disjoint Score diagnostic and one matched data-substitution
   contrast before any optimizer work. Preserve the existing official or own
   weight start, native interface, total tokens, steps and checkpoint rule;
   reuse the completed parent control. Evaluate both the three-level
   diagnostic and original typed/CSS development views. Do not tune to keyed
   JevArena v3 or public JevBench.

The NLI-to-Score mapping, rights, actual label balance, overlap and model
gain are **not yet validated**. If it passes the CPU gate, the next action is
a signed, size-specific training preregistration; otherwise it remains a
documented negative source screen.
