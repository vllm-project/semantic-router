# QuALITY long-document Choice source: CPU admission screen

**Decision: source candidate only; TRAIN, model-quality and release HOLD.** This
screen did not run inference, train a model, read JevArena labels or select a
checkpoint. It is an independent long-document Choice hypothesis after the
0.6B official-base model regressed on Choice. It does not address its Score
regression, so a future Choice-only gain could still lose the composite.

The [official QuALITY repository](https://github.com/nyu-mll/quality) and
[paper](https://aclanthology.org/2022.naacl-main.391/) document writer-authored,
multi-annotator validated four-option questions over roughly 5,000-token
articles. The pinned repository revision was
`f84977c40dbfef70c9cab48037b7becfc8e45f73`. We used its corrected
`v1.0.1.htmlstripped` files, not earlier HTML-stripping variants. The local
aggregate-only auditor is `training/data/audit_quality_source.py`, SHA-256
`6cc156cf2d41c78162f136cca6a26a4604b462d46746f0bd63fc13b32027e8bf`;
the remote exact mirror matched. The aggregate receipt SHA-256 is
`19484d676dce75b667c25a83f2dc9d36f62b0f70cd30042ec9672f4fef8f46fe`.
No article, question, label or record ID is in this note.

| Official split | Independent articles | Four-option questions | Median article words | Qwen3-0.6B raw content token proxy median / p90 | Raw proxy ≤8,192 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 150 | 2,523 | 4,752 | 6,607 / 7,650 | 2,443 |
| DEV | 115 | 2,086 | 4,728 | 6,557 / 7,608 | 2,050 |
| TEST | 116 | 2,128 | 4,864 | 6,771 / 7,636 | 2,118 |

The pinned Qwen3-0.6B tokenizer files were byte checked. The proxy tokenizes
article, question and four options but **omits the System One framing**; its
fit counts are therefore neither exact native admission nor a truncation
policy. The official test question labels are withheld; the auditor does not
open any. Across official splits, normalized article IDs, article text and
article–question–options payloads had zero exact matches. That does not
certify independence from our protected panels or pretraining exposure.

TRAIN answers by option position were 624/614/649/636, and the official
TRAIN difficulty flags were 1,272 ordinary / 1,251 hard. Source article
terms differ: 2,000 TRAIN questions appear with Project Gutenberg text,
355 with an ANC/OANC license pointer, and 168 with article-level CC BY 4.0
statements. These are **review buckets, not redistribution clearance**. The
repository does not provide a single uniform article-rights statement; the
question annotations' own reuse terms also need explicit review. Restrict any
later private candidate to sources whose exact article and annotation rights
are established. Do not infer that the model's Apache license grants rights
to republish training text.

## Next decision before GPU

1. Establish article and annotation rights per source. Exclude unresolved
   rights buckets. Keep complete `article_id` groups together and inspect
   duplicate story editions and near copies, not only exact hashes.
2. Build the complete gold-free protected prompt inventory for TRAIN, SELECT,
   CAL, typed DEV/FINAL, CSS pilot/15 and public231. Quarantine overlap by
   entire article group. The current audit did not locate or scan that
   inventory, so no training row is admitted.
3. Encode full prospective System One Choice rows with the pinned candidate
   tokenizer and reject overlength rows without cropping. Audit whether the
   answer can be predicted from options/question alone and whether deleting
   the claimed decisive article span changes the answer. Respect the source's
   multi-annotator ambiguity flags.
4. Only then freeze a source-disjoint, article-group-selected substitution
   against the existing official-base control at matched native token and
   optimizer budgets. Report Choice and Score separately, CAL probability
   quality and independent transfer. Do not use official DEV/TEST or opened
   JevArena results to tune the mixture.

This is a viable **research candidate** for long-context Choice supervision,
not evidence that any Decision 2.0 model improved. It leaves the 0.6B first
package and all six release/HOLD states unchanged.
