# Kai2 0.6B development recovery: data preflight

This is a development-only continuation of the own-Kai2 clean-v2 export. It
does not alter the post-key JevArena v3 result or qualify a model for release.
The fixed arm and stop rules were registered in
`kai06-human-recovery1024-prereg-2026-09-27.md` before materialization.

## Materialization chronology and result

- The first CPU-only builder attempt failed before writing a dataset because
  both frozen source catalogues contain some of the same upstream component
  IDs. Before any training or development outcome, the builder was amended to
  exclude all clean replay components represented anywhere in the human
  catalogue. The quotas, seed, selected human buckets and training schedule
  did not change. The amended builder has focused tests and signed commit
  `e54f7a84c`.
- The resulting TRAIN has 1,024 unique native rows from 889 complete source
  groups: Choice 576, Noul 320, Score 128. Five human-labeled TweetEval
  buckets contribute 64 rows each; the other 704 rows are clean-v2 replay.
  TRAIN SHA-256 is
  `58b94ac987b2fe134a392310a8d9d63f3d0e996a6016884f6dafadf3ae5449da`.
  The materialization receipt SHA-256 is
  `f17b5784232df1c0ab30f1e0d686deb12210b7d1d4fd04552147834acf20678a`.
- The gold-free audit used the exact 700 SELECT, 700 CAL, 1,600 typed DEV,
  1,430 CSS pilot, 1,600 typed FINAL, 6,547 CSS FINAL and 231 public prompts.
  Each panel had zero shared row IDs, exact raw or normalized states, or
  approximate near matches under the existing frozen SimHash/length/sequence
  rule. SELECT component and CAL group intersections were also zero. Audit
  code commit is `cfab422c7`; private report SHA-256 is
  `79f4cea27f8f68d384fc066110fb16890db48abfdd5064b75569105d2dc02e94`.
  Approximate matching cannot rule out paraphrases, task-family transfer,
  earlier source exposure or base pretraining contamination.
- The source Kai native packer and tokenizer accepted all 1,024 labeled rows
  without truncation at the fixed 1,024-token budget. Maximum packed length
  was 1,023 tokens, p95 679 and median 152. Packer and tokenizer SHA-256 are
  `f1dab7f032f4aaab27118606397ca67d71b1f4a2ba49e65bebcfd83a64b27819`
  and `609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f`.
  Check code commit is `f85d963b8`; private receipt SHA-256 is
  `411234d317651529596e6807d33d942b5b69c2ee7bde934b83ddb988c0a892a3`.

No GPU time or scored evaluation labels were used for this preflight.

An initial zero-step package check later failed in the parity *checker* before
emitting a result: the checker incorrectly required a Noul-only `probability`
field on Choice and Score rows. Native prediction schemas were inspected by
type, and the checker was amended to compare the exact field set, relevant
categorical fields and every present numeric field. This failure is retained;
it did not start recovery training or access formal labels.

## Remaining gates

The next discriminating step is a zero-step native parity check against the
selected own-Kai2 export, followed by a one-GPU numerical smoke and the
single frozen 16-step arm. The development thresholds in the preregistration
must all pass without selecting an alternative mixture or checkpoint. An
untouched, source-disjoint human transfer check is required before another
post-key v3 comparison can support a release claim. The selected human source
terms allow noncommercial research; the prior rights attestation concerns
another run, so resulting weight redistribution still needs an exact-package
review. Until these gates pass, this arm is **HOLD for release**.
