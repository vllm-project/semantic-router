# JevArena authored v10: prospective DEV editorial pilot

Status: **design frozen before construction**. This is a new, private DEV
candidate. V9 R5's independent reviewer solved the twelve originals but found
one source-deletion leak, two weak narrative hints, four Boolean true answers,
and three of four Choice answers in the third displayed position. V9 remains
immutable and blocked for release.

## Scope and semantic design

- Target twelve newly written case files, four each of Choice, Noul and Score.
  Use twelve distinct policy operations and domains; do not reuse a v7–v9
  scenario, policy text, evidence paragraph, fact key set, or document layout.
  A shortfall is reported rather than filled with paraphrases.
- The new mechanisms include joining independent measurements and authority,
  explicit veto or exception precedence, constrained candidate comparison,
  set reconciliation, and interval intersection. At least three cases require
  four causally necessary documents and 650–1,200 useful words. Mere length,
  an archive marker, or a changed label is not a new mechanism.
- The four Boolean answers must be two true and two false. The four Score
  answers must occupy at least three of the five grades. Choice must include
  at least three different semantic outcomes, including a hold, and the four
  answers must occupy four distinct displayed positions. Display order comes
  from a single newly generated private salt and an answer-independent hash.
  Facts are authored to meet the preregistered position targets after that
  one salt is fixed; the salt is never searched or regenerated for a pass.
- Every claim in an introduction must remain true after any signed source is
  removed. Introductions describe jurisdiction and the decision procedure,
  without signed counts, pass/fail, approval, completed actions, the number or
  presence of case documents, candidate elimination, or predicted outcome.
  Document prose must not state another document's signed fact or the answer.

## Frozen proof and blind review protocol

1. The local source defines the renderer, validator, two structurally
   independent policy evaluators, and a parser of model-visible structured
   evidence. Private case specifications and the one private salt remain on
   the authorized experiment host. The local checkout owns all code.
2. For every essential source deletion, remove the entire source block and
   keep the introduction and remaining blocks byte-identical. Enumerate a
   preregistered bounded set of domain-valid completions of the missing fact.
   At least two completions must produce different final outputs under the
   current rule. Merely comparing one alternative with the original is
   insufficient. A second evaluation path must agree on each completion.
3. Verify unique case scope, exactly one current rule, prompt/target/proof ID
   agreement, deterministic rebuild, all declared source deletions, and a
   no-result-assertion prose scan. A source not necessary under the deletion
   sensitivity definition cannot be claimed essential. A lead or surviving
   paragraph that lets a blind reader recover a deleted answer blocks the row
   even if mechanical proof passes.
4. Freeze a gold-free prompt packet and gold-free source-deletion packet,
   separate mode-0600 private targets, source specifications, and proof traces.
   Record SHA-256 commitments for code, salt, every artifact, and audit. No
   reviewer receives the targets or proof traces before sealing answers and
   deletion judgments. No frozen artifact is overwritten; revisions get new
   versioned directories and hashes.
5. The independent reviewer first seals original answers and editorial notes
   for all items, then separately seals each source-deletion judgment. Only
   afterward may a separate operator open the key for aggregate comparison.
   Review both logical answerability and practical shortcuts, naturalness,
   source necessity, document variety, answer/position balance, and absence
   of contradictions after deletion. Agent agreement is not a human signoff.
6. V10 is DEV-only regardless of its review result. No training, model
   selection, FINAL labels, release score, or Hugging Face publication uses
   this pilot. A passing review can inform a separately preregistered,
   independently authored and sealed release-sized panel.

If any mechanical, balance, or editorial gate fails, retain the failed packet
and write a prospective amendment before building a new revision. No
answer-conditioned post hoc wording change or salt resampling is permitted.
