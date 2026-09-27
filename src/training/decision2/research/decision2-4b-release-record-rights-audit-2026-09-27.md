# Decision 2.0 4B: package record and rights evidence

**2026-09-27, read-only audit. Release rights remain HOLD.** The original
[first-release gate result](decision2-4b-jevarena-v3-first-release-hold-2026-09-27.md)
and the separate post-key tradeoff decision are unchanged. This audit did not
alter weights, calibration, predictions, scores, freeze receipts or thresholds.

The private `decision2-release-package-record/2` **draft** has SHA-256
`70712f492358097c69242da656cd3001bc2854e9517f4552007dfef96ba2871b`.
Its accompanying private rights/provenance audit has SHA-256
`e714b381d19550bae1674af3fcbe1860ef73268aa4c8c07a268c12cb01a5cf48`.
Both are mode 0600 in a private mode 0700 evidence directory. The draft sets
`rights.status=pending_independent_review` and leaves `reviewed_by` empty;
the existing fail-closed publication record validator correctly returns
`Reviewed source rights and release scope are required`. Neither file is a
release signoff or a public model-repository artifact.

## Byte-bound facts

| Item | Observation |
| --- | --- |
| Candidate | `llm-semantic-router/DEV2.0-4B`, native revision `checkpoint-0232`; candidate lock SHA-256 `5e7c70f8963d36671fccc85b0abdd8c924eefe1ac0896fc397e069cce66f1535` and prospective plan SHA-256 `cd579fbe2aec27053f10e0185d5d8360472b0a5bef0504f9ca96aeee55b49b19`. |
| Native package | All 21 files listed by `SHA256SUMS` independently rehashed without a mismatch; the 22-file inventory includes the list itself. List SHA-256 `7e005ef609553d973c3a6232840436d1384657a19f5765e42752ebcc04906e39`. The three active safetensors shards contain exactly **4,205,751,296** parameters by the repository's header validator. |
| Calibration and training | CAL SHA-256 `6b6af1ba82c2fc5111c49a881f64154f1f27ac9b7e09e8859789ae3bb01f3b60`; rights-clean data manifest `61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`; run provenance `e7792c10924c89120502ed99d9d457bc1237b77a488e85d6136fc9733d71a8ab`; exact training `train.py` `092cfc70b3531d11e0f2a1acee24d7d3da97fe613ad1e577080067a3e9a0274e`. TRAIN/SELECT/CAL are 7,455/700/700; TRAIN language counts are EN 6,085 and ZH 1,370. |
| Selection lineage | Trainer `BEST.json` chose checkpoint-0224 and `COMPLETE.json` records a complete 232-step run. The later frozen native `NATIVE_BEST.json` chose checkpoint-0232 on the separate SELECT700 native readout. The draft names the latter as the actual selection policy; it does not misattribute checkpoint-0232 to trainer BEST. |
| Upstream model | [Eikos-4B](https://huggingface.co/caiovicentino1/Eikos-4B) revision `582ffb13f19a4da3f455e3db198584190bd7755b` is the pinned initialization. Its model metadata identifies MIT contributions and Qwen3.5-4B as its base; its README identifies the inherited Apache-2.0 Qwen license and FinEntity, TAT-QA and GSM8K data terms. `LICENSE`, `LICENSE-Qwen` and `NOTICE` in the candidate package are byte-identical to the pinned source snapshot. |

The frozen `publication.rights_gate.verify_rights` passed the exact source
manifest, split hashes and counts under `license_id=other`. Its source-rights
entries cover all TRAIN and SELECT/CAL rows. The manifest permits trained
weights and model-card disclosures, excludes raw rows and individual text
predictions, and requires source attribution, original notices, a separate
base-lineage audit and disclosure of CC BY-SA model-weight uncertainty. The
mechanical pass is not the independent release-rights decision.

For train/evaluation overlap, the builder manifest already records zero ID,
group, input-hash, raw/normalized state and approximate near-context matches
across its 21 split/DEV/CSS audits. The same frozen `context_overlap` routine
was then run on all six TRAIN/SELECT/CAL × typed FINAL/public JevBench prompt
pairs. Each pair again returned zero in those categories. References were
wrapped as audit-only state rows; no label was opened or changed. The
approximate SimHash/SequenceMatcher check is not proof of semantic
independence, and source or generator-family overlap remains disclosed.

## Remaining rights release decision

An independent reviewer must provide a real identity and evidence-bound
decision for the exact package and nine source-rights entries, including the
CLINC/BANKING replay components, SNLI and SQuAD 2.0 CC BY-SA derivation and
attribution wording, GoEmotions and FLUTE notices, and Eikos/Qwen upstream
lineage. That review must state the permitted noncommercial model-weight scope
and be bound to the private draft/audit hashes before `rights.status` can
become `passed`. A separate versioned train/evaluation-overlap release review
must bind the newly generated FINAL and public-panel audit receipts. No
placeholder reviewer identity or success flag has been inserted.

The general packager's `_inventory` currently rejects two literal loopback
addresses in the unchanged upstream `serve.py` as if they were private
network addresses. A separate narrow, versioned packaging exception can
address those two safe literals while retaining the package bytes; this
rights audit does not modify that validator or the source file.
