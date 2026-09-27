# Official Qwen3.5 4B: fresh three-level Score CPU candidate is HOLD

**Decision: HOLD; zero GPU-hours.** The completed official Qwen Base control
remains the relevant 4B baseline. Its matched official Posttrained arm lost
188/400 typed DEV Score answers while gaining 25/800 Choice answers. That
readout is development evidence, not a release result. The train corpus had
only 102 three-level Score rows and the original SELECT had zero three-level
Score rows. This CPU experiment tested whether a source-disjoint, three-level
candidate and SELECT3 gate could be constructed without spending another
optimizer run. It did **not** authorize training or touch formal labels.

## Immutable inputs and isolation

- Source revisions under consideration: official `Qwen/Qwen3.5-4B-Base`
  `1001bb4d826a52d1f399e183466143f4da7b741b` and the already completed
  Base control. The held Posttrained checkpoint is not a continuation source.
- Rights-clean v2 TRAIN: 7,455 rows, 4,194,465 unpadded tokens under that
  exact tokenizer; SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
  SELECT/CAL were protected from generation, not used for choosing examples.
- A private 32-byte seed has SHA-256
  `cc23e69c7e969f07f2e9ec783a7591311466d65761a3c2e7f186f6d10cda34e9`.
  The seed, generated text, answer key and blind per-item review are private.
  The CPU build used a pinned container image (digest prefix `f83b1d10f14d`),
  no network and no GPU device.
- The 32-role gold-free protected inventory SHA-256 is
  `c6d3f497b4385ff48817e9a6f98f63baf529b99b27a132a190164d5d729c18c2`.
  It includes TRAIN/SELECT/CAL, typed DEV, gold-free typed FINAL, CSS pilot and
  gold-free CSS15, public supplemental prompts, prior SELECT r1/r2 packets and
  the held v6 packet/full v6 prompt projection. Those earlier packets were
  used only for exclusion checks, never as this candidate's TRAIN or SELECT3.
  All 70 resulting train/select-to-protected comparisons found zero exact or
  near-context/full-prompt collisions. This mechanical screen does not prove
  semantic independence.

## Candidate construction and budget screen

The builder fixes structured facts and an ordinal oracle before rendering.
Four TRAIN source families generated 96 independent cases and complete
0/1/2 counterfactual triplets: 288 rows, 216 English and 72 Chinese. Two
separate SELECT3 source/rendering families generated 32 cases and 96 rows,
72 English and 24 Chinese. Every class occurs once per case. An independent
displayed-text parser oracle matched the construction oracle. The SELECT3
blind review packet omitted the key and used separate opaque review IDs.

The first budget design proposed replacing 288 synthetic Choice rows. It
failed: the new TRAIN had 78,789 tokens but the chosen Choice removal had
362,982 tokens, a **−284,193-token (−6.775%)** deviation from the control.
This failed receipt was kept; it was not repaired by selecting a favorable
model checkpoint. A second prospective CPU-only design proposed replacing
288 complete internally generated old Score singleton groups (279
`stage4_ordinal`, nine `stage4_dense_table`), preserving all Choice/Noul and
the targeted/replay Score rows. Its chosen ID roster SHA-256 is
`9bb10fda5e7971783b20699ce40f14d9a3080bdb3d7bd6a0540a3a3dfe412630`.
The exact 4B tokenizer counted 82,461 candidate tokens versus 100,743
removed, yielding 7,455 rows and **−18,282 tokens (−0.43586%)** against the
4,194,465-token control. SELECT3 has 27,172 tokens. All candidate rows fit
the 8,192-token input cap. This is a feasibility observation, **not** an
admitted training arm: the old Score family mix changed, padded exposure has
not been frozen, and no optimizer/selection/preregistration lock exists.

The final private CPU manifest SHA-256 is
`9d447448f36b20563f873d03e3498f6aa62c51bc402e170c1f03d118ea58672b`.
TRAIN candidate SHA-256 is
`48d9dfaf2b15e96a694379ae7751865d5d4c83b873699792833ae64d69bc2555`;
SELECT3 blind packet SHA-256 is
`19ebc76a6e3a00ff1bce55052c8a0ab76df5e937f8457477cec73e4144072b5a`.
These are private data fingerprints, not files for the public model repo.

## Independent blind review and decisive failure

An independent reviewer read **all 96** SELECT3 questions without the
generator or key. The reviewer found 96 unambiguous answers, 32 per class,
and complete 32/32 triplets. The private per-item blind receipt SHA-256 is
`1e8433e3df9adf36f89f75b16f4602d7929837aa124783b2ddefe1dbdf93821b`;
the aggregate summary SHA-256 is
`f63a59e7ff61885ac0885dd576be5e97cd1316cc111d46f8a32134c0cab9c91f`.
Correct answerability did **not** clear the quality gate. For **96/96** rows,
the policy thresholds can be ignored: current-versus-archive value directions
plus the fixed field polarities deterministically recover the ordinal label.
The builder's negative-control test reproduces that 96/96 bypass. The archive
has a fixed one-passed-check pattern and is unnecessary for the intended
current-rule decision. Thus this SELECT3 cannot measure the targeted
rule/evidence skill or be used to claim transfer. Further editorial findings:
all 72 English policies have a lowercase second sentence; 36 English
planetarium rows repeat units; 24 Chinese policies have mechanical spacing;
12 Chinese planetarium rows have awkward percent/staffing wording; 15
planetarium rows use implausibly high cloud limits. Shared case IDs across
triplets need explicit counterfactual framing. No row-position shortcut or
other semantic ambiguity was found by that reviewer, but this does not offset
the decisive policy bypass.

## Repair and next discriminating CPU experiment

Preserve this corpus only as a **failed shortcut probe**. Do not relabel,
lightly edit and quietly reuse its SELECT3 or train from the proposed merge.
Build a new source version with at least one genuinely independent ordinal
mechanism and natural EN/ZH writing. Make any archived evidence independent
of the answer or make version selection genuinely necessary, then require
policy-ablated and archive-delta-only negative controls to fall materially
below full-evidence performance on both TRAIN and a source-disjoint SELECT3.
Blind-review every new SELECT3 case and its counterfactual triplet. Separately
audit a human-labeled ordinal source and a source-disjoint probe so synthetic
rules do not stand in for cross-domain transfer. Verify rights, source-group
isolation, ordinal mapping and overlap before bringing raw text into any
training environment. Only after those gates pass should a fresh 4B training
lock fix initialization, exact data hashes, 4.194M-token exposure, padded
exposure, 466 updates, checkpoint selector and one-time Score/retention gate.
The completed Base control should not be retrained.

The local corpus contract suite passed three focused tests; Black and Ruff
passed. The negative-control test intentionally asserts the detected
96/96 shortcut so this failed version remains auditable. No model inference,
HF operation or GPU training occurred for this CPU task.
