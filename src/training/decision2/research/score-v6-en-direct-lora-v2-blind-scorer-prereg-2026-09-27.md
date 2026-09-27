# Score v6 English direct LoRA v2: prospective blind scoring addendum

**Status: method only, signed before the r2 answer key or any final r2 model
prediction is opened by this scorer.** This addendum specifies computation and
sealing details left implicit in the existing v1/v2 protocols. It does not
change the candidate, data, optimizer, selector, threshold, or permission to
run further experiments. A failure consumes this single paired selector use.

## Two-stage seal and unblind

The `seal` command accepts no answer key. It verifies the frozen 192-row
English native prompt SHA-256
`3ad89c37170de17ccda63f149aaf14a7d4112e75db01741a3ef1b4b2c161b726`,
the private arm-preparation manifest SHA-256
`b10eae10192096c2dcf52af60ad6805c5180abd00f7ff2a5af52a4f5dadc736c`,
the original adapter-plus-base fingerprint
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`,
and the two frozen arm TRAIN and parent SELECT/CAL hashes already named in the
v2 preregistration. For both arms, `COMPLETE.json`, `LATEST.json`, `BEST.json`
and the final checkpoint metadata must all say step 174 and the sole final
checkpoint. The 174 distinct optimizer steps must account for 474,124 admitted
raw native tokens per arm. The common trainer contract, source-code hashes,
preflight parity receipt, source fingerprint, run settings and checkpoint
fingerprints must match, except the preregistered arm identity and TRAIN hash.
The frozen precomputed padding delta must be within 5%; it is identified as a
preparation estimate, not a measured GPU padding counter.

The source, A and B gold-free native predictions must each have a matching
manifest and exactly one record per frozen prompt, in prompt order. Their
manifest/payload SHA-256, full checkpoint/source fingerprints, native adapter
code hashes, prompt SHA-256, 1,024-token limit, temperature 1, uncalibrated
output, and no truncation must agree with the seal. Invalid model answers may
remain in the files and count as errors; altered/missing prediction records or
unverifiable manifests block unblinding. Parent SELECT baseline and fixed-step
prediction files, provenance, training log and all needed model metadata are
included in the seal. The `score` command requires the seal's explicit SHA-256,
rechecks every sealed artifact and model fingerprint before opening the key,
and creates a one-use marker before the key read. No intermediate r2 score may
be used to choose a checkpoint or repeat a run.

The frozen private oracle file SHA-256 is
`8e9d1232c1d0db75d7fc1e583d07753e5db6716c85aca39040744a2ec834eda1`.
Only after sealing and revalidation may the scorer load it. It must contain
the complete 240-row oracle; only its 192 English rows are scored. The 48
Chinese rows are untouched by inference and are not reported as transfer.
Each English row must join exactly to the native prompt by opaque ID and
input identity, with 64 groups, 16 groups per operation and exactly one
0/1/2 target in each group. Key mismatch is a failed protocol, never a reason
to repair prediction files after unblinding.

## Scoring and uncertainty

For Score, accept only a native `decision` answer with type `score`, finite
probabilities for exactly the three ordered keys `0`, `1`, `2`, each in [0,1],
sum within 0.02 of one, and a finite expected score within 0.06 of the
probability-weighted mean. The predicted level is the **unique** probability
argmax; a tie within `1e-8`, wrong type, malformed map, absent answer or
invalid adapter status fails the row. Valid maps are divided by their original
sum for normalized Brier. The point-accuracy denominator always includes all
192 rows. Missing or invalid answers are incorrect and counted separately.
Report source/A/B total correct, 0/1/2 level confusion, four 48-row operation
slices, group all-three-correct rates, invalid counts and normalized Brier for
valid rows; do not silently assign a probability score to invalid rows.

The paired A-minus-B uncertainty unit is an independent three-row group.
For each of exactly 10,000 replicates, draw 16 groups **with replacement
within each of the four operations**, preserving four 48-row slices. A draw's
statistic is the total signed correct-row difference divided by 192. Use
Python's `random.Random` with integer seed obtained from SHA-256 of the UTF-8
literal `decision2-score-v6-en-r2-stratified-bootstrap-20260927-v1`; iterate
operations in sorted name order and groups in sorted opaque-ID order. Sort the
10,000 differences; take the zero-based order statistics at indices 249 and
9749 as the fixed 95% interval. The lower bound must be **strictly greater
than zero**. The bootstrap is descriptive for this frozen selector, not a
population or release-test confidence claim.

The unchanged advance gates are conjunctive: A has at least 12 more correct
of 192 than B, the paired interval lower bound is positive, at least two of
four operations have a strictly positive A-minus-B correct count, none loses
more than two of 48, and Choice277 and Noul271 parent English SELECT each
lose at most 0.02 accuracy against the same original-source baseline. Their
normalized Brier must worsen by at most 0.02 each, and each type's invalid
count must not increase. Parent SELECT scores come from the trainer's sealed
baseline and fixed-final records, with malformed/tied answers counted invalid
and assigned the worst normalized Brier of 1 for retention. The prerequisite
source, data, steps, tokens, no-truncation and padding gates above must all
pass. Report each Boolean and its numerator/denominator; a failed gate yields
`DO_NOT_ADVANCE`. A pass allows only the single independent typed DEV/CSS
pilot specified by the original protocol, never a release claim.
