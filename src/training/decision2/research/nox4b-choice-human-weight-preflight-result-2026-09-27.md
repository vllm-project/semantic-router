# Own-Nox 4B Choice-weight screen: zero-step STOP

This is the outcome of the [prospective protocol](nox4b-choice-human-weight-prereg-2026-09-27.md).
**No fixed step-128 arm, typed DEV, CSS pilot, JevArena FINAL, CSS15 or
publication followed.** The completed clean-v2 control remains untouched.

The treatment worktree is signed commit `61db4bbdd22b332a530db669578376d4c1bd67ba`;
a subsequent signed preflight note is `3fd88e6a4a0ec25cbe58f88bde4f8742d87eef23`.
The exact local trainer SHA-256 was
`47902f9cc71ab8fc65cda9b43c029dab3c3217cf2503b615ea064a912bdc2476`.
The read-only remote mirror had the same hash. CPU loss/plan tests in the
pinned container image passed 4/4. The GPU allocation check found one idle
BF16 card, and the source/data load matched the archived control on model
file hashes, 7,455 TRAIN rows, 4,194,465 unpadded tokens, maximum 6,596
tokens, 700 SELECT and CAL rows, and 35,097,088 trainable parameters.

## Failure and execution-order correction

The preregistered **all-700 zero-step parity gate failed**. IDs, native
prompt/token hashes, task types and candidate domains matched the archived
control, but four categorical predictions changed and the largest absolute
option-probability drift was `0.02677134`; the limits were zero changes and
`1e-4`. The archived zero-step prediction SHA-256 was
`eabd788e4e656974e3f380cdff0d32f1ad0887ddb2034235b2c43b9c3f994de3`;
the new receipt was
`d7c8db9989d44ed120704098fad19b6078c493be758f09040f042234b629c639`.
This is a STOP, even though both aggregate accuracies happen to be similar.
The new candidate cannot be compared causally with that preserved control.

The one-update smoke container was started before the separate parity
comparison finished. This violated the preregistered order; the mistake is
recorded rather than reinterpreted as a pass. The smoke completed one
optimizer update with finite loss and gradient, six weighted Choice rows in
the 16-row window, 5,131 native input tokens and a durable checkpoint. Its
`max-steps=1` schedule differs from the control and makes its model score
non-comparable. Its provenance/COMPLETE/metrics SHA-256 values are
`b55f7bb4f118fecd58bde93cc49c74a4a0add895e7a68e93a3ee93062c902549`,
`000683c90f57358fd2428469647b07bcf9b38b5a376e663b3046a0401b643123`,
and `93f10be48854339a7da9d15b37ee6734af293d2feee75fb47f573cbfc8fb4f25`.
The provenance-to-completion interval was 64.25 seconds, approximately
0.01785 GPU-hour; container setup adds a small amount. No further optimizer
update was launched. The card reservation was released after memory/use
returned to idle.

## CPU-only root-cause finding

The image digest and model/data files matched, but the **effective runtime
did not**. The archived control container had `PYTHONPATH=/opt/decision-fla`
and imported the optimized `fla` package. The new smoke command overrode
that variable with `PYTHONPATH=/work`; a CPU-only import check showed
`fla` unavailable. During source inference, Transformers explicitly
reported that `chunk_gated_delta_rule` fell back to its reference PyTorch
implementation because flash-linear-attention was not installed. This
runtime-path difference is a concrete explanation for the large BF16
probability drift. It was **not** repaired inside the frozen experiment.

A future attempt needs a new prospective runtime protocol, a fresh output
directory, and a read-only zero-step process that finishes and verifies
parity *before* any one-step smoke is launched. Restoring the archived
optimized import path is necessary but does not itself prove parity.
Do not rename this failed arm, reuse its checkpoint, relax the original
drift limit, or call its smoke a development gain.
