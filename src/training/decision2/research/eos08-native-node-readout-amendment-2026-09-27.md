# Eos 0.8B matched replay: prospective readout location amendment

**Sign before any soft DEV/CSS prediction or scoring.** The hard and soft
498-step arms are already complete; SELECT chose their respective BEST
checkpoints before this amendment. Their initializer, data, update/token
budget, loss, checkpoint selection, calibration policy, development margins,
Score guardrail and protected-label rules do not change. No soft DEV/CSS
label or prediction has been inspected. The earlier hard first-node readout
remains an immutable diagnostic, never a checkpoint selector.

The planned transfer of the soft serving package between the two authorized
GPU nodes is proceeding too slowly to be a good use of the development
window. Both complete, checksum-addressed hard and soft checkpoint packages
already reside on the training node. Instead of waiting for a duplicate
package transfer, run **both** arms under the same native code, image,
physical GPU, deterministic reference backend, prompt files, scorers and
inference settings on that node. The 1.0 native control must be rerun there
too if it is included in same-node comparisons. This changes the readout
location for both arms together; it does not substitute historical scores.

Before GPU readout:

1. Verify the hard and soft fixed498 and SELECT-BEST packages are complete,
   immutable and bound to their original SELECT prediction hashes. Verify
   training and scorer source hashes, prompt and gold hashes, source weight
   revision, image digest, GPU occupancy and a task-owned reservation.
2. Mirror the exact local readout source into a new isolated directory on the
   training node. The training source mirror differs only in the later
   inference/backend enforcement and calibration entrypoint. Do not edit the
   source used by either optimizer. Hash the new mirror against the local
   source and the completed first-node readout code before using it.
3. With the exact readout code, rerun all 700 SELECT rows for each of the four
   checkpoint identities. Require ordered ID, task type, prompt/token hashes,
   input lengths and offered-option domains to match the frozen source-node
   SELECT receipts. Require zero category changes and maximum option
   probability drift `<=1e-4` on both full 700 and the preselected 32 rows.
   A failed check stops all second-node DEV/CSS scoring; do not shrink the
   roster or widen the threshold.
4. Rerun hard and soft fixed498 plus each distinct SELECT-BEST checkpoint on
   the same typed DEV1,600 and CSS pilot1,430 input files. Use raw
   temperature one, reference gated delta backend, max length8,192, identical
   native model adapter, sealed prediction files before scoring, and unchanged
   typed/CSS scorers. Fit CAL700 only after SELECT-BEST identities are fixed;
   calibrated views remain separate and cannot select an arm.
5. Compare the second-node hard output to its previously sealed first-node
   counterpart. Require identical prompt hash, counts and categories, and
   option probability drift `<=1e-4`. If this parity fails, report two
   different runtime conditions rather than mix scores. The causal contrast
   is **second-node fixed498 soft versus second-node fixed498 hard** under the
   original development margins and Score/invalidity guardrail.

Keep the aborted slow-transfer bytes, elapsed time and reason in private
experiment receipts. Do not open typed FINAL, CSS15 or public JevBench in this
amendment. This prospective operational change should be identified in the
interim result, including any parity or compute-budget failure.
