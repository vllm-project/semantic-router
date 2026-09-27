# Official Qwen3.5 4B and 9B: ROCm interruption triage

This is a read-only incident diagnosis, not a change to either preregistered
training arm or its model-selection rule. All three interrupted containers
exited 139 and Docker marked `OOMKilled=false`:

| Arm | Last logged update | Durable checkpoint | Outcome |
| --- | ---: | ---: | --- |
| 4B initial | 125 | 64 | Native process core dump |
| 4B exact resume on another accelerator | 84 | 64 | Native process core dump |
| 9B initial | 112 | 64 | Native process core dump |

Three kernel events in the interruption window record Python/autograd segmentation faults
in `libhsa-runtime64.so.1.18.0` at the same library instruction offset.
There was no accompanying GPU reset, KFD fault, OOM-killer message, or
reported GPU RAS error. Host memory and disk had ample free capacity; live
GPU temperatures were ordinary. Two independent 2B arms in the same training
image continued making optimizer updates during the triage window. These
observations favor a ROCm runtime issue or its interaction with the larger
Qwen3.5 autograd workload over a Python-level model error, deterministic
training-row failure, simple memory exhaustion, or one defective GPU. They
do **not** prove the root cause or the unaffected status of the 2B runs.

The update-64 checkpoint file set exists for each interrupted arm, including
trainer state, adapter, head, and checkpoint metadata. The 4B exact-resume
process successfully loaded its update-64 state; the 9B state has not yet
undergone a post-crash resume-load verification. The recorded latest and
SELECT-best pointers both refer to update 64; later unsaved updates are not
candidate weights. The 4B original plus its single exact resume used about
0.44 GPU-hour against its 3.0 GPU-hour cap; the 9B arm used about 0.39 against
4.0. Finishing from update 64 appears possible within the original *time*
caps, but runtime integrity is currently unproven. Neither arm has a complete
one-epoch result, and neither can advance to development or formal evaluation.

## Safe disposition

1. Preserve the original container logs and complete update-64 checkpoints.
   Do not select update 64 because of any opened development or formal labels.
2. Hold another full resume until a small isolated diagnostic checks the exact
   training image's GPU runtime path. One discriminating no-training probe is a fixed
   synthetic GPU forward/backward loop with gradients but **no optimizer
   update, model weights, training data, or checkpoint writes**, repeated on
   an otherwise unreserved accelerator while collecting exit status and the
   corresponding kernel event. A pass only rules out an elementary runtime failure;
   it does not certify hours-long training.
3. A move to another environment is conditional on a genuinely available
   accelerator, the identical image digest and source-code/data hashes, the
   pinned official model snapshot, and checksum-verified transfer of the exact
   optimizer checkpoint. Verify the trainer's strict resume contract and
   source fingerprint before consuming further budget. The alternative
   environment is **not** currently an exact ready-to-run mirror; no transfer
   was attempted during this read-only triage.
4. If a resumed run fails the same infrastructure gate or the remaining
   original wall cap, mark that arm incomplete. Keep the existing SELECT
   readings as development diagnostics only; do not publish them as the 4B or
   9B Decision 2.0 result.

The trainer's exact-resume path already validates the saved source, contract,
optimizer, data cursor, and random states. A future run must retain the
original preregistered SELECT selector and downstream native-reload and
development gates.
