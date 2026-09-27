# Official Qwen3.5 4B Base: infrastructure interruption and exact resume

The preregistered one-epoch arm described in
`qwen35-4b-official-base-full-clean-v2-prereg-2026-09-27.md` exited with
status 139 after optimizer update 125. The host reported a segfault in
`libhsa-runtime64.so`; the process was not OOM-killed. The last complete
checkpoint is update 64, with saved optimizer, data cursor and random states.
The update-64 SELECT reading (515/700) is only a development checkpoint,
not a model selection or release result.

Resume the **same arm** from its complete update-64 checkpoint using the
trainer's exact-resume mode, unchanged source/data hashes and hyperparameters,
on an idle accelerator of the same model. Updates 65–125 from the interrupted
process are superseded by replay from the saved data cursor. Keep the
original 3.0 GPU-hour total cap, including time already consumed; if the
resume fails its contract/hash check, numerical gate, accelerator health, or
remaining budget, record this arm as incomplete. Do not choose the update-64
checkpoint on the basis of opened evaluation labels or restart the arm from
zero. Preserve the interrupted log, exit status, checkpoint and resume log.

Only a completed update-466 arm with the original SELECT selector, exact
reload, and then complete DEV/CSS development screen can advance to the
separately locked post-key formal comparison.
