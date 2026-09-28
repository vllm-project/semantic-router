# Lux 1.0 current-package v3 control: technical stop before predictions

The prospectively locked Lux 1.0 current-package control did **not** produce
JevArena v3 predictions or a score. The typed FINAL process stopped during
native ROCm initialization, before loading model weights or answering any
item, with `RuntimeError: No CUDA GPUs are available`. CSS15 was not started.
No formal gold or score was read. This remains a **HOLD**, not a zero score.

The CPU preflight verified all 29 current-bundle files, model revision,
previous five-slot gold-free repeatability receipt, exact native collector and
sealer, frozen typed FINAL 1,600 items/2,000 slots, CSS15 6,547 items/slots,
scorers, image and disk. It initially caught one incorrect character in the
preregistration's copied sealer SHA-256. That note was corrected and signed in
`0abc76a93` before GPU use; local and remote source bytes did not change. A
private run lock then copied the preregistration file's *pre-correction* hash;
a second immutable lock corrected that clerical reference before GPU use while
retaining the first lock. Neither correction involved a model, answer or key.

The one launched process used the pinned qualified image and package, but its
container set both `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` to the
physical device index. The native runtime reported no visible GPU. Applying
both filters may have caused double filtering after device renumbering; that
is a **hypothesis**, not a demonstrated root cause. The run stopped under the
one-shot rule rather than changing flags and restarting it in place.

| Evidence | Result |
| --- | --- |
| Typed original items / answer slots produced | **0 / 0** |
| CSS15 original items / answer slots produced | **0 / 0** |
| GPU wall time | **9.325 seconds, 0.002590 GPU-hour** |
| Retained task container | None; card memory returned to its pre-run baseline |
| Private corrected run lock SHA-256 | `f50feb543bdb3ee21cece60acf2ef38f3364efeb272c48f091c50233be54d4b1` |
| Private stop receipt SHA-256 | `469a5301edba4d20e0089d21cc7285a1c3893fcdda42b01cb18fd6d8e57ec039` |

No prior 1.0 historical DEV, pilot or public score is substituted for the
missing formal control. A separately signed, prospective runtime correction
must prove visible GPU and native current-package parity on gold-free inputs
before any future complete v3 collection. The old published-example failure
remains intact; its stale-bundle provenance does not excuse an unverified
runtime.
