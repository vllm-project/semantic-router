# Sol 2B zero-step backend × fresh LoRA result

**Decision: STOP/HOLD.** The pre-registered four-cell, gold-free factorial
completed on one exclusive physical GPU with no optimizer step, teacher
artifact, sealed label access or model upload. The source, SELECT700,
historical predictions, 32-ID roster, image and runner matched the private
lock before execution. All 128 cell predictions were valid. Four process
times summed to **106.649 seconds = 0.029625 GPU-hours**, below the fixed
180-second/0.05 GPU-hour cap; every cell finished under 45 seconds.

The signed pre-registration is
[sol2b-zero-step-factorial-prereg-2026-09-27.md](sol2b-zero-step-factorial-prereg-2026-09-27.md)
at commit `86bfe7552`. Its SHA-256 is
`aa230e73c22d0eff4a69de543d6e51e1d797e152763c5d92b11db615c842195d`;
runner SHA-256 is
`a8bbffa677d41bc404cf3b5b24eee478395e9fe699113c18e1fb69f15414f849`.
The exclusive private lock SHA-256 is
`cc42152396be5cf2a69c39ebd2f59bdeb386232d4ece065e8759d5a997eeb4fd`.
Raw prediction SHA-256 values in the frozen order are:

| Cell | SHA-256 | Historical maximum option drift | Historical categories |
| --- | --- | ---: | ---: |
| FLA bare | `59c2ac27b0376d716a94727afecfddff433ae3208567832b7d399264d1963d8c` | 0.017128855 | 0/32 changed |
| FLA fresh LoRA | `35028755bcd4f3c75586fefbcff156194c3fed71dc27757834e30b5c3c5b2387` | 0.017128855 | 0/32 changed |
| PyTorch reference bare | `4827ed37028a5c65a079187220d6beac1d4e8d45d449daedc72ab734350f676f` | 0.013650537 | 0/32 changed |
| PyTorch reference fresh LoRA | `c62e6e9fae61fbf8cf1ab1b6810aa24acaeea9d62148be5213b6c5b20c8acf15` | 0.013650537 | 0/32 changed |

The private factorial comparison receipt SHA-256 is
`3e7480dc340af8f204a5f02278badb0cd0e3852d0a8bf56165653a4d1e2e9e3c`.

| Frozen contrast | Maximum option drift | Rows over 1e-4 | Category changes | Gate |
| --- | ---: | ---: | ---: | --- |
| Fresh LoRA effect under FLA | **0** | 0/32 | 0/32 | PASS |
| Fresh LoRA effect under reference | **0** | 0/32 | 0/32 | PASS |
| Backend effect on bare source | **0.027090371** | 22/32 | 0/32 | FAIL |
| Backend effect with fresh LoRA | **0.027090371** | 22/32 | 0/32 | FAIL |

The newly run reference-bare 32 option distributions were exactly identical
to the earlier stopped reference probe across processes; maximum drift and
category changes were both zero. The zero-initialized LoRA wrapper therefore
does **not** explain this factorial's numerical difference. Backend choice
does change the numerical output materially. However, even the newly run
FLA-with-LoRA cell differs from the historical FLA+LoRA control by 0.01713,
so the preregistered rule **does not attribute the historical mismatch solely
to backend**. The historical control's exact numeric runtime state is not
recovered by either pinned path. The source/file and input hashes match;
its FLA path is inferred from the historical code and was not separately
attested in that old execution receipt. Thus
the remaining cause may involve unreproduced runtime state or nondeterminism,
but this experiment cannot identify it. There is no basis to relax `1e-4`
or to treat the old control as a strict paired numerical baseline.

The 2B replay treatment remains stopped before optimization. A later causal
replay test would require a separately preregistered common-runtime control
and repeatability proof, then a matched treatment; this four-cell result does
not authorize that GPU spend. Preserve the existing BEST320 exploratory
checkpoint and its development-only mixed evidence (typed DEV regression,
CSS-pilot transfer gain) without claiming a release score.
