# 0.6B gradient-conflict preflight: mechanism signal, no model result

The fixed read-only diagnostic in
[the prospective design](small06-gradient-conflict-next-arm-2026-09-28.md)
passed its **mechanism** threshold: six of eight prescribed mixed-type
accumulation windows had a task-gradient pair with cosine at most `-0.05`.
This is **not** a trained checkpoint, development score or release gain.
No optimizer was constructed, no weights were updated, and no SELECT, CAL,
DEV, FINAL or public benchmark input or answer was read. The one-use sample
was selected from TRAIN by its existing shuffled order and task types alone.

The signed implementation is commit `055fdfff4` and its preflight script
SHA-256 is `a3c54809b5265e992e198f65c2560bf50c386a8e38975cfc168373ed9bacfeb7`.
The official `Qwen/Qwen3-0.6B-Base` direct source remains pinned to revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; the two primary weight/config
file hashes matched the archived control. The frozen TRAIN SHA-256 is
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
The default CPU pass reconstructed all **7,455 rows and 4,094,489 native
tokens**, then sealed the first eight complete Choice/Noul/Score windows.
The private CPU-plan receipt SHA-256 is
`b4029e66e3b93b126a906d669cce82ffebdef06cf0689f322ad2c3d1d16c2699`.

One freshly idle BF16-capable GPU ran the exact mirrored code with a bounded
external watchdog. The gradient measurement consumed **26.828 seconds =
0.007452 GPU-hour** after model load and released the GPU afterward; total
device occupancy including load was not separately timed. Its private
aggregate receipt SHA-256 is
`c3c9dc09fb1fe07b3f5ffe0cc488feb1d82d378908c1e701550ddf7a986d6416`.
All eight windows completed. The largest type-sum reconstruction error was
`8.94e-8`, below the frozen `1e-5` ceiling. Across the eight windows,
Choice/Noul crossed `-0.05` in four, Choice/Score in one, and Noul/Score in
four; overlapping windows make the union **six**, not nine independent
observations. The minimum measured pairwise cosines were approximately
`-0.341`, `-0.102` and `-0.252`, respectively. These are eight related
TRAIN windows from one initialization, not a confidence interval or proof
that gradient conflict caused the downstream Choice/Score regressions.

The preregistered next gate is a **projection-disabled, one-update** test:
reconstruct the unmodified 466-step schedule and compare the modified
trainer's ordinary-gradient path against an independently executed
same-schedule control on 32 fixed SELECT prompts. It must retain zero
categorical changes and at most `1e-5` maximum probability difference,
plus finite gradients and exact source/data identities. The current
per-type bookkeeping equality does not replace that independent trainer
parity test. Only if it passes may the one fixed gradient-projection arm
start; any completed arm still needs its frozen SELECT, DEV/CSS and separate
source-disjoint confirmation gates before a new 0.6B product claim. The
existing private model and all historical scores are unchanged.
