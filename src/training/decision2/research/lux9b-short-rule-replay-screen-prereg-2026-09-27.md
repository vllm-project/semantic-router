# Own-Lux 9B short-rule replay screen

Status: frozen proposal before data build, optimizer start, or new model output.
This is a development-only, bounded test. It cannot inherit a previous 9B
checkpoint's score or a third-party model's Index result.

## Question and controls

The completed own-Lux human 5,824-row arm reached typed DEV 86.9375% and CSS
pilot task-median macro-F1 0.580324. Adding 2,536 structured examples in the
prior 8,360-row arm increased typed DEV to 87.3750% but reduced the pilot F1
to 0.573976. Its added rows were 65.14% of input tokens. Test whether a much
smaller, short, group-complete rules/evidence/transition replay can improve
typed decisions without losing human-task transfer. The 5,824-row run is the
matched source and optimizer control; the structured 8,360-row run is a
descriptive negative control. No completed control will be retrained.

## Frozen source and data

- Direct initializer: our `Decision-1.0-Lux-9B` at HF revision
  `bd45a30aee8c84032791c245c70f86dee5389cc8`. Its 16 source files and
  native 32-prompt two-process identity were already audited. New LoRA zero-step
  output must match that source on the same 32 prompts before training.
- Human TRAIN5,824 SHA `e83fb07021b779bb86d6b1d773b007c2dda9d91052aedf1f72f89bebbfef50e2`;
  SELECT600 SHA `d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38`;
  CAL900 SHA `bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf`.
- Add only the 270 previously audited, TRAIN-only short-source IDs from the
  immutable pool SHA `f2390fe0c5540c39d7ad8886fe9fc35054f253398092a841fb25bd411e894baa`
  and manifest SHA `02cdf7e644e8f5fc75f6a908ceb5b6dcc6a7d7e172c047b0bcfc684d4e73be13`.
  Do not refill dropped groups or create new labels. The first CPU check found
  zero shared IDs, groups or input hashes with human TRAIN/SELECT/CAL.
- The signed builder binds five answer-free panel prompt SHA values and
  quarantines whole groups touching any human partition or protected panel by
  ID, group, input hash, exact/normalized state, or approximate near state.
  Limit each added row to 1,024 **Lux-tokenized** tokens, retain 180–270 rows,
  cap the added input-token share at 20%, and require no contradictory gold
  groups. Fail closed if these conditions do not hold. The near check is an
  approximation; record its limitation. Do not read FINAL or CSS15 answers.
- Keep exact source/rights and omission metadata in the private manifest. The
  public release decision requires a separate source-by-source assessment; a
  private dataset is not a public weight license.

## Bounded GPU plan and stop rules

After CPU build and complete frozen-input audit, run one exact-source,
32-prompt BF16 zero-step source versus fresh-LoRA smoke. Require 32/32 valid,
zero category changes, and probability max drift ≤0.02 under one runtime.
If it fails, stop with no training. A one-step numerical/length preflight must
have finite loss and gradient, no overlength admission, and memory below the
single assigned GPU; failure stops.

Use the existing Qwen3.5 text/decision-head trainer with direct own-Lux start,
rank-16 LoRA, alpha32, dropout0.05, CE + 0.5 Brier, LoRA LR 2e-5, head LR
1e-5, microbatch1, accumulation16, max length8192, seed20260926, BF16
backbone compute and FP32 head/loss. Train **at most 128 optimizer steps** on
one confirmed idle GPU, saving and scoring SELECT every32. No replay teacher,
new model initialization, or extra data is allowed. Cap the training screen
and smoke together at 2 GPU-hours; stop on nonfinite values, invalid
checkpoints, or a projected budget breach. Choose BEST by SELECT family macro,
then normalized Brier, then earliest step. Do not pick a DEV or public peak.

The existing human-only control reached SELECT600 0.596667 family macro at its
BEST192. If the new BEST is below **0.576667**, stop before calibration or
complete development inference. Otherwise fit CAL900 only after BEST is
fixed, then run one complete native typed DEV1,600, CSS pilot1,430 and public
JevBench231 diagnostic. Invalid and over-budget answers count as failures.
Promote to a prospective formal v3 roster only if the development proxy
`100*sqrt(T_dev*H_pilot)` exceeds the unchanged own Lux1.0 proxy by at least
**2.0 points**, public231 is at least its same-panel Lux1.0 count, and the
package satisfies full gold-free plus DEV/CSS-pilot native parity. Report every
slice tradeoff; no individual slice must improve if aggregate improves. A
candidate that misses these gates remains HOLD. The present 128-step screen
does not authorize an unplanned full-epoch continuation.

Formal typed FINAL/CSS15 labels have previously been opened for another 4B
arm. Any later formal 9B comparison must be described as a **post-key
prospective same-panel** run with its model, predictions and formula sealed
before this 9B score, not as a virgin blind test.

## CPU feasibility and zero-step receipt, before optimizer start

The fixed builder retained all **270/270** short source groups, zero
exact/normalized/near exclusions and zero overlength rows. The merged TRAIN
has **6,094** rows, SHA
`a9264ff9f97cba440d71317b514b6f1f9828146d8047d2ac229812013260e9eb`;
Lux-tokenized human/addition input counts are 1,846,849 / 135,143, so the
addition is **6.8185%** of training input tokens. The builder SHA is
`91ff157da550d3a875b14bc0225b52d09a31a4368ca6fce0189ae763c3678029`.
This is a gold-free data audit, not a model result.

The fixed 32 SELECT inputs passed a one-process native source-to-fresh-LoRA
comparison before any optimizer update: 32/32 valid, zero category changes,
maximum probability drift **0**, private receipt SHA
`9ff0d5247667fad57b3fb9d6137591fef3cd97db1e60db1c07bf33b6087872a6`;
model elapsed 28.33 seconds. The same pinned reference backend is required
for the one-step and 128-step arms. A later backend substitution needs a
separate zero-step gate and cannot inherit this receipt.

The trainer defaults omitted above are frozen here before optimization:
`eval_batch=2`, `epochs=1`, `weight_decay=0.01`, `warmup_ratio=0.05`,
`backbone_lr=1e-6` (frozen backbone), and gradient checkpointing enabled.
One-step preflight uses the same data/source/objective with `max_steps=1` and
`save_every=1`. The 128-step arm uses `max_steps=128`, `save_every=32` and a
distinct output directory; no one-step weights are transferred.

## One-step numerical feasibility, before 128-step start

The one-step arm completed and did not contribute initialization weights to
the planned 128-step arm. Baseline SELECT600 was 322/600, matching the
historical own-Lux source count. Step 1 used 16 examples / 4,609 input tokens;
loss 2.46328, gradient norm 14.61228 and all other logged values were finite.
Peak GPU allocation was 32.85 GiB. Step-1 SELECT remained 322/600. Private
provenance, completion, and SELECT receipt SHAs are respectively
`5e02b664373bd6fde4d76c8f6fcbeb12fb3d9263a59523f07716d97892faa37b`,
`c27de0a3e1dd1fa6140c93d23e9ba72dac6e7b66dad07533b0f4d2d552f57825`,
and `5c53704a56b4678eb2b52f7c1d99877bdedeeedf8a92f3b053038729ecd90572`.
The observed one-step compute took 9.84 seconds; the 128-step arm retains the
already frozen 2 GPU-hour cap and will stop on a projected overrun. No
development, public, or formal evaluation was performed on step 1.
