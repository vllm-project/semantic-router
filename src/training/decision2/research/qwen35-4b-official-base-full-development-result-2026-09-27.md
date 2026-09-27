# Official Qwen3.5 4B Base: complete clean-v2 development screen

**Decision: ADVANCE to a separately frozen post-key formal comparison; not a
release result.** The one-epoch, official-source 4B arm completed its fixed
466-update budget, selected its checkpoint using SELECT alone, passed the
32-row native reload test, and exceeded the preregistered own-Nox 1.0
development proxy by 12.921 points. No CAL fitting, typed FINAL, 15-task CSS
evaluation, public JevBench, Hugging Face upload or release decision was made
from this result. The source protocol is
[`qwen35-4b-official-base-full-clean-v2-prereg-2026-09-27.md`](qwen35-4b-official-base-full-clean-v2-prereg-2026-09-27.md).

## Frozen identity and completed training

- Direct initialization: official `Qwen/Qwen3.5-4B-Base` revision
  `1001bb4d826a52d1f399e183466143f4da7b741b`; new dynamic-option
  decision head and rank-16 LoRA. No third-party Decision weight was used as
  an initialization. The direct source's two weight-shard SHA-256 values begin
  `df547074dce7` and `590fbaac095d`; the tokenizer hash begins
  `fe000e3ed39e`.
- Rights-clean v2 TRAIN 7,455 rows / 4,194,465 encoded tokens SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`;
  SELECT 700 SHA-256
  `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`;
  CAL 700 SHA-256
  `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`
  remained untouched.
- All 466/466 optimizer updates completed with exit 0 and a durable
  `COMPLETE.json` SHA-256
  `a8543f3b8173cecfe7028444a6e3b015a98c1a6cdd80757d362cc299dd4490c9`.
  The earlier two non-OOM native-runtime interruptions exited 139; the
  authorized continuation replayed from the durable update-64 optimizer/data
  cursor rather than starting a new training arm or changing a hyperparameter.
  No nonfinite loss appeared in the retained training event log.
- The fixed selector chose **BEST update 466**: SELECT 636/700, family-macro
  accuracy `.910278`, normalized family Brier `.070560`.
  `BEST.json` SHA-256 is
  `e96277b9e0b07b7ef8c8df4b2fcbb71b7bdb9e8b4b8ed58816d8381dc221e3b6`.
  Update 448 also had 636/700, but lower family-macro accuracy `.906944`;
  the selector, not DEV, fixed the final checkpoint.
- The exact 32 SELECT rows reloaded from BEST466 in the pinned image with
  batch size two. Category changes, p99 probability drift, and maximum
  probability drift were all **zero**, meeting the fixed parity limits of
  0 / `.005` / `.02`. Both development collectors used the same immutable
  LoRA checkpoint plus pinned Qwen source and returned model fingerprint
  `dc2a8267ec7315a48aa1b2c31e7157e79e50582c4061177bed32ca0ed2735372`.

## Same-panel development result

The candidate and `Decision-1.0-Nox-4B@0bb833504965c0eabdb9630b7bbd385cb2fe5cd4`
answered the same typed DEV 1,600 and human CSS pilot 1,430 prompts under
their recorded native adapters. Typed and pilot gold-free input SHA-256 values
are `a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a`
and `598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`.
Both candidate prediction files were sealed at `2026-09-27T13:31:43Z`, before
either candidate score file was produced; the private seal SHA-256 is
`4cc282a570621778356bb8c4b1768d2c15d8acc68ae388e30e106dffa070fa47`.
The candidate answered all **3,030/3,030** items, with zero invalid,
missing, over-budget or truncated outputs. Nox also had 3,030 valid answers.

| Development metric | Own Nox 1.0 | Official-source BEST466 | Change |
| --- | ---: | ---: | ---: |
| Typed four-family macro accuracy `T` | `.664375` | `.820625` | `+.156250` |
| Typed correct / 1,600 | 1,063 | 1,313 | +250 |
| Choice correct / 800 | 461 | 720 | +259 |
| Noul correct / 400 | 224 | 230 | +6 |
| Score correct / 400 | 378 | 363 | **−15** |
| Typed normalized Brier ↓ | `.249509` | `.119841` | `−.129668` |
| Typed ECE10 ↓ | `.136509` | `.023122` | `−.113388` |
| CSS pilot task-median macro-F1 `H` | `.413540` | `.520209` | `+.106669` |
| CSS pilot micro accuracy | `.460839` | `.513287` | `+.052448` |
| CSS pilot median Brier sum ↓ | `.750944` | `.638399` | `−.112545` |
| CSS pilot median ECE15 ↓ | `.168437` | `.175852` | **+.007415** |
| Frozen proxy `100 × sqrt(T × H)` | **52.41619** | **65.33730** | **+12.92111** |

The three pilot task macro-F1 values were discourse `.413540 → .520209`,
implicit hate `.374689 → .364611` and SemEval stance `.588763 → .638807`.
The implicit-hate regression remains visible despite the higher median.
Score predicted ordinal levels 0/1/2 in `101/95/204` cases versus Nox
`86/86/228` and gold `85/107/208`; the candidate did **not** collapse to
one level. However, its Score Brier increased `.046052 → .091505`, ECE10
increased `.033695 → .154857`, and expected-value MAE increased
`.102614 → .262087`; eleven gold-level-2 cases were predicted as level 0.
This is a material Score limitation to carry into formal evaluation and the
eventual product card.

The preregistered promotion gate was candidate proxy at least Nox proxy
**+2.0** points with invalid/over-budget excess at most one percentage point.
The observed +12.92111 and zero/zero invalid rates pass this *development*
gate. It does not establish JevArena v3 performance or justify publication.

## Reproduction receipts and resource use

The candidate typed and CSS prediction SHA-256 values are respectively
`f845d202dd20467bf866e6cc4d8a74f5665b195d1b3050ae63c0d4bef9e9ef70`
and `998a7d32963a2415de7ec095b9e27772d56519b3f27686fe40dd4096c10f1ac1`.
Their score-report SHA-256 values are respectively
`efcfc9d86ade66af72e91e6f9a0037a936877751731ed26a399e89e2c6a87e3d`
and `b772018a2c557fe18c7efd65e66a1c8ae508ad2fe67b0dd8e9dc68cadc3a26db`.
The Nox control predictions / reports were also retained with their pinned
release revision. The local source and the readout mirror contained 831
identical tracked Decision 2.0 files, with canonical manifest digest
`de76da19c0034d55b1b41637a81866ada70affe796296d1eeeebd0fef81ee3be`.
The native inference, typed scorer, CSS scorer and reload checker SHA-256
values are respectively `b25382a2667f4a82167d52e35a54dadea6b0838eef407127b23f5dc45b18914b`,
`d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`,
`cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`,
and `19160f9d7aeb0a1726c0a4efb7887457b97a61af9ff0c6b9d65eb0f174a373f1`.

The original, first continuation and successful final continuation used
`.37768`, `.06568` and `1.11069` one-GPU hours, totaling **1.55405 GPU-hours**
under the original 3.0-hour training cap. The reload, typed DEV and CSS
pilot readouts used `.00711`, `.04244` and `.03953` GPU-hours, totaling
`.08908`; candidate training plus readout used **1.64313 GPU-hours**. The
separate once-only own-Nox development control used `.04026 GPU-hours`.
CPU-only scoring is excluded. All final readout containers exited zero and
remain preserved with their private receipts; no GPU job from this screen is
left running.

**Next action:** review this frozen development result, then lock the same
BEST466 package and unaltered adapter in a separate post-key JevArena v3
and public-231 comparison against the own 1.0 and admitted open peers. Preserve
the Score and implicit-hate losses in all downstream analysis. Neither
CAL-derived temperature nor public/formal performance may select another
checkpoint for this arm.
