# Decoder Milestone 5 — data lock, part 1: N5N; part 2: N5BN and MLX-DEV (2026-09-29)

Under `dec-m5-prereg-2026-09-29.md` (c62cc0853). This part locks **N5N** only. **N5B, N5BN and the MLX-DEV panel
are not locked**: the preregistered MLX-DEV segment rule is infeasible as written (below), which changes the N5B
mixture, so those arms wait for an amendment and a part 2 of this record. N5N (N4XF rows unchanged) does not
depend on MLX-DEV. All files are on node B under `/data/dev2/runs/dec/m5/` unless stated.

## Code and runtime

- Code: exact mirror of `8e7f345e1` (`/data/dev2/src/8e7f345e1237820e52fed7a0df8fea7486130cf8-src_training_decision2`,
  tree `a2a83a2a`), which adds `v2/dec/{m5_block,mlx_dev,m5_labels}.py`, `specs/m5-block-superset.json` and
  `v2/dec/ops/m5/` (prep, lock check, chains, soup, MLX-DEV readout, GPU hours).
- Image `sha256:dbe5f32b2263…`, Triton cache `/data/dev2/runs/dec/triton-cache/dbe5f32b2263` (shared, rw, autotuning on),
  node B GPU3 (labels); containers `dec-m5-*`, `--network none`.

## N5N mixture

N4XF's file unchanged: `m4/data/m4-xl-full-29m/train.jsonl` `c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60`,
58,742 rows, 29,407,326 native tokens (budget ±0%), 0 rows of the 305 r2-excluded groups. Multilingual block
(non-English A7q / A7k / A7s / H5 / H8): 14,298 rows (A7q 1,895, A7k 316, A7s 1,852, H5 Noul 6,607, H5 Choice 492,
H8 Noul 1,266, H8 Choice 1,006, H8 Score 864); the 16 English H8 Noul rows are outside the block.

## Own-Lux h-w1 (H8)

Pinned revision `75e557f170979bdbc428b6ea698a2047e2d2a5cd`, downloaded with the node's token into the node-B HF cache:
`m3/teachers/lux1/xl/h-w1.targets.jsonl` `679ef009aba751c23061b406319ba5c6f255089361fee5baefafaabac12dc2c3`,
`coverage-r2.json` `ecb6dc36e19b5847d3b11d60d18a7f2d79190f680ec4fb41706e47b6ed629bab` (both match the prereg prefixes);
also `h-w1.report.json` `0c8d8d78…`, `h-w1.attestation.jsonl` `9fc691d9…`. It covers all 3,152 N4XF H8 rows
(input hashes equal). Restricted to the 16 English H8 rows for N5N: `teacher/hw1-n4xf-en/targets.jsonl` `0d6ef41e…`.

## Own-Nox replay labels (N4XF block rows)

`v2.dec.teacher_label` with the M3 own-Nox convention (default flags: `--teacher-kind decision1`, the package's
`temperature.json`, 8,192 tokens), `llm-semantic-router/Decision-1.0-Nox-4B@cde2a68dbaa557ea65dc458104d410a0802ee259`
(temperature 1.3231 for all three types), one job on node B GPU3:

- Input `teacher/nox-rows-n5n/rows.jsonl` `63b6bd6f961b8c2ce159504996263f3823cdf856ba05dd4d7f1a78f85667628e` (14,298 rows = the block).
- Labels `teacher/nox-n5n/labels.jsonl` `f1dded5343b8f9ed9ff0ff70e3547a004bcc4fae80015921dfdbddbfae4984d7`, 14,298 rows, 126 s
  of labeling (147 s wall). Gold agreement: Choice .826, Noul .789, Score .438.
- **Re-label check (preflight 2): PASS.** Sample = 4 whole `teacher_label` batches in sha256(`dec-m5-relabel-v1\0<first id>`)
  order until ≥ 300 rows (334 rows, `teacher/nox-check-n5n/check.jsonl` `828c7389…`), so the re-label sees the same batches,
  order and padding as the full run; re-label `teacher/nox-n5n-relabel/labels.jsonl` `8e24f21e…`: 334 / 334 records
  bitwise identical (max |Δp| 0).
- **M3 overlap (preflight 2): PASS.** 1,885 rows (all H5) share id and input hash with the M3 own-Nox file
  (`m3/teacher/nox/nox-teacher.jsonl` `2afe3048…`): argmax agreement 1.000 (threshold .97), mean |Δp| 1.0e-5.

## N5N teacher (preflight 3)

`teacher/n5n/teacher.jsonl` **`d42c8781fdfd73a6f5a661e554c6f2e5cc2ba9a8c88ab3d1458cd8e62b313ee6`** (manifest `34e295c6…`),
first-source-wins: own-Nox labels (14,298 used = every block row), N4XF's composed own-Lux file
(`m4/teacher/m4-xl-full-29m/lux-teacher.jsonl` `e2ff27ce…`, 43,821 used), h-w1 on English H8 (16 used). Covered 58,135 of
58,742; missing exactly the 607 H7 rows (gold-only, `--allow-missing-pool mx-xl-full-r2.ids.jsonl:H7`, trained with
`--teacher-partial`); 0 input-hash mismatches. Nox vs Lux argmax agreement on the 11,162 block rows both cover: .819.
Lock check `lock-n5n.json` `561cfd28053b159092382c863fb2cee04642c2e61d1a09154e1fe921a92f5745`: **PASS** (preflights 1–3).

## Training-data diagnostic (report only)

Block Noul rows of N4XF (answerability / relevance only), argmax of the targets: predicted-yes rate / gold agreement /
gold-No recall. Lux = N4XF's own-Lux file (H5) and h-w1 (H8; N4XF trained H8 on gold only); Nox = the labels above.

| Scope | rows | gold yes | Lux yes | Lux agree | Lux No-recall | Nox yes | Nox agree | Nox No-recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 7,873 | .500 | .557 | .795 | .738 | .516 | .789 | .773 |
| H5 | 6,607 | .500 | .565 | .816 | .751 | .511 | .811 | .800 |
| H8 | 1,266 | .499 | .513 | .683 | .670 | .542 | .673 | .631 |
| ar | 384 | .500 | .638 | .763 | .625 | .604 | .766 | .661 |
| bn | 220 | .500 | .509 | .882 | .873 | .436 | .818 | .882 |
| de | 462 | .500 | .662 | .786 | .623 | .608 | .801 | .693 |
| es | 770 | .500 | .543 | .770 | .727 | .506 | .786 | .779 |
| fa | 217 | .498 | .452 | .687 | .734 | .498 | .668 | .670 |
| fi | 364 | .500 | .574 | .805 | .731 | .588 | .786 | .698 |
| fr | 684 | .500 | .556 | .775 | .719 | .531 | .765 | .734 |
| hi | 218 | .500 | .486 | .665 | .679 | .495 | .656 | .661 |
| id | 366 | .500 | .530 | .795 | .765 | .557 | .795 | .738 |
| ja | 884 | .500 | .549 | .877 | .828 | .502 | .860 | .857 |
| ko | 736 | .500 | .538 | .780 | .742 | .444 | .770 | .826 |
| ru | 346 | .500 | .523 | .751 | .728 | .483 | .769 | .786 |
| sw | 241 | .498 | .469 | .813 | .843 | .444 | .755 | .810 |
| te | 344 | .500 | .517 | .802 | .785 | .442 | .808 | .866 |
| th | 346 | .500 | .552 | .734 | .682 | .471 | .705 | .734 |
| zh | 741 | .499 | .632 | .776 | .644 | .584 | .794 | .709 |
| zh-hant | 550 | .500 | .551 | .905 | .855 | .500 | .891 | .891 |

Lux targets lean yes on the block's H5 Noul rows (.565 vs gold .500; ja .549, ko .538, de .662, zh .632); Nox targets
are closer to balanced (.511) with higher gold-No recall (.800 vs .751). File `teacher/diag-n4xf/diag.json` `1f0bdd6f…`.

## N5B / N5BN / MLX-DEV: blocked pending an amendment

The prereg drops an MLX-DEV candidate group if any normalized input segment (≥ 20 characters) occurs in any N4XF row.
Instruction and option lines are shared templates (for example "rate the overall sentiment that the message
expresses." in 2,716 N4XF rows, "final assistant reply:" in 3,237), so every N4XF-unseen candidate group of
JCommonsenseQA, MTOP, A7q, A7k, A7s and H8 Spanglish conflicts: those six Choice / Score cells would be empty
(Choice-ML, Score-ML and M_dev undefined); the two Noul cells fill (H5 15 languages × 100 groups, MIRACL 5 × 80).
Ignoring segments that occur in more than N distinct N4XF groups (content lines occur in ≤ ~5 groups, templates in
~800–3,200) gives identical exclusions for every N in 10–50 and fills every cell. The N5B mixture is defined under
either reading but differs (it excludes the MLX-DEV groups and rows sharing a segment with them), so no N5B / N5BN
training starts before the rule is amended; the builder takes the threshold as `--template-max-groups`.

## GPU hours so far

Node B GPU3: own-Nox labels 0.0408 + re-label 0.0061 = **0.047 GPU-h** (receipts `teacher/nox-n5n{,-relabel}.launch.json`).

## Launch (after this record is pushed)

`data/n5n/READY` is written after this commit; `ops/m5/m5-chains.sh <mirror> n5n` then runs, seed-paired with N4XF,
GPU3 `m5-N5N-s1` → `m5-N5N-s3`, GPU4 `m5-N5N-s2`, each through `drive_arm.sh` (zero-step, one-step + reload, preflight
gate, full run, postrun) with N4XF's arguments except the teacher: `--train /runs/m4/data/m4-xl-full-29m/train.jsonl
--teacher /runs/m5/teacher/n5n/teacher.jsonl --teacher-kl-weight 1.0 --teacher-partial --train-mode full --backbone-lr
5e-6 --head-lr 5e-5 --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64 --seed
20260926 / 20260927 / 20260928`; the chain finishing the last seed builds the soup and its readouts (`m5-soup.sh`).

# Part 2: N5BN and MLX-DEV locked; N5B stopped (2026-09-29)

Under the prereg and [amendment 1](dec-m5-amendment-1-2026-09-29.md) (`e574bbf56`: `--template-max-groups 50`, the
implementation clarifications, N5N-first schedule). This part locks **N5BN** and the **MLX-DEV** panel. **N5B is
stopped at preflight 3** (teacher coverage, below) and does not train until it is amended; no N5B file is READY.

## Code

Built by the block phase of `8e7f345e1` (mixture, panel, own-Nox labels on the added rows, checks). After the N5B
teacher compose failed, `8e7060627` (tree `50557396`, mirrored to node B) made the launchers per-arm and nothing
else: `m5-prep.sh` goes on past a failed teacher compose, writes the uncovered-row file and still builds the N5BN
teacher. `m5-lockcheck.py` reports each arm's status. `m5-chains.sh` gains the phases `n5bn` / `n5b`, which split
the block phase by arm and keep each seed on the same GPU. The report-only diagnostic now reads the Lux sources
directly. No builder, labeller, composer or trainer code changed. `8e7060627` also adds `ops/m5/m5-select.py`
(selection rules 1–4).

## N5B / N5BN mixture (shared file)

`data/n5b/train.jsonl` **`ae47b8258005a267d1f9a8c85ec20e77e22a07197ad765d78aa5608c9e6cd5c3`** (manifest `18891bbd…`),
63,075 rows (Choice 24,989 / Noul 20,069 / Score 18,017), 29,450,191 native tokens (+0.146% vs 29,407,326). It is
identical to the staging build under the same threshold. Superset `data/superset/train.jsonl` `d8e6c796…` (107,836
rows, 42,627,506 tokens; `specs/m5-block-superset.json`). Rows outside the block are identical to N4XF's (same ids,
order and bytes). The block has 18,631 rows: 11,297 N4XF block rows kept (the Noul cell shrank by 3,001 rows,
stratified by language) and 7,334 rows added (A7k 909, A7q 780, A7s 1,093, H5 Choice 1,165, H8 Choice 2,424, H8 Score
963). Quotas are block-token shares against N4XF's block (5,513,920 tokens); the tolerance is ±5% relative:

| cell | quota | share | rel. dev. |
| --- | --- | --- | --- |
| Noul (H5 + H8) | .45 | .4553 | +1.19% |
| A7q | .20 | .2022 | +1.11% |
| A7k | .06 | .0601 | +0.15% |
| A7s | .09 | .0900 | +0.03% |
| H8 Choice (JCQA) | .09 | .0900 | +0.01% |
| H8 Score (Spanglish) | .05 | .0500 | +0.01% |
| H5 Choice (MTOP) | .06 | .0601 | +0.14% |

The N5B mixture and N4XF both have 0 MLX-DEV groups, 0 MLX-DEV ids, 0 rows sharing a non-template segment with
MLX-DEV (406 template segments ignored) and 0 rows of the 305 r2-excluded groups.

## MLX-DEV panel

`mlxdev/build/panel.jsonl` **`100ae4e770973640e00b79115e50c7fcca7d4837c22e596690ed6023530652a7`** (index `6ffa4b84…`,
selection `data/n5b/mlxdev.selection.json` `98cf454d…`), 9,386 rows in 3,802 groups (`split` / `evaluation_role`
select). A cell can overshoot its row target by at most its last group.

| cell | groups | rows | languages (rows) | gold balance |
| --- | --- | --- | --- | --- |
| H5 Noul | 1,500 | 6,077 | 15 × 100 groups: ar 200, bn 200, de 834, es 564, fi 200, fr 974, id 200, ja 392, ko 220, ru 200, sw 219, te 200, th 200, zh 846, zh-hant 628 | 3,038 yes / 3,039 No |
| H8 MIRACL Noul | 400 | 801 | es / fa / fr / hi 160, zh 161 (80 groups each) | 400 / 401 |
| H8 Choice (JCQA) | 396 | 400 | ja | 5 keys, 73–89 each |
| H5 Choice (MTOP) | 196 | 401 | de 87, es 80, fr 69, hi 90, th 75 | 9 keys |
| A7q | 212 | 607 | 15 languages (es 283, ru 130, zh 61, …) | 5 levels, 116–128 |
| A7k | 398 | 400 | ja 173, ko 227 | 6 levels |
| A7s | 400 | 400 | hi-en 390, sw 10 | 3 levels |
| H8 Score (Spanglish) | 300 | 300 | es-en | 3 levels |

## Own-Nox labels on the added block rows (N5BN)

Same convention and package as part 1. Co-located on node B GPU4 while `m5-N5N-s2` trained (113.9 of 274.5 GB VRAM
in use before the job). Input `teacher/nox-rows-n5b-add/rows.jsonl` `c9afd673…`: 7,334 rows, which are the N5B
block rows not already labelled in part 1. Labels `teacher/nox-n5b-add/labels.jsonl`
**`545c513f0231188039db0b399f7d9d6019f3344f7808ace88b78b97e78d00c2f`**: 75.5 s of labelling (95 s wall).

- **Re-label check (preflight 2, amendment 1 item 6): PASS.** 3 whole batches (324 rows,
  `teacher/nox-check-n5b-add/check.jsonl` `9270617d…`); re-label `613243dd…`: 324 / 324 bitwise identical.
- **M3 overlap: PASS.** 319 rows: argmax agreement 1.000, mean |Δp| 2.7e-6.

## Teachers (preflight 3)

- **N5BN: PASS.** `teacher/n5bn/teacher.jsonl` **`eeb39c2401e546039bce6a8f7a3f35739f86652cb010b9b4cc738f06b9ebdf0d`**.
  First source wins: own-Nox on all 18,631 block rows (11,297 from `nox-n5n`, 7,334 from `nox-n5b-add`), then
  N4XF's Lux file for 43,821 rows, then h-w1 for the 16 English H8 rows (`teacher/hw1-n5b-h8/targets.jsonl`
  `3464648f…`, 6,059 rows, all of N5B's H8). The XL-r2 waves are used for 0 rows. 62,468 of 63,075 rows are covered;
  exactly the 607 H7 rows are missing; 0 input-hash mismatches.
- **N5B: FAIL, so N5B is stopped.** The preregistered sources (N4XF's Lux file, the XL-r2 own-Lux waves w1–w5 and
  c-w1, h-w1 for H8) leave **1,054 added MTOP rows** (H5 Choice; de 222, es 213, hi 213, fr 205, th 201; 535 groups)
  without a target. The waves cover only the 111 added MTOP rows that were in XL-r2 mixtures. The composer refused
  the file, as it should. The uncovered rows, those 1,054 plus the 607 H7 rows, are in
  `teacher/missing-n5b/uncovered.jsonl` `0f466536…`, ready if an amendment chooses own-Lux labels. The alternatives
  are gold-only rows or a different MTOP draw. No N5B training until then.

Lock check `lock-block.json` `4d9b13b4248ef7b1bbb68d1f5fe0eb2569a176ab56efe6c717726001ea568873`: overall FAIL,
`fails = [teacher_n5b]`. Per arm: **N5BN PASS, MLX-DEV PASS, N5B FAIL.**

## Training-data diagnostic, N5B / N5BN block Noul rows (report only)

4,872 rows (H5 4,086, H8 786), gold yes .500. Lux targets: predicted yes .564, gold agreement .792, gold-No recall
.728. Nox targets: .521 / .785 / .764. On H5 alone, Lux is .571 / .813 / .741 and Nox .513 / .807 / .794; on H8, Lux is
.525 / .686 / .660 and Nox .562 / .669 / .607. File `teacher/diag-n5b-sources/diag.json` `7506f747…`.

## GPU hours so far (completed receipts)

N5N-s1 and N5N-s2 zero-step, one-step and gate: 0.0761 + 0.0739. Own-Nox labels, part 1: 0.0469. Own-Nox labels on
the added rows, part 2: 0.0264 + 0.0069. **Total 0.230 GPU-h.** The two N5N full runs are in flight.

## Launch (after this record is pushed)

`data/n5bn/READY` and `mlxdev/READY` are written after this commit, and `ops/m5/m5-chains.sh <8e7060627 mirror> n5bn`
queues behind the N5N chains on each GPU's flock:

- **GPU3:** the MLX-DEV baselines (Nox 1.0, N4XF s1–s3 BEST, N4XF soup, then the pending N5N-soup comparison), then
  `m5-N5BN-s2`.
- **GPU4:** `m5-N5BN-s1`, then `m5-N5BN-s3`.

Each seed keeps its preregistered seed and uses N4XF's arguments, with `--train /runs/m5/data/n5b/train.jsonl
--teacher /runs/m5/teacher/n5bn/teacher.jsonl`. The chain that finishes the last seed builds the soup and its readouts,
including MLX-DEV. N5B's chain (`n5b` phase) launches only after an amendment and a part 3 of this record.

# Part 3: N5B locked (2026-09-29)

This part is written under [amendment 2](dec-m5-amendment-2-2026-09-29.md) (`a131563c5`). The added rows take Lux
targets from the full published source list of the M4 composition. Code: `b35c65954` (tree `d7c244ca`, mirrored to
node B). The prep script composes `teacher/n5b-a2` from N4XF's composed file first, then the M4 sources in M4
precedence order, then h-w1 on H8 rows. The failed attempt with only the prereg's list stays in `teacher/n5b` as the
record of that failure. The lock check now also compares rows outside the block with N4XF's targets byte for byte.
The mixture, the N5BN teacher and the MLX-DEV panel are unchanged from part 2.

## N5B teacher (preflight 3): PASS

`teacher/n5b-a2/teacher.jsonl` **`9c85485a751bad82dee1519be7772287e03ea086d77fb8268de274fa20fd2da4`**, first source
wins. Rows used per source:

| source | rows used |
| --- | --- |
| N4XF composed file `m4/teacher/m4-xl-full-29m/lux-teacher.jsonl` (`e2ff27ce…`) | 52,462 (43,821 outside the block + 8,641 kept block rows) |
| `A0-train.canonical` (`d8eae3e4`) | 0 |
| RP-v2 wave 1 (`7885baf6`) | 241 |
| RP-v2 wave 2 (`6bd8eb4d`) | 523 |
| RP-v2 wave 3 (`002e5b42`) | 290 |
| RP-v2 wave 4 (`03b1e72d`) | 0 |
| XL w1 (`10053613`) | 2,782 |
| XL w2, w4, w5, c-w1 | 0 |
| XL w3 | 111 |
| h-w1 on N5B's H8 rows (`teacher/hw1-n5b-h8/targets.jsonl` `3464648f…`) | 6,059 |

The file covers 62,468 of 63,075 rows. Exactly the 607 H7 rows are missing, and they train gold-only with
`--teacher-partial`. There are 0 input-hash mismatches. All 43,821 rows outside the block that N4XF's file covers keep
their target byte for byte (0 changed). The 1,054 added MTOP rows that part 2 found uncovered take the RP-v2 waves
(241 + 523 + 290). M4's own ret-gap labels are not a published source and are not used; N4XF's own composition used
them for 0 rows.

Lock check `lock-block.json` **`51d6414f2cd9b1e4e528ac1784d5e278450addf861e50739f90b114b38285418`**: **PASS**. Per
arm: N5B PASS, N5BN PASS, MLX-DEV PASS (preflights 1–3). The part-2 result is kept as `lock-block-part2.json`
(`4d9b13b4…`).

## Launch (after this record is pushed)

`data/n5b/READY` is written after this commit. `ops/m5/m5-chains.sh <b35c65954 mirror> n5b` then queues behind the
running chains on each GPU's flock:

- **GPU4** frees first, after N5BN-s3: `m5-N5B-s1`, then `m5-N5B-s3`.
- **GPU3**, after N5BN-s2 and the N5BN soup: `m5-N5B-s2`.

Each seed runs with N4XF's arguments, `--train /runs/m5/data/n5b/train.jsonl` and
`--teacher /runs/m5/teacher/n5b-a2/teacher.jsonl`. The chain that finishes the last seed builds the soup and runs
the CAL698, typed DEV, CSS pilot, SELECT, development and MLX-DEV readouts, including the comparison against the
N4XF soup.
