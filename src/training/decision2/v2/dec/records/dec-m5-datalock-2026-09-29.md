# Decoder Milestone 5 — data lock, part 1: N5N (2026-09-29)

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
