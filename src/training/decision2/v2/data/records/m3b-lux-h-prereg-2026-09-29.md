# Preregistration — own-Lux targets for the H7 / H8 rows of the XL r2 recipes (wave `h-w1`, 2026-09-29)

Committed at 2026-09-28 ~19:30 UTC (2026-09-29 ~03:30 UTC+8), before any prompt file of this wave exists and before
any GPU work. Approved by the coordinator on 2026-09-29 03:10 UTC+8 (decision 3: "an own-Lux wave for the 25,664
H7 / H8 rows, ~1 GPU-hour, new preregistration, node B GPU7"). It supersedes the "no teacher targets for H7 / H8"
line of [M3b amendment 2 §4](m3b-prereg-amendment-2-2026-09-28.md) for these rows only; nothing else in M3b changes.
AutoJev targets for these rows stay deferred (coordinator, 23:50 UTC+8).

## 1. Prompt set

- **Definition:** the union of `mx-xl-full-r2.lux1.missing.jsonl` and `mx-xl-short-r2.lux1.missing.jsonl` of the XL
  r2 release recipes (private revision `100536133e192c54ec57c2599a5e4706f6d334ff`, `m3/mixtures/xl-r2/`).
- **Expected:** 25,664 ids (mx-xl-full-r2: H7 4,292 + H8 21,372; the 17,945 mx-xl-short-r2 ids are a subset), all of
  them TRAIN rows of the H7 / H8 files published at `09f73967bc21b2b1e27160397272b7f66a1ef3af`
  (`h7.train.jsonl` `7c4133b0…3a32`, `h8.train.jsonl` `1f19e5ab…d361`), 28,498,792 native tokens.
- **Build:** node A, CPU, from the exact mirror of a pushed commit. `v2.data.m3.xl_prompts` with those two missing
  lists (full first) and the two TRAIN files as the only pools. Ids are sorted; one wave of at most 60,000, named
  `lux-xl-h-w1`, published as `h-w1.*`.
- **Stop before any GPU work if:**
  - the node-A missing lists differ from the files at `10053613…`;
  - a TRAIN file differs from the SHA-256 recorded in the r2 manifest;
  - the count is not 25,664;
  - any id is not an H7 / H8 TRAIN row.

## 2. Guard

- **Target guard:** the one of waves w1–w5 / c-w1 / c-w2 (`v2.data.m3.guard`), with the same protected files
  (SELECT700, CAL700, CAL698, every v2 AHO / SHO slice, the v1 and A7 AHO slices, every gold-free panel prompt file)
  plus the H7 and H8 AHO slices.
- **H7 / H8 SHO:** not read, because the program never accesses `/data/dev2/private/sealed/`. TRAIN and SHO are
  disjoint by the hash group split, checked at build time (`m3b-gap-sources-2026-09-28.md`).
- **Pass:** zero violations: not-a-row, not-TRAIN, prompt ≠ row prompt, shared id, input hash or prompt digest.
  Anything else stops the wave.
- **Repeat subset:** the guard also runs on the repeat prompt file (§4).

## 3. Teacher run: identical to the earlier XL waves

- **Teacher:** own Lux 1.0 `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8`, on node
  B GPU7.
- **Launcher:** the Milestone 2 `teach.sh lux` (`49e50c6e…`), in docker `--network none`, offline.
- **Image:** `decision20-lux-runtime:latest` = `sha256:ce895822…`, container source mirror `5f5cd80a…`.
- **Collector:** `inference.run --backend lux --over-budget-invalid`.
- **Triton cache:** the frozen node-B Lux autotune cache `triton-cache-lux-nodeB`, 1,762 files, tree `299151ab…`.
- **Queue:** a committed queue script, run detached from the node-B mirror. It starts the wave only when the prompt file
  has the expected SHA-256, then runs one process per file. On node B it writes only under `/data`
  (`TMPDIR=/data/dev2/tmp`).
- **Frozen check:**
  - Before the wave, the image id, launcher hash and cache must match the values above.
  - The cache file count and tree digest are recorded before the wave, after it and after the repeat run. All three
    must be 1,762 / `299151ab…`.
- **Run check** (`v2.data.m3.luxxl provenance`, as for w1–w5):
  - launcher rc 0;
  - collector summary: `collected_now = input_items = 25,664`, `previously_completed = 0`, the pinned identity, an
    attested revision, the validated runtime and no runtime differences;
  - the image id.
- **Over budget:** an over-budget prompt becomes a listed `no_target` entry, as in earlier waves. None is expected:
  the longest row is about 7.6K native tokens, below the 16,384 limit.
- **Image disclosure:** every report repeats the disclosure of the earlier XL waves (no causal-conv1d kernel; 43 of
  8,778 Lux1 argmax answers differ against the kernel image).

## 4. Repeat check

- **Prompts:** 256 prompts of the wave, the first 256 in `sha256("m3b-lux-h-repeat:" + id)` order. This is M3a's
  hash-order rule. The M2 repeat256 took the wave's first 256 lines, which here would all be H7 HoVer rows.
- **Run:** a separate, fresh `teach.sh lux` process after the wave, with the same image and cache.
- **Compare:** against the wave's answers for the same ids, with `v2.data.replay_targets.repeat_max_abs_diff`, plus
  counts of identical answers and argmax agreement.
- **Pass:**
  - 256 compared;
  - 0 validity mismatches;
  - 0 argmax flips;
  - max abs probability difference ≤ 1e-3 (the M3a G2 bound).
- **Expected:** bitwise-identical answers, as M2 repeat256 was. The receipt goes into the wave's `report.json`
  (`provenance.repeat_check`).

## 5. Conversion, publication, readback

- **Conversion:** as w1–w5, on node A from the exact mirror.
  - `v2.data.m2.targets --prompts --attestation --provenance`.
  - Per-row attestation: identity, attested revision, validated runtime, node B / GPU7 / image.
  - The prompt digest is recomputed from the training row.
  - Teacher-output ids = prompt ids = row ids.
  - Every prompt without a target is listed.
- **Pre-upload checks:** the path-leak check and the private-value guard on the report.
- **Upload:** a new revision of the private dataset `llm-semantic-router/decision-2.0-training-data`, under
  `m3/teachers/lux1/xl/`:
  - `h-w1.{targets.jsonl,attestation.jsonl,report.json}`;
  - a refreshed `README.md`;
  - a new `coverage-r2.json`. It is `v2.data.m3.luxxl coverage` over all eight r2 recipes, from the r2 manifest and
    missing lists, with waves `c-w2` (pending in that manifest) and `h-w1`. `coverage.json` (r1) is not changed.
- **Readback:** download the new revision and compare the SHA-256 of every uploaded file.
- **Expected coverage:** 100% of mx-xl-full-r2 (365,970) and mx-xl-short-r2 (373,577). The upload is refused unless
  `coverage-r2.json` shows zero rows left without a target in both.

## 6. Stop rules, budget, records

- **On failure:** any failed check stops the wave and nothing is published. It is recorded and not rerun without a
  new amendment (program rule).
- **Budget:** predicted 0.39 GPU-h for the wave (0.036 s per prompt + 1.66e-5 s per native token, fitted to w1–w5)
  plus under 0.1 GPU-h for the repeat run and model loads. The lease expects about 1.5 h. If the wave has not finished
  after 3 h wall-clock, it is stopped and recorded.
- **GPU-hours:** wall-clock × 1 GPU per teacher process.
- **Records:** results go to `m3b-lux-h-targets-2026-09-29.md` and the gist `02-decision-2-research-data.md`.
- **C1:** H7 / H8 passed the C1 registry gate at build. This wave adds no source.
