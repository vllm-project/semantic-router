# M3b — own-Lux targets on the XL rows that lack them (2026-09-28)

Per [prereg §3](m3b-prereg-2026-09-28.md) with [amendment 1](m3b-prereg-amendment-1-2026-09-28.md). Amendment 2 adds no
teacher targets: H7 / H8 rows of the r2 recipes train on gold labels.

- **Teacher run:** own Lux1 `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30a…` on node B GPU7, one wave per process.
  - Launcher: the Milestone 2 `teach.sh` (`49e50c6e…`), started by the queue `m3b-luxxl-queue.sh` (`e2ef3193…`). The
    queue starts a wave only when its prompt file has the expected SHA-256.
  - Image `decision20-lux-runtime:latest` = `sha256:ce895822…`, source mirror `5f5cd80a…`.
  - Collector: `inference.run --backend lux --over-budget-invalid`.
  - Triton cache: the node-B Lux cache of own-Lux RP-v2 waves 1–4. At the w1, w2 and w3 publications it held 1,762
    files with tree digest `299151ab…`, and no file in it was newer than the XL prompt files.
- **Conversion and upload:** on node A from the exact mirror of `81f673aed`. Private dataset
  `llm-semantic-router/decision-2.0-training-data`, tree `m3/teachers/lux1/xl/`, with `README.md`
  ([source](hf-lux1-xl-readme.md)) and `coverage.json`.

## Waves

| Wave | Prompts | Answered | Over budget / invalid | Rows C / N / S | Argmax vs gold C / N / S | Wall | GPU-h | HF revision |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |
| w1 | 60,000 | 60,000 | 0 / 0 | 9,674 / 8,268 / 42,058 | 0.785 / 0.808 / 0.428 | 2,547 s | 0.708 | `b1df84c48c35e965d5ad4276b125e080f325ffb4` |
| w2 | 60,000 | 60,000 | 0 / 0 | 47,923 / 10,462 / 1,615 | 0.913 / 0.894 / 0.871 | 2,298 s | 0.638 | `30e0a1f79ee9b81b726c6c58d4458361b2f986c6` |
| w3 | 60,000 | 60,000 | 0 / 0 | 23,971 / 23,334 / 12,695 | 0.767 / 0.774 / 0.644 | 2,919 s | 0.811 | `530b0bce352b2edbe32436c2a0acf3561640a990` |
| w4 | 60,000 | pending | pending | pending | pending | pending | pending | pending |
| w5 | 12,215 | pending | pending | pending | pending | pending | pending | pending |

| Wave | Targets SHA-256 | Attestation SHA-256 | Report SHA-256 | Teacher output SHA-256 |
| --- | --- | --- | --- | --- |
| w1 | `4a852b74794801e061facf42911f2a26434af8083a773268a481a1a481aebc7f` | `e4fc42f8dd3ee41b0787f48c25ae6ecf52d575808bc7d7e4e2458433091682fb` | `a7a1042b69f2f4a15e73b06fbf69fcebd75eadee38fb3ad50e748d112557477d` | `2d788516cfaf1fbb4b52d16daedb60a7692efc861cee5305fc41bc18caae54f0` |
| w2 | `a659c3e59774387f6f9f008a55182c374e321685f57ea3dcc0fe3c5fe5a1f2f2` | `a75c7dea9a7531eb57d588fcc2ee191ea3a6c99bd8a808d9a570342c455bf48a` | `fda194e4705c23aa2dc0e5ef453f06454b4c6584592a6754c52e53288d49123e` | `6141df2e60415d3c76028eca150a30a1a4697dfee9abbcfea2b6e49570db8ce8` |
| w3 | `6d141c96b54a9548b4572c04c4fbd30e413b168a43d583f3086000413a207c73` | `68f11e6ccd51ad697d16595239047d38f5c4e0bbf3a07fe30e862707765fbb03` | `564ed94522ec16ad456e99ab757e7e45ebc784ed0a82e03fa860a644f4a936d2` | `711adb23e1aa36fc466c7e30b4a75f0342e68ab0c1b90077da97805fd2b9265d` |

`README.md` `fd2379c2…` and `coverage.json` `cc54efde…` were uploaded with w1.

- **What each wave holds:** ids are sorted, XL-full first.
  - w1: the v1 arms and A7h / A7k / A7q / A7r / A7s. Its Score rows are mostly A7q (OASST1 quality, 22,401) and A7s
    (12,860), hence Score 0.43.
  - w2: A7i / A7m / A7o / A7p / A7r, mostly Choice.
  - w3: A7g plus H1 / G2 / G6 / H6 / H3 / H5.
  - w4: the last 32,554 XL-full rows and the first 27,446 XL-short-only rows.
  - w5: the remaining XL-short-only rows (H3 / H6 / H5 / E11).
- **Native tokens per wave:** 22.5M, 22.3M, 44.9M, 27.9M and 6.4M.

## Checks (every wave, automated)

1. **Guard receipt of the prompt file:** PASS, with 0 shared ids, input hashes and prompt digests.
   - Protected: SELECT700, CAL700, CAL698, every v1 / v2 / A7 AHO and SHO slice, and every gold-free eval panel.
   - Shared normalized states, informational only: w1 1, w2 1,118, w3 73, w4 283, w5 33.
   - `protected_files` counts 27 distinct basenames. The many `aho.jsonl` files collapse under one key; the check still
     used every fingerprint.
2. **Prompt file:** the same SHA-256 on node A and node B.
3. **Teacher run** (`v2.data.m3.luxxl provenance`):
   - the launcher line has rc 0;
   - the collector summary shows `collected_now = input_items =` prompts, `previously_completed = 0`, the pinned
     identity, an attested revision, the validated runtime and no runtime differences;
   - the image id is `ce895822…`.
4. **Copy:** node B → local → node A (gzip), with SHA-256 checked locally and on node A. w1 and w2 reused node-A
   copies whose SHA-256 matched node B; w3 was transferred by the script.
5. **Conversion:** `v2.data.m2.targets --prompts --attestation --provenance`.
   - Per row: model identity, attested revision and validated runtime.
   - Prompt digest checked on the prompt file as sent; each prompt equals the training row's native prompt.
   - Teacher-output ids = prompt ids = row ids, exactly.
   - Every prompt without a target is listed in `no_target` with its reason (none so far).
6. **Readback:** the new revision is downloaded and every uploaded file's SHA-256 compared. A path-leak check and the
   private-value guard run on the report before the upload.

The attestation has one line per prompt with these fields:

- `input_sha256`, `source_input_sha256`;
- `model_id`, `model_revision`, `revision_attested`, `adapter_version`, `backend`, `model_config_sha256`,
  `runtime_matches_validated`;
- `target`, plus `no_target_reason` / `native_error` for a prompt without a target;
- `node`, `gpu`, `image_id`.

I also checked w2 independently. All 60,000 attestation lines have `target = true`, the attested revision `bd45a30a`,
node B GPU7 and image `ce895822`. The ids equal the teacher output and the targets file. No receipt carries a
`native_error`.

**Image disclosure:** image `ce895822…` has no causal-conv1d kernel; Transformers uses its reference PyTorch
convolution. The eval track found that this image changes 43 of 8,778 Lux1 argmax answers against the kernel-equipped
image (note 23:40 UTC+8). Own-Lux RP-v2 waves 1–4 used the same image, so the XL targets stay consistent with every
earlier own-Lux target. Every report carries this note.

## Coverage after each wave (own-Lux targets / recipe rows)

| Recipe | Rows | Before | After w1 | After w2 | After w3 | After w4 | After w5 | Never covered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mx-xl-full` | 340,698 | 128,144 | 188,144 | 248,144 | 308,144 | **340,698** | 340,698 | 0 |
| `mx-xl-short` | 355,855 | 123,876 | 181,049 | 238,873 | 286,569 | 343,640 | **355,855** | 0 |
| `cx-xl-a7v1-full` | 191,605 | 12,413 | 72,413 | 132,413 | 153,527 | 170,089 | 170,089 | 21,516 |
| `cx-xl-a7v1-short` | 167,071 | 11,172 | 68,345 | 126,169 | 135,956 | 157,460 | 157,460 | 9,611 |
| `cx-xl-v2v1-full` | 296,514 | 127,186 | 146,340 | 146,340 | 185,226 | 219,152 | 225,542 | 70,972 |
| `cx-xl-v2v1-short` | 282,695 | 123,974 | 142,199 | 142,199 | 180,108 | 215,220 | 227,074 | 55,621 |

Computed by `v2.data.m3.luxxl coverage` from the published `*.lux1.missing.jsonl` lists and the wave prompt files
(`coverage.json`). The per-wave gains are these:

- `mx-xl-full`: 60,000 / 60,000 / 60,000 / 32,554 / 0.
- `mx-xl-short`: 57,173 / 57,824 / 47,696 / 57,071 / 12,215.

**Deviation from §3:** the waves cover every row of XL-full ∪ XL-short that lacks a target, but not the controls' own
rows.

- 100,957 rows (64.9M native tokens) of the four `cx-xl-*` controls are in neither recipe, and no wave covers them. The
  previous worker set them aside when building the waves.
- By pool: H5 27,814, G6 11,439, A7g 10,524, E11 10,158, A7o 9,761, H1 9,347, H6 8,037, H3 7,264, G2 5,382, V1:A3
  692, A7r 244, A7i 169, A7q 87, V1:A4v2h 39.
- Covering them takes two more waves, about 1.2–1.6 GPU-h on node B GPU7. That estimate uses the w1–w3 rates of 23
  prompts/s and 11.6k tokens/s. Until then, control rows without a target train on gold labels.

## Automation (w2–w5)

`v2/data/m3/luxxl_publish.sh` (commit `81f673aed`) runs on the local machine from the committed copy
`/tmp/m3b/luxxl_publish.81f673aed….sh` (local pid 646364, its own session, started 16:26 UTC). It reaches the nodes
only through the alias wrapper. For each wave, in order, it does the following:

1. It polls node B's `teach.log` for `LUX_XL_W<k>_DONE` every 120 s, for at most 6 h.
2. It runs steps 2–6 of the checks above.
3. It uploads `w<k>.{targets.jsonl,attestation.jsonl,report.json}` and reads the revision back.
4. It writes `publish/w<k>/published.json` on node A.

Events are JSON lines in `/tmp/m3b/luxxl-publish.events.jsonl` (local) and in
`/data/dev2/runs/data/m3b/lux-xl/publish.events.jsonl` (node A); the driver log is
`/tmp/m3b/luxxl-publish.driver.log`.

- **On failure:** any failure stops the script with a named event. The copy is retried 3 times; nothing else is.
- **On a re-run:** a wave with `published.json` is skipped. A wave with an earlier upload directory or upload log stops
  the script, so no wave is ever uploaded twice by it.
- **To check progress:**
  - `tail /tmp/m3b/luxxl-publish.events.jsonl`;
  - `pgrep -f luxxl_publish`;
  - on node A, `cat /data/dev2/runs/data/m3b/lux-xl/publish/w<k>/{published.json,readback.txt}`.

**Earlier upload at the same path:** at 16:13 UTC another session committed `38c2db3c3a1fe2b288b8c36e9c82a6456d77fba1`
("waves 1-2"), which put w1 / w2 targets and short reports without attestation under `m3/teachers/lux1/xl/`.

- Its targets are byte-identical to the ones above (`4a852b74…`, `a659c3e5…`).
- `b1df84c4` and `30e0a1f7` replaced its two reports with the full ones and added the attestations, `README.md` and
  `coverage.json`.
- Use `30e0a1f7` or later.

## GPU-hours (node B GPU7)

- **Finished:** w1 0.708, w2 0.638 and w3 0.811, 2.157 GPU-h so far.
- **Pending:** w4 started at 16:38 UTC and w5 follows. Both are expected to be published by about 17:45 UTC.

Conversion, upload and readback run on node A and the local machine, CPU only.
