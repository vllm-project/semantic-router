# Decoder M16 — data lock (2026-10-01)

Staged 17:40–17:44Z by `ops/m16/m16-prep.sh` (mirror `dfb86c08d`) on node A and node B, CPU only; nothing ran on a
GPU. The prereg (`9923e8b51`) was committed at 17:27Z; its header's "≈17:30Z" is a slip (no M16 action had run).
Nothing on nodes E / F was modified (read-only copies of finished M12 / M13 soups; M15 not touched).

## Lineage (`m16_interp.py lineage`, per pair; every pair passes, **no pair skipped**)

| Tier | Release R (model_sha256) | Arm X (model_sha256) | Shared source | Tensors (backbone / head) |
| --- | --- | --- | --- | --- |
| 0.8B | DEV2.0-0.8B C0 `3f02f0e5…` | `08b-RAUP` `0bf0401d…`, `08b-RASD` `02c17086…`, `08b-RA` `f25beedd…` | Eos 1.0@`363c4a5e` (full fine-tune) | 320 / 10 |
| 2B | DEV2.0-2B C0 `32872f29…` | `2b-RAUP` `37c85fff…`, `2b-RASD` `341c2bd2…`, `2b-RA` `3d4fde06…` | Sol 1.0@`ce0c018a` (full fine-tune) | 320 / 10 |
| 4B | LH soup `5fa2a699…` | `4b-LHA10UP` `9e80e765…`, `4b-LHA10SD` `255021e0…` | merged LoRA on Qwen3.5-4B-Base@`1001bb4d` | 426 / 10 |

- Each check: equal architecture, prompt version, head dimension, checkpoint format, maximum options and
  `full_training_source`; identical tokenizer files; identical backbone and head tensor names and shapes. The 2B
  release stores its two shards with a different split of dtypes than the arms (same tensor names / shapes); builds
  read tensors by name and write FP32 in the arm's layout.
- The copied M12 / M13 soups (node A `08b-RASD`, `08b-RA` from node E; node B `2b-RA` from node E, `2b-RASD`,
  `4b-LHA10SD` from node F) have content manifests equal on both sides, and their model_sha256 above equals the
  source's soup build log.

## MLX-DEV panels (`m16_mlxpanel.py`, node B; equal to M15's lock)

| Tier | Excluding (M12 arm TRAIN) | Rows / groups | Dropped groups | Panel SHA-256 | Index SHA-256 |
| --- | --- | --- | --- | --- | --- |
| 0.8B | `12bd63d8…` | 9,386 / 3,802 | 0 | `100ae4e7…` (whole) | `6ffa4b84…` |
| 2B | `08140409…` | 7,742 / 3,395 | 407 | `4f153722…` | `99283f94…` |
| 4B | `d41cdd1a…` | 9,386 / 3,802 | 0 | `100ae4e7…` (whole) | `6ffa4b84…` |

The 0.8B panel was relayed to node A through node E (tree hash `3fe3906a…`, 4 files).

## Reference readouts (M14's same-node readouts, staged; sha256 over the sorted per-panel prediction hashes)

| Point | Node | Panels | Hash |
| --- | --- | --- | --- |
| `08b-C0-a` | A | 8 | `2633fa12…` |
| `2b-C0-b` | B | 8 | `677952bd…` |
| `4b-LH-b` | B | 8 | `b4b52ef6…` |
| `08b-RAUP-a100` (report only) | A | 8 | `f42f8bc4…` |
| `2b-RAUP-a100` (report only) | B | 8 | `856833ae…` |
| `4b-LHA10UP-a100` (report only) | B | 8 | `57886fa2…` |

Read caches: node A `08b-read`, node B `2b-read` / `4b-read`, `cp -a` copies of M14's. Leases: node A GPU3–5 and node
B GPU2–4 owner files now `track=dec-m16` (the previous files kept as `owner.prev-<UTC>`).
