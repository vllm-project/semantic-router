# Decoder M14 — state (polled ≤ 30 min)

Prereg [`dec-m14-prereg-2026-10-01.md`](dec-m14-prereg-2026-10-01.md). Branch `xunzhuo/decision-2-training-dec-m14`
(from integration `4cd24bcd9`). GPUs: node A GPU3–5, node B GPU2–4 only. Cap 80 GPU-h.

## 2026-10-01 15:25Z — prereg

- Trainer `--example-weights` (`11ec9c5e4`, tests in `test_dec.py`); prereg and `ops/m14/` committed before any M14
  data build or GPU job. Arms `08b-RAUP` (released rows ×1.5), `2b-RAUP` (Choice / Noul ×1.5, Score ×2), `4b-LHA10UP`
  (×1.5), IB rows ×1, window-normalized loss weights on M12's locked TRAIN files.
- Lease check: node A GPU3–5 have no live owner (MoE entries released 11:23–11:40Z); node B GPU2 (27B M4b, idle since
  09-29), GPU3 / GPU4 (decoder M8, idle since 09-30) idle, 0% use, no VRAM in use.
- Next: mirror, prep on A and B (pulls from E / F), data lock, READY, launch.

## 2026-10-01 15:12Z — data lock, launch

- Mirror `e48a66f46` on A and B. Prep 15:04–15:05Z (CPU): M12 TRAIN + ids, teachers, C0 packages, LH soup and the
  `ib-dev` / `m10-probes` prompts pulled from E / F and hash- or manifest-checked; weights byte-identical on both
  nodes (`99b77df5…`, `e1ae6144…`, `3f741da8…`). Data lock [`dec-m14-datalock-2026-10-01.md`](dec-m14-datalock-2026-10-01.md)
  `940fb56a2` pushed; `READY-m14.json` written and checked on A and B. Trainer tests (39) pass in the image.
- Leases `track=dec-m14` on node A GPU3–5 and node B GPU2–4 (previous idle owners kept as `owner.prev-*`).
- Launched 15:08Z from mirror `e48a66f46`: node A GPU3 `08b-RAUP-s1` (pre-warm), GPU4 `08b-RAUP-s2` (waits),
  GPU5 refs `08b-C0-a` then post `08b-RAUP`; node B GPU2 `2b-RAUP-s1` (pre-warm) then `4b-LHA10UP-s1` (pre-warm),
  GPU3 `2b-RAUP-s2` then `4b-LHA10UP-s2`, then post `4b-LHA10UP`; GPU4 refs `2b-C0-b`, `4b-LH-b`, then post `2b-RAUP`.
- 15:11Z: both pre-warm preflights running (2B zero-step done); reference reads at `ht-dev2` on both nodes.
