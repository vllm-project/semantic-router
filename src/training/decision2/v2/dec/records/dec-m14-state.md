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
