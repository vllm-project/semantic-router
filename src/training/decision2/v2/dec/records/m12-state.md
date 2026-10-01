# Decoder M12 — state

## 2026-10-01 11:15Z — prereg, data lock, launch

- Prereg [`dec-m12-prereg-2026-10-01.md`](dec-m12-prereg-2026-10-01.md) + ops `1f7fc94be` (before any M12 data or GPU
  job); data lock [`dec-m12-datalock-2026-10-01.md`](dec-m12-datalock-2026-10-01.md) `d506595c2` (E / F builds
  byte-identical; `READY-m12.json` written on both nodes after the push).
- Arms: `4b-LHA` (LH + 25% IB, the same IB rows as M11's `4b-LHB`), `4b-LHA10` (+10%), `2b-RA` (S2T recipe + 25%),
  `08b-RA` (E8F recipe + the IB pool three times, +22.1%); optional `4b-LHAx` / `4b-LHA10x` built, report-only.
- Probe golds on node A: 2B 2,189 items (`1c6c8b76…`), 0.8B and 4B 2,203 (`f712fc53…`); GSM8K thin (114 / 100 items).
- Launched 11:10Z from mirror `1f7fc94be`: node F GPU2 `4b-LHA-s1` (pre-warm), GPU3 `4b-LHA-s2`, GPU6 `4b-LHA10-s1`,
  GPU7 `4b-LHA10-s2`; node E GPU0 `08b-RA-s1` (pre-warm), GPU1 `08b-RA-s2`, GPU2 `2b-RA-s1` (pre-warm), GPU3
  `2b-RA-s2`. Post chains (wait on the GPU flock): F GPU3 `4b-LHA`, F GPU7 `4b-LHA10`, E GPU1 `08b-RA`, E GPU3 `2b-RA`.
- Reference readouts copied from M11 (`4b-LH-f`, `2b-C0-e`, `08b-C0-e`); the C0 IB DEV panel is read by the post chains.
