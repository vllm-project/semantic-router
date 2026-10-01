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

## 2026-10-01 12:15Z — pre-warms passed; 2B closed (no finalist)

- Pre-warm preflights passed on all three tiers (warm-4b-f, warm-2b-e, warm-08b-e); all 8 seeds started by 11:18Z.
- `2b-RA`: seeds DONE 11:41Z / 11:45Z; soup + 8 panels 11:56Z. Rules (run once, 12:12Z): **not eligible** — Score
  typed floor 227 < 257 − 0.03·400. Otherwise: T .641 vs C0 .610; HT-DEV v2 −.004 [−.021, .012] TIE; retention
  −.010 [−.038, .017]; Score5t no flags; false-yes .524 vs .625; IB DEV +.174 [.150, .200], transfer +.151 [.123, .183].
  The 2B Score head again loses typed items under breadth (milder than M11's collapse, still below the floor).
- 4B (4 seeds) and 0.8B (2 seeds) training.

## 2026-10-01 13:05Z — M12 complete (no finalist)

- 0.8B `08b-RA` and 4B `4b-LHA` / `4b-LHA10` finished 12:35Z–12:55Z; rules run once at 12:58Z. **No finalist at any
  tier**: `08b-RA` fails only the `attribute_gate` family floor (HT-DEV v2 GAIN +.049, retention +.044); `4b-LHA` fails
  choice / Noul floors and retention; `4b-LHA10` fails the Score / `set_reconciliation` floors. The optional
  transfer-only 4B arm is not trained (no passing 4B arm); no formal, hand-offs or Index requests.
- Report-only LHA vs M11 `4b-LHB` contrast read 13:00Z (M11 readouts copied on node F as `4b-LHB-m11`).
- GPU-h 10.01 of 100 (E 4.44, F 5.57); no M12 job running; all M12 leases idle. Results:
  [`dec-m12-results-2026-10-01.md`](dec-m12-results-2026-10-01.md).
