# Decoder M15 — state (prereg `dec-m15-prereg-2026-10-01.md`)

## 2026-10-01 16:50Z

- Timeline: prereg `c62a78619` (committed ≈16:22Z; its header's "≈16:40Z" is a slip, nothing M15 had run); IX1
  diagnostic entry `cdaae0f32`; ops `6abb5358e` (13 CPU tests, with M13's); data lock `30ec83646`
  (`dec-m15-data-lock-2026-10-01.md`: five arms byte-identical on E and F, multilingual shares equal the released
  shares; MLX-DEV-M15 panels 4B / 0.8B whole, 2B 7,742 rows).
- **Part A** (node D GPU7, harness lease `eval-ix1`): restaged onto `13d42143` (12 model files replaced); the first
  launch was refused by the harness's lease check (GPU7's owner file was an older one-line form; no container ran)
  and its write-once `launcher-parity.json` stub was moved to the private `void/`; the owner file was rewritten in the
  harness's multi-line form (still `track=eval-ix1`) and the run launched once: **parity gate PASS (86 requests)**
  16:33Z; panel-8 shards running one after another. Results stay private.
- **Part B**: chains launched 16:45Z from mirror `6abb5358e`; pre-warm seeds `08b-RA10SDML` s1 (E GPU0), `2b-RASDML`
  s1 (F GPU2), `4b-LHA10SDML` s1 (F GPU6); the other seeds wait for their tier's marker. Post chains queued: E GPU3
  `08b-RASDML` (+ `08b-C0-e` and `08b-RA-m12` MLX-DEV), E GPU1 `08b-RA10SDML`, F GPU7 `4b-LHA10SDML` (+ `4b-LH-f` and
  `4b-LHA10SD-m13`), F GPU6 `2b-RASDML` (+ `2b-C0-f`), F GPU3 `2b-RA10SDML`.
- M14 untouched (no M14 container on E / F GPUs). GPU-h so far: Part A parity ≈ 0.06.
