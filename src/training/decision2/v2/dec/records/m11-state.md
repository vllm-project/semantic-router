# Decoder M11 — state (polled ≤ 30 min)

Prereg `b8c3caf22`; ops `cef1a1b66`; data lock `7c3aed8c8`; amendment 1 (node F layout). Branch
`xunzhuo/decision-2-training-dec-m11`.

## 2026-10-01 07:15Z — launch

- Stage 1 chains launched from the mirror of the amendment-1 commit:
  - node F GPU2 / 3 / 6: `2b-LH` s1–s3 then `08b-LH` s1–s3 (GPU2 pre-warms); GPU7: post chains `2b-LH`, `08b-LH`;
  - node E GPU0 / 1 / 2: `2b-NT` s1–s3 then `08b-NT` s1–s3 (GPU0 pre-warms); GPU3: refs (C0 and BASE per tier), then
    post chains `2b-NT`, `08b-NT`.
- Node F GPU4–5 are held by a foreign workload (amendment 1); untouched.
- Stage 2 (4B LH + IB1) waits for a release-safe IB1-r3 on origin.

## 2026-10-01 07:17Z — poll

- Pre-warm preflights PASS: `2b-NT-s1` (E GPU0, 07:14:14Z), `2b-LH-s1` (F GPU2, 07:14:42Z); seeds 2–3 of both arms
  started on the markers at 07:15Z. All six 2B seeds train (≈2 s / update NT, ≈3.4 s / update LH; ≈877 updates).
- E GPU3: `2b-C0-e` readouts in progress (refs chain), then `2b-BASE-e`, `08b-C0-e`, `08b-BASE-e`.
- Node A: mirror `f45c2f481`; tier probe gold built (2B 2,989 items `0c0781ee…`; 0.8B = M10's 3,089 `5c674e35…`).
- Next: when the 2B seeds finish (≈08:10–08:30Z), the 0.8B seeds start (pre-warm on E GPU0 / F GPU2 first); post
  chains build soups and read; then `m11-score.sh pull / points / readout / rules` on node A per tier.

## 2026-10-01 07:20Z — poll and hand-off

- All six 2B seeds in their full runs; no FAILED / STOPPED marker. E GPU3: `2b-C0-e` readouts done (07:18Z),
  `2b-BASE` zero-step done, `2b-BASE-e` reading.
- C0 path check: `2b-C0-e` against node B's stored M8s `2b-I` readouts of the same weights (`m11-lines.sh parity`,
  inputs `m11/inputs/nodeB-I/<t>-I`, predictions only): dev, css-pilot, ht-dev2, score5t-dev, hs1-dev all 0 decisions
  differ, max drift 0.0. Run the same for `08b-C0-e` (`…/nodeB-I/08b-I`) and for the node-F C0 reads.
- Formal masters relayed from node B to E and F under `formal/m11/masters/` (`cache-frozen-2b{,-mlx}`,
  `cache-frozen-08b{,-mlx}`, with manifests); all four match their frozen manifests on both nodes (manifests exclude
  `./.cache`, as m8s-formal's `tree_manifest`).
- Hand-off (next session), in order:
  1. Polls: 0.8B seeds start after the 2B seeds (pre-warm on E GPU0 / F GPU2); post chains soup and read.
  2. Node A per tier: `m11-score.sh pull`, `points`, `readout <t> <REF> …`, `rules <t> <t>-LH=<t>-C0-f
     <t>-NT=<t>-C0-e -- <t>-LH:<t>-NT`; report BASE probe ceilings and C0 E-vs-F parity.
  3. Formal wrapper (not written yet; shared-module change, commit with a test before use): add `M6_SMALL_NODE=E|F`
     to `ops/m6/m6-formal-lib.sh` (2b / 08b on image dbe5f32b, `--isolate`, plain model dirs `/data/dev2/models`,
     masters from `M6_SMALL_MASTER_DIR` = `formal/m11/masters`, `MASTER_SHA=frozen`), and `ops/m11/m11-formal.sh`
     modeled on `m10-formal.sh` with a tier argument. The library's node-F GPU list includes GPU4–5: the M11 wrapper
     must refuse them (amendment 1). Check that the library's `tree_manifest` (no `./.cache` exclusion) agrees with
     these manifests before the first copy. C0 first, path-checked against the bars (2B node-A
     `dev2-2b-t1-derived`; 0.8B node-B `m8s-ref-08b-I`); 0.8B scoring via the `m8s-score08b.sh` pairing.
  4. Stage 2 only when IB1-r3 is release-safe on origin (data lock first).
