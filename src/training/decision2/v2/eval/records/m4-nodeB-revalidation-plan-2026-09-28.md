# Node-B comparator re-validation plan (kernel-equipped image) — PLAN, not run (2026-09-28)

Trigger: coordinator note 21:00 ("node B image gap").

## Finding (CPU probe, 2026-09-28)

| Node | Image | `causal_conv1d` | FLA | torch / Transformers |
| --- | --- | --- | --- | --- |
| A | `decision20-train-fast:host2` (`f83b1d10…`) | 1.7.0 | 0.5.2 | 2.12.0+git6bbd260 / 5.17.0 |
| B | `decision20-train-fast:latest` (`dbe5f32b…`) | 1.7.0 | 0.5.2 | same |
| B | `decision20-lux-runtime:latest` (`ce895822…`) | **missing** (fallback path) | 0.5.2 | same |

- **Which node-B collections used the kernel-less image.** Every eval-track node-B
  collection used `ce895822`:
  - `m2/n1-lux1-nodeB-frozen`, 413 GPU-s;
  - `m2/n2-autojev27-nodeB`, the 27B comparator, 1,189 GPU-s;
  - `m2/q3-eikos27b-nodeB`, 1,336 GPU-s;
  - `m2/q7-jebadiah27b-nodeB`, 936 GPU-s;
  - the M1 node-B Lux1 run `r4`.
- **The mismatch.** The 27B track's own M2 formal runs (`27b/m2-F0-formal`,
  `m2-C1-formal`) used the kernel image `dbe5f32b`. Their comparisons against AutoJev
  therefore mix two runtimes.
- **Correction to M1/M2.** The M1/M2 records said the node-B image was byte-identical
  to node A's. That is wrong for `causal_conv1d`. The Lux1 node-A vs node-B difference
  (43 of 8,778 answers), which was attributed to hardware, may instead come from this
  kernel difference. The re-validation tests that.

## Plan (one-shot runs, preregistered before launch)

1. **Image.** Node B `decision20-train-fast:latest` (`dbe5f32b…`). Record its id, confirm
   in a sidecar that `causal_conv1d` imports and that FLA is on the path, and use a
   persisted Triton autotune cache per run. Code from an exact mirror of the integration
   head. Adapters are the same modules and revisions as the original runs (hash-checked).
2. **Smoke.** A 20-item gold-free smoke for each adapter on the new image checks answer
   shapes. This is a precaution: the image changes, the adapters do not.
3. **Re-collect once each on the formal panels** (typed FINAL, CSS15, public 231):
   AutoJev-27B, Eikos-27B (BF16) and Jebadiah-27B. Add Lux1 (9B) as the cross-node
   diagnostic.
4. **Score on node A.** Gold stays on node A, so node-B predictions are copied there.
   Report v3, public 231, and answer agreement and max probability drift against the
   old node-B run and, for Lux1, against node A D1. Paired bootstrap new vs old.
5. **Decision rule.**
   - The new runs become the node-B comparators for every later 27B formal comparison.
     The change is disclosed. The old runs stay as records.
   - If AutoJev's v3 moves, the 27B release threshold (≥ 90% of the best measured peer)
     is recomputed from the new value.
   - If Lux1 on the kernel image matches node A bit for bit, the same-node rule is
     reworded as a same-image rule.
6. **The 27B track** confirms that every 27B candidate and comparator it compares ran on
   `dbe5f32b`. Its M2 formal runs already do.

## Cost and slot

About 1.3 GPU-hours on one node-B GPU:

- the original collections sum to about 3,870 GPU-s;
- plus four smoke runs (~0.1 GPU-h);
- plus warm-up for the new autotune caches.

Wall time is about 1.5 h on one GPU, or about 45 min on two. **Requested slot:** node B
GPU5 (or GPU6) from the ~27B track, before the next 27B formal comparison. Borrowing both
halves the wall time.
