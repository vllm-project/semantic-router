# Reasoning wave 1 (9B and 2B) — preregistration

Written 2026-10-04 before any 9B / 2B training. The 4B wave-1 recipe
([`rsn-w1-4b-prereg-2026-10-04.md`](rsn-w1-4b-prereg-2026-10-04.md)) transferred unchanged, started before the 4B
dev results exist (to use node D once the teacher finishes); its data hashes follow in a data-lock amendment.

## Start points (full continuation, `train_dec --init decision2 --train-mode full`)

- 9B: `vllm-sr/Decision-2.0-Lux-9B` (`main` `78bf3c03`, weights M10 `KIB4-a40`, Index 0.2.1 46.26); FP32 training
  checkpoint copied to node D `runs/reasoning/starts/9b-KIB4-a40`. Image `f83b1d10` (the 9B M10 image).
- 2B: `vllm-sr/Decision-2.0-Sol-2B` (`main` `64235bef`, weights M15 `2b-RASDML`, Index 0.2.1 29.53); FP32 training
  checkpoint (the two-seed soup) copied to node D `runs/reasoning/starts/2b-RASDML`. Image `dbe5f32b`.

## Data (`v2.reasoning.build_v1`, one build per size)

- The same program-verified problems as the 4B build (same generators and seed namespaces, so the same instances).
- Teacher graph problems from the complete teacher run (all 25,441 pool rows).
- Replay: 30,000 rows of the size's own released TRAIN (9B: M10 `KIB4` TRAIN `2e72bcfd…`; 2B: M15 `2b-RASDML` TRAIN
  `97157068…`), fixed-seed sample excluding pool ids, labelled by the size's released model (T = 1).
- Same node caps, weights, view forms, dev split, SELECT / CAL (`sel700-cal698`) and 13-gram decontamination.

## Arms (two seeds each, 20261004 / 20261005; node D, one seed per GPU)

`R9-TF`, `R9-F0` (GPU0–3) and `R2-TF`, `R2-F0` (GPU4–7). The rewired control is not repeated: its question (graph
structure vs any node text) is answered at 4B. Trainer settings identical to 4B (`--backbone-lr 5e-6 --head-lr 5e-5`,
KL 1.0 on replay and teacher-problem finals, 1 epoch, token batching 32,768 / 64 rows / 64-row updates).

## Candidates, selection, formal reads and gate

As at 4B: per arm the two-seed soup of final checkpoints; α ∈ {0.5, 0.75, 1.0} towards the release; the TF point with
the best RP-DEV final macro accuracy whose SELECT family-macro accuracy is at least the release's minus 0.01 (ties to
the larger α); its F0 point at the same α as the control. At most two formal Index reads per size (TF candidate, F0
control), each a BF16 release copy restaged on the tier's IX1 package, paired-bootstrapped against the release's
reference run (9B `M10-KIB4-a40-bf16`, 2B `IS-2b-RASDML-bf16`). Release gate as at 4B.

## Stop rules

As at 4B. In addition: if the 4B result shows no tree effect (TF not above F0 on RP-DEV finals), the 9B / 2B TF
points are still read on the dev panel, but a formal read needs the same TF-over-F0 dev evidence at that size.
