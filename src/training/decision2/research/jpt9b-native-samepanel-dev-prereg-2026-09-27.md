# JPT-9B native development-panel comparison: prospective protocol

Status at freeze: **not yet run on the full panels**. This experiment is an
external, development-only 9B comparison. It cannot establish a JevArena v3
release score, official JevBench rank, or Decision Index standing. Protected
typed FINAL and CSS 15-task labels and prompts are outside this run.

## Frozen roster and inputs

| Role | Identity | Output contract |
| --- | --- | --- |
| External comparator | `kirp/jpt-9b` at `7114b0c3d9bea6b82dfa2d0691e8d5562cd26d4e` | Native `llm2jev` `LLM2Jev(AutoProcessor, HF, temperature=1.087)`; original state and Choice/Noul/Score questions, no relabeling or task conversion. |
| Own 1.0 reference | `llm-semantic-router/Decision-1.0-Lux-9B` at `bd45a30aee8c84032791c245c70f86dee5389cc8` | Existing same-panel native receipts, verified against identical prompt/score protocol before comparison. |
| Own research candidate | Lux-start structured8360 low-rate checkpoint `0000224`, selected before this run | Existing same-panel native receipts only; this candidate remains a private development control, not a release. |

The JPT `llm2jev` source is pinned at
`2b252d504972764211ef172c1155ac0fedc9c3de` with a clean checkout.
The model must have revision-attested local bytes. The prior 32-row gold-free
smoke passed twice with unchanged categorical answers and all three types;
its prompt SHA-256 is
`ce16f6107b8a8da6d0e5ef501cc8360da07a7232ecd7289d5a437d1d77767b67`.
The adapter is `jpt-9b-llm2jev-hf-v1`; its local source SHA-256 is
`1c1bfe9ced064e5a5175408dd76f317a8f7d88c810a8a765a37059053cdf90b4`.
The imported prompt loader SHA-256 is
`b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce`.
The pinned ROCm runtime image is
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.

| Development panel | Items | Gold-free prompt SHA-256 | Scorer SHA-256 |
| --- | ---: | --- | --- |
| Typed DEV | 1,600 | `a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a` | `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc` (`benchmark.score`) |
| CSS three-task pilot | 1,430 | `598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda` | `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca` (`transfer.score`) |
| JevBench public subset | 231 | `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd` | `aec840b6497beeae8b63e9270670c22506265c086a84e889bc82f694146518cf` (`jev_arena.jevbench_public`) |

The panel gold or public target file hashes are checked before scoring; gold
is not mounted into inference. Typed DEV has 400 independent four-row groups.
CSS is reported per task and by median macro-F1 across three tasks. Public
JevBench has easy 48, standard 72, hard 111. Missing, invalid and over-budget
responses count as failures. No prompt truncation, dropped options, output
substitution, or post-hoc temperature choice is allowed.

## Frozen execution and stop rules

1. Before each GPU launch, confirm the selected device has no GPU PID and no
   occupied VRAM; use only a device available to this experiment. Copy the
   exact committed adapter/loader/scorer files into an isolated remote mirror,
   and verify SHA-256. Preserve prior experiments and make each output path
   create-only.
2. Re-run the fixed gold-free 32-row smoke with the committed adapter. Stop if
   source/model revision verification fails, any row is absent, any answer
   lacks a typed response, any category differs from the earlier frozen smoke,
   any numeric value is non-finite, or the collector raises an error. No
   development gold is inspected at this gate.
3. Run the three full gold-free panels, sequentially on one GPU or on distinct
   verified idle GPUs. Each output must contain the exact expected unique IDs,
   per-input SHA-256 and native model receipt. Stop a panel on process failure,
   out-of-memory, repeated invalid native output, or a 90-minute wall timeout.
   A failed/partial file is retained and reported, never silently resumed as a
   complete score. The parent task reserved GPU 2 for this arm and capped the
   experiment at **2 GPU-hours total**, including smoke and all three panels;
   stop at that limit and record load/inference wall times and GPU-hours.
4. Hash and seal every complete prediction file **before** reading any panel
   gold. Then score once with the frozen scripts. Report overall and per-type
   typed accuracy, CSS task macro-F1/median, public tier accuracy, invalid
   rates, and calibration where the native output supports it. Compare only
   against 1.0/candidate receipts whose prompt and scorer versions match.

One run only. Development diagnostics may inform subsequent research, but
neither these labels nor historical Decision Index numbers may be portrayed
as unseen, independent release evidence.

## Planned receipt fields

The private run manifest records host-side start/end UTC timestamps, chosen
device, image digest, model file hashes, llm2jev commit, adapter and scorer
hashes, input/prediction/report SHA-256, complete/valid/invalid counts,
runtime versions, wall time and GPU-hours. The later public research note may
retain only aggregate metrics and digests; no private path, raw prediction,
training text, credentials, or protected labels are published.
