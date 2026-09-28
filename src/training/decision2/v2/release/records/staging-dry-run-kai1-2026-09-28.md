# Release pipeline dry run: Kai1 at 8K into a private staging repository (2026-09-28)

**Result: PASS on every step, CPU only, 0 GPU-hours.** A complete release chain ran
through `v2/release/release.sh` on node A, CPU only, in the pinned image
`decision20-train-fast:host2` with the qualified Kai/Lex venv (Transformers 4.57.6):
build, two-process native examples, the card's own example, scored-panel parity,
private upload, real `hf download`, re-hash, post-download examples, card example and
parity from the download, and Hub readback. The staging repository is **not** in the
"Decision 2.0" collection and no Decision 2.0 model was published. Receipts:
[`staging-dry-run-kai1-2026-09-28/`](staging-dry-run-kai1-2026-09-28/) (step receipts,
`RELEASE-RECEIPT.json`, spec), [`cpu-integration-qwen-2026-09-28.json`](cpu-integration-qwen-2026-09-28.json).

## What was packaged

| Item | Value |
| --- | --- |
| Checkpoint | Decision 1.0 Kai native export at runtime-bearing revision `7185f514f54b8f93c55998b1e8f9c5cc67f0d029` (weights byte-identical to current `main` `9d6872cd`), native manifest `c1bf07ab1c4c3fa1f819256d3de858d1ed87869bdfa663553280d7e78b88bee4` |
| Profile / cap | `kai-native`, 8,192-token complete-input cap (the 0.6B track's `kai1-native-8k` formal run) |
| Builder source | exact mirror of `0c593668b23527b98a4e11fc59b2ab3bbeb5150e` (subtree `src/training/decision2`) |
| Repository | `llm-semantic-router/dev2-release-staging`, **private**, revision `5afd8fc89502294aa63ee0e0d823ef60f1def29e` |
| Package | 64 files, 2,323,014,676 bytes; `MODEL_MANIFEST.json` `592123aa950554ae54786004c7f2694fc6b6342c63996b054666a04bfa537352`; root `config.json` `aa1fcd1e4576049882dd8db52f10b45bcfc938b3179a2778765c50439abf2dcd` |
| Parameters | 571,909,635 from safetensors headers = loaded count asserted by the runtime (pre-upload and post-download) |
| Weights | `native/encoder/model.safetensors` `4c95e05a…`, `choice_encoder` `bb520f1e…`, `score_encoder` `400d1d9c…`, `decision_heads` `730fa21b…`: equal to the public Kai `main` LFS hashes |
| Card metadata | `license: other` (the inherited Gemma-origin tokenizer terms are not Apache-2.0; component table in `LICENSING.md`), `base_model: llm-semantic-router/Decision-1.0-Kai-0.6B` |

## What was verified

| Step | Evidence |
| --- | --- |
| Build | exact native tree and manifest; pinned Kai runtime sources (10 files) and native runtime hashes; header count = declared 571,909,635; name `DEV2.0-0.6B` follows the loaded size; 62 text files screened (credentials, private paths, addresses); PNGs without text chunks |
| Native examples (pre-upload, two processes) | 5 System One requests (Choice with criteria, Noul with and without criteria, Score with 3 and 5 levels, structured state, Chinese state, Choice with null descriptions) all valid; an over-budget request returned explicit `context_overflow` errors with nothing truncated. Answers SHA-256 `3fe1ef1c847f8aa540cc1656652bab1781a9ca98f86df6e0981d986f2c738e67` in both processes (bit-identical, 11/11 slots) |
| Card example (pre-upload) | the README's Python block, executed unchanged in a fresh isolated interpreter, printed the same 3 answers bit for bit |
| Scored-panel parity (pre-upload) | first 24 typed-final, 12 human-transfer and 12 public-231 gold-free prompts vs the sealed `kai1-native-8k` predictions (typed `762d721a…`, transfer `48593d4d…`, public `2fd20c6e…`): 48/48 slots, 0 category changes, 0 input-digest mismatches, max probability drift 2.95e-6 (CPU FP32 vs the scored ROCm FP32 run; tolerance 1e-4) |
| Upload | `ensure` found the repository private; upload committed `5afd8fc8…` with the manifest above; still private afterwards |
| Real download | `hf download … --revision 5afd8fc8… --local-dir <fresh dir>` with a fresh cache: 64 files plus the Hub's own `.gitattributes` |
| Re-hash | every downloaded file equals `MODEL_MANIFEST.json` and the pre-upload package (0 missing, 0 extra, 0 mismatched) |
| Post-download examples | same answers SHA-256 `3fe1ef1c…` (bit-identical to pre-upload), 571,909,635 loaded parameters |
| Card example and parity (download) | card block identical again; parity identical to pre-upload (48/48, drift 2.95e-6) |
| Readback | private, exact revision, 64 remote files with matching LFS SHA-256 or git blob ids, Hub-parsed card metadata (`license: other`, `base_model` Kai) equals the front matter, every card image and link resolves, collection "Decision 2.0" private with **0 items**, staging repo not in it |
| Collection calls | on a temporary **private** scratch collection (never "Decision 2.0"): created private, staging repo added and read back, collection deleted and deletion confirmed |

The card rendered with the DEV2.0-0.6B owl banner, the same-panel table and the three
charts (v3 rank, model × task, public-231 rank; no Pareto chart). The licence filter kept
Kai, Lex, Bosun and GLiNER2.5 (roster licences Apache-2.0) and excluded the deleted old
DEV2.0-0.6B control as internal-only. The tradeoffs table correctly reported no result
below Kai (same weights; the 8K cap only adds answered long inputs: public 231 127 vs 114,
v3 35.969 vs 35.938, paired interval [−0.06, +1.96]).

Separately, `v2.release.tests.cpu_integration` built tiny but real `qwen-full` and
base-bound `qwen-adapter` packages through the shared training code and passed the same
isolated load, examples, bit-identical repeat and (full) card-example checks; the adapter
changed the outputs, proving it is applied.

## Timing and cost

CPU wall time: 837 s for the full chain with upload (builder 6 s, each examples process
~1 min, parity ~4.5 min per side, upload 2.6 s because the Hub already stores these public
weight bytes, download 7.5 s). A no-upload rehearsal (`dryrun-kai1-local-…`) and the qwen
integration test (34 s) also passed. **0 GPU-hours**; no GPU lease was taken.

## Issues found and fixed during the rehearsal

1. The launcher lacked its executable bit in git (`core.fileMode=false` in this checkout): fixed.
2. The Hub's metadata validator rejects a relative `license_link`; the first upload attempt
   stopped client-side before any file was sent (the repository had been created private
   and stayed empty). The card now links `LICENSING.md` by its https Hub URL.
3. Readback now also accepts a one-element `base_model` list; the card now draws the packaged
   candidate in the Decision 2.0 colour (the staging chart shows it in slate because that
   report's family is Decision 1.0).

The staging repository is left **private** at `5afd8fc8…` as evidence (it holds the
unchanged public Kai weights); delete it with the HF CLI once no longer needed.
