# Lux 1.0 native JevArena v3 control: prospective execution lock

**Scope:** one fixed own-1.0 control paired with the already sealed JPT-9B
same-panel peer and future eligible 9B Decision 2.0 candidates. No existing
Lux 1.0 typed FINAL/CSS15 complete prediction pair was found on either
authorized node. The panel keys were opened earlier for unrelated arms, so
this is explicitly **post-key same-panel**, not unseen validation. This lock
precedes the Lux FINAL/CSS15 predictions and scores.

## Fixed model, runtime and panels

| Item | Frozen value |
| --- | --- |
| Own published model | `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8`; downloaded revision metadata attests that exact commit. The native bundle contains 29 manifest-listed files. Bundle-manifest SHA-256 `985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd`, decision config `656b1717b8cfebad6253e9d1321023323651ad3c80327562b7d1542ec74c23d1`, runtime profile `2fffc75c6c681d7056b24ca660f95491f12816d0ed1c4a70ff3dbee3896044bb`. |
| Native decision inference | Unmodified `inference/run.py` SHA-256 `b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce`, backend `lux`, exact `state`/`questions` in published `DecisionModel.decide`. No chat projection, input truncation, option dropping or unvalidated-runtime override. |
| Gold-free seal | `jev_arena/seal_lux9b_peer.py` SHA-256 `ddc526425b6377f15ea37117c95a9315c7bc7cb5b27a80eedeed5c8571ef2753`; requires exact revision/manifest, original question IDs/hashes and `runtime_matches_validated=True` on every output. |
| Typed FINAL | 1,600 items / 2,000 answers, input SHA-256 `e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd`; separate gold SHA-256 `707dd28dfbab10d124d437434023729f319501542e999e536fce9b7ff7f2361e`. |
| CSS15 | 6,547 items / 6,547 answers, input SHA-256 `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6`; separate gold SHA-256 `1cda9623032138bb7b124be0c1b0a4239c06bed7be3169264e6eb31805c19ba4`. |
| Fixed metrics | Typed scorer SHA-256 `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`; CSS scorer SHA-256 `cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`; v3 score `100 × sqrt(T × H)`. Pair to frozen JPT with `jev_arena/compare_v3.py`, 5,000 paired bootstrap draws, seed `20260927`. |
| Compute | One isolated live-idle MI325X GPU on the other authorized 8-GPU node. Preloaded published weight package, exact gold-free prompts and pinned offline ROCm image `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`; no new weight download, training or restart of unrelated jobs. Budget **at most 0.5 GPU-hour**, including smoke/load/typed/CSS; hard wall cap 30 minutes. |

The fixed executable for each panel is `python -m inference.run --backend lux
--model-path /model --model-revision <revision> --input
/panels/<panel>.prompts.jsonl --output /out/<panel>.predictions.jsonl --device
cuda:0`. `/model` and `/panels` are read-only mounts and contain no gold.
The earlier independent 32-question gold-free prompt SHA-256 is
`ce16f6107b8a8da6d0e5ef501cc8360da07a7232ecd7289d5a437d1d77767b67`.
Its prior answer-map receipt is a screening reference from an older wrapper,
not a source for FINAL scores. Require the native published runtime to match
its profile and the smoke to have all three types, 32 complete answers, zero
category changes and maximum probability drift at most 0.02 against the old
map. If wrapper differences violate that gate, stop and investigate rather
than silently changing the protocol.

## Fixed stop and score order

1. Verify source/mirror SHA, model revision, bundle files, image, prompts,
   idle GPU, disk and the smoke reference before reserving GPU. Run the
   bounded smoke, then typed FINAL once and CSS15 once in separate processes.
   A missing, invalid or over-budget answer counts as wrong in the same
   scorer; if the collector aborts or times out, retain the partial file and
   report incomplete rather than dropping rows or changing prompts.
2. Validate all 8,147 item IDs and 8,547 answer slots against original input
   hashes. Hash and fsync each complete prediction and a joint gold-free
   freeze **before reading either formal gold file**. Retain UTC start/end,
   GPU-seconds, source/image/model hashes and any native runtime failure.
3. Only after the joint freeze, run unchanged typed/CSS scorers and compute
   T/H/v3, type, family, task, invalidity and calibration views. Run the
   fixed paired v3 bootstrap against JPT on the same two panels. Previously
   reported public231 Lux/JPT scores remain separate; reuse only if its
   exact public prompt, target, model and scorer identities match. Do not
   call this an official or sealed new blind ranking.

Raw prompts, labels, predictions, private paths, host details and logs stay in
the private research directory. The repository and unified gist receive only
aggregate findings and non-sensitive digests. No HF upload or publication.
