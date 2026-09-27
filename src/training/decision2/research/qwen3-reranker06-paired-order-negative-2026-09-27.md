# Qwen3-Reranker-0.6B paired option-order screen: negative result

**Decision: STOP_AT_SELECT; research only.** The prospective protocol is
[`qwen3-reranker06-paired-order-prereg-2026-09-27.md`](qwen3-reranker06-paired-order-prereg-2026-09-27.md),
signed before this arm's first optimizer update. The fixed 64-step treatment
finished, but it did not clear the first, source-disjoint SELECT gate. No typed
DEV, CSS pilot, public231, protected FINAL, authored release or CSS15 score was
collected for the treatment. Do not place this adapter in a JevArena rank,
claim unseen-task transfer, or publish it as `dev-2.0-0.6b`.

## Exact source and isolation

The 595,776,512-parameter [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
source revision was `e61197ed45024b0ed8a2d74b80b4d909f1255473` with model SHA-256
`27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b`.
The completed, **not retrained** sampled-eight control adapter had SHA-256
`d6fb8ac11f9e9cb95fc082b759c5ec5c716ad200c6efb735fe0bf41cc5e3a05a`.
The same rights-clean parent TRAIN7455, 512-row fixed slice, SELECT700 and CAL700
bytes were verified against SHA-256 values `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
`32a1226931967fd7a0f53cb2a4fd51189d5b643c56eeb1b88cea94ebf4312398`,
`32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`
and `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
The rights manifest SHA-256 was
`61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8`.
Its exact ID/group/canonical-context checks and approximate 8-band SimHash
near-context audit reported zero TRAIN and SELECT overlap with the frozen
development and protected panels. The near audit is approximate, not a
mathematical guarantee. Source licenses and redistribution limits remain in
the private data manifest; no raw rows or prediction text is in this branch.

The code was implemented locally, signed at `657ffc0dac9de64f73c6b721e5c094352d060c86`,
and mirrored at that exact revision before GPU execution. Treatment code
SHA-256 was `3b75abb11707f02e593a113ba2e7c932b78629ce2979838330dcc6a162a5b085`;
unchanged native candidate-renderer and control-trainer SHA-256 values were
`2892c0b577c5867d03c66ea72871077e26cf1144e2a605d9d1225b882971b9a1`
and `eafdaa1de3e04188b34023060f69c319515b0e2ccbd42b384278b7bc35bd7142`.
Local focused tests and full `make check` passed. The runtime was one visible
BF16 AMD GPU in an image whose exact digest is retained in the private run
receipt, with PyTorch `2.12.0+git6bbd260`, ROCm `7.2.53211`, Transformers
`5.17.0`, and PEFT `0.21.0`. The isolated inference/training container had
networking disabled.

## Preflight and fixed treatment

The pre-optimizer receipt SHA-256 was
`c2091b6a2ec6d109f12c3cac48a6296d2587abe168ba72f6c0afccfcb00780d4`.
All 192 Choice rows formed semantic-key-preserving reverse-order pairs;
the longest native view was **4,042 tokens**, below the unchanged 4,096
budget. A zero-update LoRA model agreed categorically with the source on a
seeded 32-row gold-free SELECT packet, and paired loss produced finite,
nonzero LoRA-only gradients. No base gradient appeared.

The treatment ran exactly **64/64** optimizer updates with the source's
rank-eight LoRA and fixed sampled-eight negatives. It used 192 paired Choice
rows and 320 unchanged single-view Noul/Score rows. Its mean training loss was
`1.20335`, runtime about **182 seconds**, and final adapter SHA-256
`c89a7e669f4cc339b6ce0e11728062cf4cb5fed49a8afdae5aa5efdf1a7b9633`.
The training report SHA-256 was
`eac43e80f9b11b88ec09c8223cb807ed4b671a5526ef14e3319cd3ab238940c7`.
Because paired Choice rows require a second forward view, this arm used more
FLOPs than the historical control although source rows, optimizer updates,
selected negative keys, rank, LR, seed and inference adapter match. It is a
causal loss/augmentation screen, not a compute-matched throughput comparison.

The saved adapter produced 700/700 native-valid gold-free SELECT answers;
the sealed prediction SHA-256 was
`a43e2d02e040f3c0c0d875e69377d099987242375af5947691d027ceefe8aa4c`.
A **second independent load of those saved bytes** matched the first load's
categorical answers on the same seeded 32/32 packet, receipt SHA-256
`17c7aa6b54f07f4a7ba202a9a5a4986c73e29c3624dc96282060df520c57e1d7`.
The original in-memory pre-save logits were not retained, so this establishes
saved-package reload repeatability, not pre-save versus post-save numeric
parity. That narrower evidence is insufficient for a release package gate.

## Frozen SELECT comparison

Only after sealing predictions and the second-load check was SELECT gold
scored. The source and historical control were scored on the **same** SELECT
bytes under their pinned native adapter. The control prediction SHA-256
`8ecbe12707f048f3793463d3fe6ad68c9449f9ed7aace8ee48ed562bb53cc20c`
and score SHA-256
`72caf24695eff764baaaaaa8f80df802366f61986b361eee114b67ee04641104`
matched their original receipts. The new score report SHA-256 was
`5308f35c7de581fedc22fbe9353df5fdd077a37b60a004a375ee713adcc4bfb8`.

| SELECT700 metric | Published source | Completed 64-step CE control | Paired-order treatment |
| --- | ---: | ---: | ---: |
| Correct | 302 | **351** | **351** |
| Family-macro accuracy | .37798 | .43355 | **.43175** |
| Choice correct / 320 | 158 | 167 | **172** |
| Noul correct / 290 | 139 | 160 | **155** |
| Score correct / 90 | 5 | 24 | **24** |
| Invalid | 0 | 0 | **0** |

The treatment gained five Choice answers while losing five Noul answers.
Against the control, GoEmotions Choice rose 110→116/200, GoEmotions Noul
fell 105→103/200, and natural narrative fell 89→85/130; the other three
family counts were unchanged. Thus the paired Choice objective moved model
behavior, but it did **not** increase overall source-disjoint selection
accuracy or transfer-oriented family macro performance. This is a result for
this one fixed loss, data slice and budget; it does not rule out other
equivariant objectives or architectures.

The preregistered SELECT advance required **at least 365/700** and family
macro at least control +`.02` (≥`.45355`), with type and validity floors.
Observed treatment was **351/700** and `.43175`, failing both positive
conditions. The remaining conditions cannot override this failure. The
sequential stop rule therefore forbids treatment DEV1600, CSS pilot1430 or
public231 evaluation. Earlier native same-panel baselines—Kai 1.0
425/1600 typed DEV, GLiNER English source 652/1600 and 561/1430 CSS—remain
context only. They are **not** compared to this unevaluated treatment on those
panels.

## Next architecture decision

The old Qwen3 reranker control itself had only 431/1600 on typed DEV and no
demonstrated held-out long-input quality, despite admitting long TRAIN rows.
This paired-order treatment supplied no SELECT gain and did not reach a
transfer test. Do not spend the next 0.6B budget repeating this exact
single-slice LoRA recipe. A subsequent prospective experiment should use a
typed decision encoder with a measured longer native window, structured
replay that preserves transition/attribute behavior, and an independent
human/long-input selector; it still needs explicit paired option/label and
counterfactual evaluation, not a generalization claim from training pairs.
