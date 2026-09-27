# Decision 2.0 27B: v3 gate and next discriminating experiment

**Status: HOLD.** This is a read-only audit of the existing BEST368 candidate,
its adapter-preserving package and the completed Score experiments. It adds no
training, GPU inference, JevArena FINAL or heldout human-transfer result. The
first-release v3 headline now uses the 1,600 typed FINAL and 6,547 human
transfer items; authored items are reserved for v3.1. No historical DEV score
can be substituted for either sealed v3 axis.

## Fixed candidate and native package

The posttrained `Qwen/Qwen3.8-27B` source is pinned at
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. BEST368 has native
model fingerprint
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`.
The adapter-preserving package manifest SHA-256 is
`e6ca0f51bf7f27c938f35cbdbc6802d72b10a1d3d33405765fe491c32e25110b`;
the native CAL file SHA-256 is
`e4e0d9fda575d807503299bdf7c67828a7c4d3c3b09e9fa1bf3750a51b79db78`.
The package counts **25,688,227,840 loaded inference parameters**:
25,624,600,064 base text, 58,363,904 LoRA and 5,263,872 decision head. The
upstream vision path and generation head are not active in this decision
adapter. The original nine-item/27-question gold-free parity receipt SHA-256
is `c23868b95324634aa0a1c2ef8fd4408e0e6118f0966ec957b51f4cc1771b4b2d`.

Historical source inference on another machine differed from the package on
some near-tie predictions, so its DEV 1,213/1,600, CSS pilot 842/1,430 and
public 199/231 scores are **not transferable**. Under matched physical GPU,
runtime, model and CAL, full-panel source/package parity subsequently passed:
all 1,600 DEV and 1,430 CSS pilot answer objects matched, with zero scalar
drift; the two shared over-budget CSS answers remained failures. The release
parity receipt SHA-256 is
`25937c0ea07a9fdbb3ca8e64e856f337437221c83233f44fe26556e5fe53a621`.
That check cost 0.184 incremental GPU-hours and does not identify a unique
cause for the earlier cross-machine BF16 drift. The package-native development
observations are DEV **1,211/1,600**, CSS pilot **845/1,430** and public
JevBench subset **200/231**. They are neither sealed v3 nor official
JevBench results. The complete evidence is in
[the package-native report](adapter-best368-package-native-dev-2026-09-27.md)
and [same-GPU parity report](adapter-best368-full-panel-same-gpu-parity-2026-09-27.md).

Both existing 27B package manifests use the former lowercase model ID. The
current publication contract and the user's final repository name require
`llm-semantic-router/DEV2.0-27B`. Before any v3 prediction, rebuild a fresh
uppercase package from the exact pinned base, BEST368 LoRA/head/tokenizer,
scored inference sources, CAL and runtime; reverify every functional byte and
the 25,688,227,840 loaded count. The new manifest and package revision are
new identities even if every model tensor is unchanged. A gold-free native
repeatability and source/package parity check must bind the new package before
any sealed panel. Do not rename the old result or transfer its score by
changing only a label. No same-size Decision 1.0 27B comparator exists in the
v3 roster; any Lux 9B comparison must be predeclared as **nearest-size**, not
an equal-size win.

## Capability blocker and prospective work

On package-native typed DEV, Score is **162/400** despite Choice **791/800**.
The original native Score error audit found 161/400, with 102/107 true
middle-level rows and 121/208 true high-level rows predicted at level 0; its
corrected aggregate SHA-256 is
`73aed9d30c05ebc9829522209101b729f1b2a42ba15b62ac2f48821c9935a3c4`.
This is a hard-decision problem, not merely a temperature choice. The
independent hard-CAL transport improved Score Brier but regressed Noul Brier
by 0.088918, beyond its frozen 0.005 limit, so it was rejected; the frozen
fit/report hashes are in
[the hard-CAL note](qwen38-27b-hardcal-screen-prereg-2026-09-27.md).

The adapter-preserving matched Score v6 English continuation completed both
arms from the same BEST368 start with zero-step parity. Treatment gained
**11/192** over its token-matched replay control, one below the prospective
**12/192** advance threshold; it recovered middle-level answers while losing
high-level answers. The private paired-score receipt SHA-256 is
`78160b60ccca791be9fcb2d879ea66d6611e1d40b736d69b3694b9244c9cf1f9`.
Its result is `DO_NOT_ADVANCE`; the consumed selector, failed merge-path pilot,
and both completed checkpoints remain unchanged. No new 27B training is
justified by those development observations alone.

The next highest-information intervention is the proposed **v7 evidence-state
curriculum**, still a design draft, not an admitted dataset. Before spending
GPU-hours, freeze and verify independent structured oracles, source necessity,
group-held-out shortcut screens, overlap against protected gold-free prompts,
rights and multilingual editorial review. A new independent, blind Score r3
selector must pass its own QA and be frozen before any candidate response; the
used r2 selector cannot be reopened. Then preregister a matched direct-LoRA
A/B continuation from the exact BEST368 base plus adapter, including identical
replay, optimizer, seed, native adapter, token and update budget, one final
checkpoint, zero-step output gate, Score level/operation floors and parent
Choice/Noul retention. A passing new selector would authorize separate
DEV/transfer diagnostics, not immediate v3 publication. See
[the v7 proposal](score-v7-prospective-synthesis-2026-09-27.md).

This audit used **0 new GPU-hours**. The current 27B release blockers are the
known Score deficit, a prospectively admitted replacement experiment, the
uppercase package requalification, and subsequently the same-panel sealed v3
and public-subset gates. It does not hold up a different size's first release.
