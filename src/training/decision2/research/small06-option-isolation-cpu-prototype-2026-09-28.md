# 0.6B independent-option encoder: CPU shape and cost gate

**Status: renderer/readout shape passes; training and release remain HOLD.** This
is a prospective architecture probe, not a trained model, an accuracy result,
or a change to the private 0.6B package. It uses synthetic text only and reads
no SELECT, CAL, DEV, FINAL, JevBench, or human-transfer answers.

## Hypothesis and semantic boundary

The current Qwen3-0.6B decision path concatenates all candidates into one
causal prompt, then reads candidate endpoints and a final global query. A later
candidate can attend to earlier candidates; the earlier one cannot attend to
later candidates. Its 69/400 jointly correct original/reordered typed DEV
pairs motivate, but do not prove, an architectural cause. A previously tested
set-interaction head did not fix the backbone asymmetry and failed its own
frozen SELECT gate.

The candidate construction in
[`option_isolation.py`](../training/model/option_isolation.py) instead encodes
each candidate from the **same state and question plus only its own semantic
description**. A shared scalar scorer gives logits, and a symmetric softmax
normalizes them. For a fixed candidate identity, reordering the physical
candidate list therefore leaves its prompt and logit unchanged. The output
probability map and Choice winner are reconstructed by stable keys. The
renderer tokenizes a common prefix separately from each branch so its prefix
token IDs are exactly identical, which is necessary for future prefix-cache
forking. No real Qwen backbone or learned head was exercised here.

The [System One API](https://docs.typesafe.ai/api) gives Choice 2–255 keyed
criteria, Noul a yes probability, and Score 2–10 **ordered** levels with a
probability-weighted value. The prototype accepts these ranges. For Choice,
opaque keys are omitted from the candidate prompt when a nonempty description
exists, so a key rename only remaps the **probability** keys. If two descriptions
are indistinguishable and their logits tie, no key-free model can select one
semantic identity consistently after arbitrary key renaming; the prototype
uses a stable lexical-key tie break. A null or empty criterion
can encode its meaning solely in the key; the key must then enter the prompt,
and arbitrary key-renaming invariance is impossible without losing meaning.
For Noul, true/false are semantic values. For Score, the numeric level is part
of the candidate meaning. Reordering the physical storage of indexed levels
does not change the answer; changing the **rubric's level order** changes the
task and should change the answer. A purely independent candidate scorer may
also fail tasks whose meaning is defined by comparison with the whole option
set. We need data and evaluation to test that tradeoff.

The seven pure CPU tests cover 2/3/10/255 Choice, Noul, 2/3/10 Score,
permuted storage, opaque Choice key renaming with descriptions, null-key
semantics, fixed-score ties, bad option cardinality, duplicate keys, finite
logits, exact shared token prefix, and independent tails. All pass. The test
scorer is a deterministic hash, so this proves shape/equivariance conditional
on a deterministic per-candidate encoder; it does **not** establish accuracy,
calibration, or real-model numerical parity. The prototype intentionally omits
the API's `confidence`, whose formula is not specified in that reference; a
production adapter must implement and validate its own documented definition.

## Pinned tokenizer-only cost probe

An isolated CPU container used the already cached official
`Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd4`
tokenizer and config, without loading weights or using a GPU. The tokenizer,
tokenizer config and model config SHA-256 values are respectively
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`,
`3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5`,
and `504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59`.
The JSON receipt SHA-256 is
`5911aa1b565d92a999025d3c50e3eed110c3b10cd8ef1c386efa54e0f00c7b43`.
The probe is reproducible from
[`option_isolation_probe.py`](../training/model/option_isolation_probe.py);
its state/option strings are synthetic cost fixtures, not benchmark items.
The final renderer source SHA-256 was
`f2646a0c18e99f6c71b6ab5c93c52665a24c59bfabc957fc00d07106a5cba675`;
the separate probe source SHA-256 was
`9370a100b8a7c680a08c1592b2560443559f8eb156eb4f6ba285b404765dd4e5`.
The CPU image ID was
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`;
the exact final-source repeat produced the identical JSON hash on 2026-09-28.
No container remained running after the probe.
The source contains 28 layers, 8 KV heads, 128 dimensions per head. BF16 KV
alone is 114,688 bytes per cached token; listed memory excludes weights,
activations, allocator overhead and training gradients.

| Type | Synthetic state words | Options | Existing joint tokens | Independent no-cache tokens | Token multiplier | Ideal shared-prefix tokens | Batched no-cache KV | Ideal shared-prefix KV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Choice | 64 | 2 | 175 | 250 | 1.43× | 142 | 27 MiB | 16 MiB |
| Choice | 64 | 10 | 375 | 1,250 | 3.33× | 278 | 137 MiB | 30 MiB |
| Choice | 64 | 255 | 7,300 | 32,275 | 4.42× | 4,843 | 3,530 MiB | 530 MiB |
| Choice | 512 | 2 | 670 | 1,240 | 1.85× | 637 | 136 MiB | 70 MiB |
| Choice | 512 | 3 | 695 | 1,860 | 2.68× | 654 | 203 MiB | 72 MiB |
| Choice | 512 | 10 | 870 | 6,200 | 7.13× | 773 | 678 MiB | 85 MiB |
| Choice | 512 | 255 | 7,795 | 158,500 | 20.33× | 5,338 | 17,336 MiB | 584 MiB |
| Choice | 4,096 | 2 | 4,608 | 9,116 | 1.98× | 4,575 | 997 MiB | 500 MiB |
| Choice | 4,096 | 10 | 4,808 | 45,580 | 9.48× | 4,711 | 4,985 MiB | 515 MiB |
| Choice | 4,096 | 255 | 11,733* | 1,162,690 | 99.10×* | 9,276 | 127,169 MiB | 1,015 MiB |
| Noul | 512 | 2 | 659 | 1,242 | 1.88× | 638 | 136 MiB | 70 MiB |
| Score | 512 | 3 | 695 | 1,890 | 2.72× | 684 | 207 MiB | 75 MiB |
| Score | 512 | 10 | 870 | 6,300 | 7.24× | 873 | 689 MiB | 96 MiB |

`*` The 4,096-word/255-option joint prompt exceeds the **current 8,192-token
complete-input cap** and would fail rather than run. Its multiplier is only a
hypothetical token-work ratio. All 255 independent branches are about 4,560
tokens each and fit separately. Token counts depend on this synthetic wording;
they are not latency or FLOP measurements. The no-cache count assumes separate
full backbone passes. The ideal shared-prefix count assumes one prefix
prefill plus each branch continuation; it has **not** been implemented or
validated. Batch KV and shared-prefix KV are mathematical bounds only. Serial
branch execution can lower peak KV memory but repeats work and may make
latency prohibitive. Attention-pair proxies in the JSON receipt omit linear
projection/MLP costs, so a lower pair count does not imply faster inference.

## Falsifiable advancement rule

1. **CPU shape: GO.** Current tests prove the narrow renderer/readout
   invariants, including the documented key and ordinal exceptions.
2. **Full 255-option training/inference: HOLD.** At 512 synthetic state words,
   naive batched no-cache processing needs 20.33× input token work and about
   17 GiB KV alone. The 4,096-word/255-option case is about 124 GiB KV if
   batched without cache. Do not allocate a full training arm based on this
   CPU proof. First implement a real shared-prefix branch path or a bounded
   serial/chunked path, and check actual forward/backward memory, latency and
   numerical parity on fixed 2/3/10/255-option synthetic inputs. Freeze an
   explicit latency/memory ceiling relative to the existing joint path
   **before** the GPU check; reject the architecture if it cannot meet that
   ceiling without dropping supported options, truncating state or changing
   Score semantics.
3. **Quality: untested.** If cost and native parity pass, run one preregistered
   same-official-source, same-group/data/token-exposure contrast, with separate
   Choice, Noul, Score, permutation, CSS transfer and calibration gates. Use
   independent validation because the existing v3 labels were previously
   opened. A correct permutation-equivariant renderer is not evidence of a
   better decision model.

The next research effort with immediate payoff may still be source-disjoint
Choice and three-level Score data rather than this architecture. The CPU result
does not warrant re-training the failed set-interaction or type-separated arms.
