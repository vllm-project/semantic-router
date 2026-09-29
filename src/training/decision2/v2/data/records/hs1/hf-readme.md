# `m4/hs1/` — HS1 hard-skill families (private)

Three synthetic English training families for skills where Decision 2.0 trailed peers: checking a quoted
conclusion against the evidence, reading long policy packets, and answering "no" when one required condition is
not met. Built by the research & data track (HS1, build commit `21bdb5e90e2b`) and frozen like the `m2/arms/` and
`m3/arms/` arms: content hash = SHA-256 of the canonical JSONL (rows sorted by `id`). Rows follow the Decision 2.0
training contract. They carry **gold labels only**: no teacher targets exist for them.

**Revision of 2026-09-30: `train.jsonl` was replaced by the template-fixed build `46702e2fce2b`**
(`0dfaa6ebff1b614e5f1c432f2c7432c3fd217f7c344d402c7a7b37fe507de5e7`).

- **The defect.** A blind spot-check of 153 TRAIN rows (51 per family) agreed with the gold on all 153. It
  confirmed one template defect: in 150 F2 travel-expense rows the policy said "at least N nights consecutive
  nights". This was a duplicated word; the answer was unaffected.
- **What changed.**
  - 138 rows differ only in that phrase, which now reads "at least N nights in a row".
  - In 12 rows the packet also gained one filler section, because of the generator's length rule, and its clause
    numbers shifted to match.
  - Ids, labels, options and instructions are identical to the first build.
- **Audits.** All were re-run on the new file:
  - PI-v4 full scan: the same 52 report-only groups; quarantining roles: 0; PI-hs1: 0.
  - Structural scan: 0.
  - Isolation: PASS.
  - Local audit (A2, A3a–d, A7): PASS.
  - Native tokens: 14,030,228.
- **Kept from the first build.** `train.tokens.jsonl`, `train.manifest.json`, `build.json`, `isolation.json` and
  `audits/` describe the new build.
  - `dev.jsonl`, `dev.tokens.jsonl`, `dev.manifest.json` and the `hs1-dev` panel are unchanged, so development
    readouts stay comparable. Twenty dev rows keep the duplicated word.
  - `build-21bdb5e90e2b.json` is the first build's receipt.
  - `audits/dq-hs1-spot.json` is the spot-check.

| Family | Content | TRAIN rows (groups) | Choice / Noul / Score (TRAIN) | Dev rows: dev-id + dev-ood | Language |
| --- | --- | ---: | --- | --- | --- |
| F1 `hs1_quote_check` | Quoted-conclusion verification. Evidence records (a short note, or a long thread or report with distractor records) plus a named person's conclusion and rationale. The two rows of a world differ only in the quote: one follows the solver, the other follows one generic reasoning slip. | 7,178 (3,589) | 2,946 / 3,332 / 900 | 958: 638 + 320 | en |
| F2 `hs1_policy_packet` | Long policy packets of 4–12k characters (TRAIN 4,415–11,472, median 5,860) with amendments in and not yet in force, superseded text, exceptions, annexes, temporary measures and an explicit precedence section; two cases per packet. | 2,398 (1,199) | 1,088 / 832 / 478 | 478: 318 + 160 | en |
| F3 `hs1_unmet_condition` | Condition-not-met Noul negatives (one required condition fails, or two) with matched positive twins: the same world with the failing attribute moved just inside the rule. | 10,392 (5,196) | 2,072 / 7,280 / 1,040 | 960: 640 + 320 | en |
| **Total** | | **19,968 (9,984)** | 6,106 / 11,444 / 2,418 | 2,396: 1,596 + 800 | |

`dev.jsonl` is the **`hs1-dev` development slice** (split `select`): dev-id uses the TRAIN kinds and domains with
fresh seed groups, dev-ood only held-out kinds and domains that have no TRAIN rows; TRAIN and dev come from disjoint
seed namespaces. **Never train on it.** It is for development readouts only. The gold-free prompts and gold of the
`hs1-dev` panel are not in this dataset.

Files: `train.jsonl` (`0dfaa6ebff1b614e5f1c432f2c7432c3fd217f7c344d402c7a7b37fe507de5e7`; first build
`c90ef3164d90d3fd6a1ab0397529faec172760b1756de7d4d8cbf2677a131a71`), `dev.jsonl`
(`2e9ee9ab6768aa2b3d0d397670248e5a8f2e524d60887d06698a4e84516533bf`), `train.tokens.jsonl` / `dev.tokens.jsonl`
(`{id, native, kai}` per row: Qwen3.5-0.8B-Base native encode and raw Kai-0.6B; join to rows by `id`),
`train.manifest.json` / `dev.manifest.json` (freeze manifests: counts, token totals, licence table; dev frozen with
role `aho`, partition `select`), `build.json` (generator code hashes, per-family counts, the dropped-group count),
`isolation.json`, and `audits/`: the public overlap receipts against PI-v4 (`overlap-piv4.public.json`, all roles;
`overlap-piv4q.public.json`, quarantining roles) and PI-hs1 (`overlap-pihs1.public.json`), the structural scan
(`struct.public.json`), the local audit (`hs1.audit.json`, counts only) and `confirmation-summary.json` (counts
only: overlap by role class, tokens per family, gold and duplicate checks). `license-registry-hs1.json` is the
licence registry. `registry.json` lists every file here with its SHA-256 (paths relative to `m4/`).

**Tokens** (native Qwen3.5-0.8B): TRAIN 14,030,228 in the fixed build (first build 14,029,697: F1 5,594,547;
F2 3,939,776; F3 4,495,374), dev 1,946,468. The
longest row has 2,875 native tokens (TRAIN) and 2,774 (dev). Rows over 1,024 Kai tokens: 3,891 TRAIN (F2 2,393,
F1 1,498, F3 0) and 693 dev.

**Gates** (preregistration §5 with amendment 1; all pass):

- A1 gold: the oracle equals an independent re-check on 22,364 of 22,364 rows. Any disagreement stops the build.
- A2 label balance (TRAIN and dev): Noul true share .499 / .500 / .500 (F1 / F2 / F3 TRAIN); F1 right-quote share
  exactly .50; Choice gold and quoted-option positions within 3 points of uniform and Score levels within 0.8–1.2×
  uniform in every cell of 100 or more rows.
- A3a shortcut cells (`v2.data.shortcut`, state-removed and option-only): 3 family-pooled and 22 per-kind cells
  pass.
- A3b bag-of-words family probes (F1 without the evidence, also Noul-verify alone; F2 without the policy; F3):
  PASS, no WARN.
- A3c heuristics: adopting (or rejecting) the quote is right on exactly .50 of F1 rows; the naive F2 readings (base
  text only, latest amendment, first matching provision) reach at most .41 on Noul and .33 on Choice; the F3
  negation and near-threshold heuristics .50.
- A3d length probe: at most majority + .016 on TRAIN.
- A4a lexical overlap (`v2.data.overlap`, rules E / S / L / N): a first scan removed 18 groups (below). The
  confirmation scans of these files find **0 quarantining hits** against PI-v4 (full and quarantining-role
  manifests) and PI-hs1 (HT-DEV, Score5-DEV and score5t-dev gold-free prompts, which PI-v4 lacks). Report-only:
  52 groups (104 rows), all against PI-v4 report-only roles (the v1 TRAIN arms A2, A3, A4v2h, A4v2r and the v2
  AHO slice G2).
- A4b structural near-duplicates (masked word 8-grams): 0 flags against 19,623 items in 11 evaluation and
  development panel files.
- A4c: HS1 has no external source; no C1 source-registry entry applies.
- A5 lengths: every F2 state has at least 4,000 characters; every row is within the 8,192 (TRAIN) and 16,384 (dev)
  native-token limits.
- A6 isolation: TRAIN and dev share no id, group or input hash; 0 within-family exact duplicates (state +
  instructions).
- A7 independence: 0 hits for the benchmark family names listed in the preregistration in generator code, row
  metadata and row text. The generator imports nothing from `benchmark`, `jev_arena` or `v2.eval`.

**Disclosure.**

- 18 groups (32 TRAIN rows, 4 dev rows) were quarantined: the PI-v4 lexical overlap scan of the first build flagged
  them against quarantining roles, and both rows of each group were dropped. That is why TRAIN has 19,968 rows and
  dev 2,396, not the preregistered 20,000 and 2,400.
- The TRAIN token total (14.03M native) is above the preregistered target (~8.2M, ±25%). The preregistration's token
  estimate was wrong by about a factor of two (amendment 1); rows and state lengths stayed as preregistered and no
  row was cut.
- Adjacent typed skills: F2's precedence and exception reasoning and F3's condition gating are the same abstract
  skills as typed FINAL `exception_stack` and typed DEV `rule_precedence` / `attribute_gate`. HS1 uses none of
  their generators, schemas or items, but gains on those typed families after HS1 training may be skill transfer.
- 23 groups share lexical matches with the report-only v2 AHO slice G2, so for a model trained with HS1 a G2 AHO
  readout is slightly familiar.
- No LLM text: every string is rendered by project code from project-authored templates. No model wrote,
  paraphrased, filtered or labelled any row; labels come from a deterministic solver on a structured world.

Rules and results: branch `xunzhuo/decision-2-training` (track branch
`xunzhuo/decision-2-training-data-hardskills`), `src/training/decision2/v2/data/records/`
(`hs1-prereg-2026-09-29.md` and amendment 1, `hs1-prereg-amendment-1-2026-09-29.md`). HS1 is synthetic text
generated by project code from project-authored templates; it contains no third-party text and no model-written
text, and may be redistributed with the models under Apache-2.0. Keep this dataset private.
