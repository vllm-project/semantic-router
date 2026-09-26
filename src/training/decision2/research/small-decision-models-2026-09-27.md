# Decision 2.0 0.6B architecture comparison: additional primary sources

**Research note, not a same-panel model result.** This extends the existing
Kai/Laya/GLiNER/Reranker 0.6B diagnostics. Claims about the two additional
checkpoints below are their authors' model-card results; none is entered into
our JevArena rank without a pinned native rerun.

| Checkpoint and primary source | Mechanism and training | Evidence limit |
| --- | --- | --- |
| [`thefloydd/qwen3-0.6b-rlcd`](https://huggingface.co/thefloydd/qwen3-0.6b-rlcd) | Qwen3-0.6B-Base plus merged LoRA; prefill-only option states and block-diagonal question isolation; proper-scoring supervised training over about 507k questions/2,456 task schemas; separate in-list confidence head. | Author reports 0.783 accuracy and .031 ECE on an internal 79,396-question task-held-out panel. The card says English only, training inputs up to 2,304 tokens, weaker plain-text/NLI/multihop performance, and domain-variable calibration. The visible model-index metadata still lists an earlier held-out protocol, so pin a revision and use the card's exact result version. |
| [`anthonym21/qwen3-0.6b-rlcd-decision`](https://huggingface.co/anthonym21/qwen3-0.6b-rlcd-decision) and [source repository](https://github.com/anthony-maio/eve-rlcd) | Qwen3-0.6B-Base decision-only body and 26-letter head; supervised letter-format warmup followed by 500 steps of bandit proper-scoring reward on separate training rows. Shared-state prefill, one batched option-letter readout per question. | Author reports 0.807 vs 0.746 warmup accuracy and .021 ECE on its in-distribution test. Training/evaluation prompts were at most 512 tokens; the card explicitly says longer inputs and unseen domains are unmeasured. The loader is external to stock Transformers. |

The first model's training is supervised despite the `rlcd` repository name;
the second actually runs a policy-gradient loop. Their self-reported numbers
are on different private/public mixtures, so cannot rank them against each
other, Kai, GLiNER or Jev. The authors' confidence definitions also differ:
one estimates whether the correct answer is within the options, while the
other reports maximum option probability. Do not merge these into a single
calibration score without declaring what is measured.

## Same-panel test before a Decision 2.0 architecture choice

1. On an authorized experiment host, pin each upstream commit, tokenizer,
   custom loader code and safetensor hashes via HF CLI. Inspect executable
   custom code before importing it. Count actual parameters and verify that
   native Choice, Noul and Score responses can be mapped without changing
   their meaning. The `anthonym21` state/question truncation defaults must be
   recorded per row; any inadmissible or truncated question is an invalid
   answer under the unchanged benchmark rule.
2. Run both on the exact existing typed DEV1600, CSS pilot1430 and public231
   prompts with the same panel hashes as Kai/GLiNER. Keep the 15-task CSS
   formal set and authored FINAL sealed. Report type/family/task results,
   validity, calibration, paired intervals, context-length buckets and
   option-order robustness. Published internal scores stay in a separate
   context table.
3. Only after that compare training starts: our existing Kai encoder, the
   GLiNER native head, Qwen3-0.6B-Base with a declared option-scoring head,
   and any well-audited pretrained decision head. Use the same rights-clean
   TRAIN/SELECT/CAL and matched token/update budgets. Test 0.6B transfer and
   long-context retention explicitly. Do not choose a release backbone from
   a self-reported internal number or a public diagnostic alone.

The prior same-panel records remain: Kai 1.0 typed DEV425/1600; Kai rights-
clean continuation DEV587/1600 but CSS pilot regression; GLiNER English
source DEV652/1600 and a 64-step continuation DEV584/1600; Qwen3-Reranker
0.6B continuation DEV431/1600 against source422. These development figures
use different native adapters and are diagnostic, not release claims.
