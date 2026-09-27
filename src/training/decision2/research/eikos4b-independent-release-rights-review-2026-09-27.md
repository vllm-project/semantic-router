# Independent 4B release-rights review

**Decision: HOLD the public weight upload on rights documentation.** The
mechanical package/provenance checks and direct TRAIN-to-evaluation overlap
checks pass. The frozen draft still has no named independent rights sign-off,
and it does not resolve how the two CC BY-SA 4.0 training sources affect the
license description for redistributed weights. `license: other` alone is not a
statement of the rights granted to downloaders. This is an evidence and
release-record decision, not a determination that training on those sources
was prohibited.

## Evidence independently checked

- Frozen release draft SHA-256:
  `70712f492358097c69242da656cd3001bc2854e9517f4552007dfef96ba2871b`.
  The private mechanical rights/provenance receipt SHA-256 is
  `e714b381d19550bae1674af3fcbe1860ef73268aa4c8c07a268c12cb01a5cf48`.
  The draft records an empty `reviewed_by` and pending rights status.
- The candidate package's `SHA256SUMS` verified for every listed file. Its
  `LICENSE`, `LICENSE-Qwen`, and `NOTICE` hashes match the pinned Eikos source
  copies recorded in the mechanical receipt. The upstream
  [Eikos-4B card](https://huggingface.co/caiovicentino1/Eikos-4B) describes
  MIT for its own contributions, Apache-2.0 for the Qwen base, and the
  third-party datasets preserved in `NOTICE`. The
  [Qwen3.5-4B-Base card](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
  identifies Apache-2.0. The package therefore must not be described simply
  as wholly MIT-licensed.
- The clean-v2 rights manifest is bound by hash to the candidate provenance;
  its scope is trained weights and model card, excluding raw rows, SELECT/CAL
  rows, and individual text predictions. The manifest includes 272 SNLI and
  334 SQuAD 2.0 TRAIN rows, both listed as CC BY-SA 4.0 by their
  [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and
  [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) publishers.
  [Creative Commons' ShareAlike terms](https://creativecommons.org/licenses/by-sa/4.0/legalcode)
  apply when adapted material is shared. Whether this checkpoint is an
  adaptation of those input texts is not established by the receipt. The
  release record must state a reviewed disposition before assigning a public
  weight license.
- For GoEmotions, the
  [original Google Research repository](https://github.com/google-research/google-research/blob/master/README.md)
  states CC BY 4.0 for datasets and Apache-2.0 for source files. The
  [Hugging Face dataset badge](https://huggingface.co/datasets/google-research-datasets/go_emotions)
  instead says Apache-2.0, apparently reflecting the code repository. The
  frozen manifest's CC BY 4.0 treatment is supported by the original dataset
  publisher; the model card should cite that primary source to explain the
  discrepancy. Other checked source cards identify
  [FLUTE as AFL 3.0](https://huggingface.co/datasets/ColumbiaNLP/FLUTE),
  [CosmosQA as CC BY 4.0](https://huggingface.co/datasets/allenai/cosmos_qa),
  [BANKING77 as CC BY 4.0](https://huggingface.co/datasets/PolyAI/banking77),
  and [CLINC150 as CC BY 3.0](https://github.com/clinc/oos-eval/blob/master/LICENSE).
- The separate gold-free overlap audit receipt SHA-256 is
  `9d66daaac33121e290b45db036cc713fc86a3a9e59577a46ffd1129c2f0cb2a8`.
  It found zero shared IDs, exact states, normalized states, and approximate
  near matches between TRAIN 7,455 and frozen typed FINAL 1,600, CSS15 6,547,
  or public JevBench 231. The 120 TRAIN and 500 evaluation FLUTE examples have
  no shared original record IDs. These checks do not prove semantic or
  pretraining independence; FLUTE remains a same-source task-family comparison.

## Minimum changes to clear this HOLD

1. Add a named, dated rights-review disposition for the 606 CC BY-SA training
   rows and the intended public weight-license text. If the disposition cannot
   be supported, retrain without those rows and repeat model selection,
   calibration, package validation, and frozen-release evaluation under a new
   candidate identity. Do not delete their attribution while retaining their
   training contribution.
2. In the model card, name the exact Eikos source revision and Qwen ancestry;
   keep Eikos's MIT and Qwen's Apache-2.0 notices, and retain the inherited
   Eikos third-party `NOTICE`. Give linked, source-specific attribution for
   our own fine-tuning data, including the Google Research GoEmotions license
   distinction. Describe the research use as an intended use, not as a
   substitute for a clear distribution license.
3. State that the overlap screen covered the exact pinned TRAIN and prompt
   hashes and was approximate for paraphrases. Avoid a blanket claim that the
   model was trained on no evaluation-related material; Eikos/Qwen pretraining
   exposure was not independently audited, and FLUTE is a shared task family.
4. Re-run the mechanical package and rights checks on the exact public upload
   artifact. Record the final reviewer and decision without changing frozen
   evaluation inputs or retroactively altering the pre-key protocol.
