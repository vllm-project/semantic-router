# GLiNER2.5-Decide native classification baseline

The baseline uses the official [GLiNER2 classifier implementation](https://github.com/fastino-ai/GLiNER2/tree/55656fbfa01d3d4a77485e1a1eeeaf682990ccdf/gliner2/classification) and the pinned [GLiNER2.5-Decide checkpoint](https://huggingface.co/fastino/GLiNER2.5-Decide/tree/7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6). Both are Apache-2.0. The checkpoint's safetensors contain 486,444,053 float32 parameters; its model card's 340M designation is not the measured checkpoint parameter count. The GLiNER2 [paper](https://arxiv.org/abs/2507.18546) describes the schema-driven encoder family, but published model-card benchmarks use a different dataset and are not interchangeable with Decision's unified panel.

`inference.gliner25` projects one Decision question into one exclusive native classification schema. The state is the document text; the question instruction and option descriptions use GLiNER's native schema fields. Valid option keys remain native schema labels. Reserved schema-marker characters trigger a stable alias or punctuation substitution; an aliased original key is retained in the text and restored in the answer. The official classification scorer supplies logits and its exclusive softmax probabilities. Choice uses the highest-probability option, Noul uses the native probability of `yes`, and Score uses the native ordinal distribution's expectation. Every native label must be present; a dropped option is a hard error. No gold is read by the collector.

The projection is an explicit compatibility layer. GLiNER2.5-Decide was trained for classification, not Decision's typed state-machine tasks. The collector checks the official processor's exact encoded length against the encoder's 512-position limit before scoring. It records overflows as invalid answers, with the length and limit, rather than silently truncating a long state. It records `context_policy=native-tokenizer-default` and leaves usage null rather than inventing token counts. The shared scorer counts malformed outputs as invalid. The DEV and CSS pilot panels have at most 4 and 7 Choice options respectively; larger public JevBench rows require their own observed label-alignment check.

Build `inference/Dockerfile.gliner25-rocm` from a compatible ROCm PyTorch base image with the pinned source commit. It creates a separate runtime because its `transformers==4.48.3` dependency may conflict with newer vLLM packages in a training image. The base image remains unchanged.

```bash
# Set DECISION_ROCM_BASE_IMAGE, DECISION_GPU_INDEX and DECISION_WORKSPACE first.
docker build --build-arg BASE_IMAGE="$DECISION_ROCM_BASE_IMAGE" \
  -t gliner25-native:local -f inference/Dockerfile.gliner25-rocm .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e ROCR_VISIBLE_DEVICES="$DECISION_GPU_INDEX" -e PYTHONPATH=/work/source -e HF_HUB_OFFLINE=1 \
  -v "$DECISION_WORKSPACE":/work gliner25-native:local \
  python3 -m inference.gliner25 \
  --model-path /work/models/GLiNER2.5-Decide --input /work/dev.prompts.jsonl \
  --output /work/gliner25-dev.predictions.jsonl
```

The collector verifies the exact checkpoint revision and SHA256 of the weights, config, tokenizer, tokenizer config, and special tokens. Its output records those identities, the official library commit, adapter version, and a per-row input fingerprint. `--resume` checks all prior receipts before continuing.

After a panel is complete, `python -m inference.gliner25_receipt` checks every prompt/prediction pair, model identity, option coverage, and explicit context overflow. It writes a JSON manifest with input and output SHA256, exact source code SHA256, runtime versions, valid count, and invalid reasons. A one-row check against the model card's `AutoExtractor.classify_text` path gave the same class and probability to within `2e-7` with the native schema projection.

## Unified public-panel pilot

| Panel | Correct / total | Valid | Finding |
| --- | ---: | ---: | --- |
| Synthetic DEV | 652 / 1,600 (40.75%) | 1,600 | Choice 227/800; Noul 208/400; Score 217/400 |
| CSS human pilot | 561 / 1,430 (39.23%) | 1,363 | 67 long discourse rows exceeded 512 positions |
| JevBench public only | 116 / 231 (50.22%) | 175 | Easy 48/48, standard 46/72, hard 22/111; 56 hard rows exceeded 512 positions |

The public JevBench result is not the sealed official composite. On this same public subset, pinned Eikos-4B scored 198/231 and the Decision 2.0 Eikos development adapter scored 195/231. Even perfect answers on all 56 GLiNER overflows would yield 172/231 (74.46%), below either Eikos result. The CSS analogue is at most 628/1430 (43.92%) against Eikos clean-v1's 788/1430 (55.10%). Long-text chunking alone therefore cannot explain the gap. The human pilot also exposed class imbalance in predictions: only 5 discourse rows were predicted `elaboration` and 6 implicit-hate rows `threatening`, although the held-out gold is balanced across these classes. This baseline informs encoder research but does not support an open-source SOTA claim for Decision's mixed typed tasks.
