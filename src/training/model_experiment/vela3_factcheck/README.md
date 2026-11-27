# Vela3 FactCheck qualification

This experiment adapts the FactCheck branch of `vllm-sr/vela3-fast` at
`7cbefd4b92b57234bd0ffb9872451e81abcdac79`. FactCheck predicts whether a
request needs external verification. A calculation with all inputs supplied
and a request that needs missing private context are both negative examples.
A question about an external fact not supplied in the request is positive.

The released FactCheck branch scores self-contained math problems as positive
even when their answers can be derived from the prompt. The experiment starts
from the complete shared Router adapter, including its upper layers and readout.
It freezes the backbone and upper layers, then trains the LoRA factors and the
marker readout. The objective combines smoothed binary classification with
an evidence pair loss. Within each pair, the same question appears with either
sufficient or irrelevant evidence. Only the exported FactCheck adapter changes.

## Inputs and environment

Install `src/model-runtime` in a Python environment with a CUDA build of PyTorch,
then install `jsonschema` for provenance validation. Run the following commands
from the repository root. Use a separate directory for data, reports, and weights.

- `BASE` points to a regular-file copy of the full pinned [Vela3 model package](https://huggingface.co/vllm-sr/vela3-fast/tree/7cbefd4b92b57234bd0ffb9872451e81abcdac79).
- `SUITE` points to `text/` in [Router Signal Suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite/tree/fa08b2a642df30955ad2ad2206d74050c7f12b5c) at `fa08b2a642df30955ad2ad2206d74050c7f12b5c`.
  Use the FactCheck JSONL files and respect their per-source terms in `sources.json`.
  Dolly carries CC BY-SA 3.0. Search Arena training uses its user prompts only.
- `AUGMENT` is an authored JSONL input. Single rows have `id`, `text`, `label`,
  and an optional `weight`. Pair rows have `id`, `kind: pair`, `neg_text`, and
  `pos_text`. Version the input separately and record its digest.
- `PROBES` is a separate JSONL file with `id`, `text`, `label`, and `slice`.
  Evidence pairs also share a `group`. Keep probe text separate from augmentation.

## Reproduce a candidate

```sh
entry=src.training.model_experiment.vela3_factcheck
mkdir -p "$OUT"
python -m "$entry.data" --suite "$SUITE" --augmentation "$AUGMENT" --probes "$PROBES" \
  --clean-train "$OUT/train.jsonl" --out "$OUT/audit.json"
python -m "$entry.train" --model "$BASE" --train "$OUT/train.jsonl" \
  --augmentation "$AUGMENT" --device cuda:0 --seed 42 --out "$OUT/run"
python -m "$entry.package" --base "$BASE" \
  --adapter "$OUT/run/factcheck_adapter.safetensors" --out "$OUT/candidate"
python -m "$entry.evaluate" --model "$BASE" --suite "$SUITE" --probes "$PROBES" \
  --out "$OUT/baseline.json"
python -m "$entry.evaluate" --model "$OUT/candidate" --suite "$SUITE" --probes "$PROBES" \
  --out "$OUT/results.json" --scores-out "$OUT/predictions.json"
```

The audit compares normalized text, source groups and character 5-grams.
Reports show results at 0.5 and at a threshold selected only on `dev`.
Use the separate MGSM language slices and paired evidence results alongside
aggregate metrics. The suite documents source artifacts in `no_robots` and
WildBench that can make their aggregate AUC optimistic. Both are secondary to
the independent boundary probes and the language breakdown.

## Verify serving and record the result

Serve each package with `vllm-srun serve` on a local port. Capture the released
package, the candidate on the same device and settings, and the restored released
package using `python -m "$entry.serve_check" capture "$ENDPOINT"`. Run
`python -m "$entry.serve_check" compare released.json candidate.json rollback.json`
to verify FactCheck changes and domain, feedback, PII, and unsupported-claim
responses stay equal. This comparison also reports HTTP latency for the same
requests. Record the run with `python -m "$entry.record" --help`, using the
immutable input and artifact receipt revisions, and validate the bundle.

```sh
PYTHONPATH=src/training/model_eval python -m provenance.cli validate "$BUNDLE"
```

Use a separately configured worker for shadow traffic while the released worker
continues to supply the answer. Promote only after the predefined gate passes.
Rollback by restoring the released configuration and checking its artifact
identity and decision responses.
