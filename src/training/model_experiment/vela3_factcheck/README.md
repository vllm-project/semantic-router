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
Reports show results at the neutral 0.5 threshold, at a threshold selected only
on `dev`, and at the deployed `vela3-fast` preset threshold of 0.9999. A change
in calibration needs a candidate-specific threshold before promotion.
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

Start both workers to mirror the replay with `python -m "$entry.serve_check"
shadow "$RELEASED_ENDPOINT" "$CANDIDATE_ENDPOINT"`. The released worker supplies
the primary results. Promote only after the predefined gate passes.
Rollback by restoring the released configuration and checking its artifact
identity and decision responses.

## Reference run

The primary `vela3-fast` run used 5,248 audited suite rows, 638 authored rows,
and 100 paired training entries. Four suite training rows overlapped held-out
prompts and were removed from the training input. The audit also checked both
sets of evidence probes against the training input.

| Metric | Released | Candidate |
| --- | ---: | ---: |
| Test ROC AUC | 0.673 | 0.934 |
| Test Brier score | 0.381 | 0.098 |
| Test recall at 0.5 | 387/400 | 352/400 |
| Test false positives at 0.5 | 303/400 | 49/400 |
| MGSM false positives at 0.5 | 1,991/1,991 | 72/1,991 |
| New evidence pairs ordered correctly | 1/8 | 7/8 |
| Both labels correct on the new pairs at 0.5 | 0/8 | 3/8 |
| Local `/v1/decisions` replay p95 | 86.5 ms | 86.1 ms |

The HTTP replay used the same CUDA device and settings, with 80 timed requests
per package after warmup. The multi-question requests covered FactCheck,
domain, feedback, PII, and unsupported claims. The latter four answers matched
across the three captures, and restoring the released package reproduced its
entire responses. The 0.9999 released preset threshold flagged zero positives
on the candidate test scores. A candidate deployment needs its own explicit
FactCheck predicate.
