# Residual-failure escalation classifier

Offline training scaffold for [#3282](https://github.com/vllm-project/semantic-router/issues/3282).

The classifier estimates the probability that a request on its normal route
will fail in a way a stronger model would have avoided. Its output is a typed
routing signal only. It never selects or escalates a model, and it never
overrides authorization, safety, residency, context, capability, candidate, or
budget policy.

> **Status: test evidence only.** Everything here runs on synthetic fixtures.
> A model trained on them is not a qualified or shippable artifact. Nothing in
> this directory changes live routing.

## Files

| File | Purpose |
| --- | --- |
| `fixtures.py` | Seeded synthetic rows shaped like `shadowdataset.Example`, with typed features, missing-data states, and synthetic verdicts |
| `train.py` | Train on `train`, calibrate and pick the threshold on `calibration`, report held-out metrics on `test`, write a JSON artifact |
| `pyproject.toml`, `uv.lock` | uv project and lock (CPU only, no GPU needed) |
| `requirements.txt` | Python dependencies for `pip` users (CPU only, no GPU needed) |

## Quick start

This directory is a uv project (`pyproject.toml` and `uv.lock`). Create the
environment from the lock, then run the scripts through it:

```bash
uv sync --locked

uv run python fixtures.py --rows 5000 --seed fixture-seed-1 --out synthetic.jsonl
uv run python train.py --data synthetic.jsonl --out artifact.json
```

To run the tests, use the same command as `make test`, from the repository root:

```bash
uv run --project src/training/model_classifier/escalation_risk \
  python -m unittest discover -s src/training/model_classifier/escalation_risk/tests -p 'test_*.py'
```

`requirements.txt` stays for anyone who installs with `pip`.

Generated data and artifacts stay outside Git.

## Input rows

Each row mirrors one example of the shadow comparison manifest
(`src/semantic-router/pkg/shadowdataset`): `id`, `input_digest`, `split`,
`primary`, `shadows`, `lineage`. Splits use the same seeded hash as
`shadowdataset.Policy.splitFor`. Rows carry digests, never prompt or response
text.

Two fields are not in the manifest yet:

- `features`: content-minimized routing facts. Every feature has a `status`:
  `present`, `absent`, or `not_applicable`. Nothing is silently imputed. The
  trainer encodes a missing value as `0` plus an explicit `<name>:missing` flag.
- `verdict`: a stand-in for the judging step of
  [#3280](https://github.com/vllm-project/semantic-router/issues/3280), which
  has not landed. Real verdicts replace it later.

## Labels

| Verdict | Label |
| --- | --- |
| `primary_failed_shadow_ok` | 1 (escalation would have helped) |
| `both_ok`, `primary_ok_shadow_failed` | 0 |
| `both_failed`, `tie`, `abstain` | excluded and counted, never guessed |

## Evaluation

The threshold is chosen on the calibration split to reach a target recall
(default 80%). It is never tuned on the test split. The test report compares
the classifier against three fixed baselines: never escalate, always escalate,
and escalate on low domain confidence.

Reported metrics: false-negative rate, false-positive rate, escalation rate,
AUROC, Brier score, and expected calibration error (ECE).

## Not in scope yet

- Training on real shadow comparison data (waits for #3280 judging)
- Provenance manifests through `src/training/model_eval/provenance`
- Registration in `model_artifacts.json` or any runtime signal
