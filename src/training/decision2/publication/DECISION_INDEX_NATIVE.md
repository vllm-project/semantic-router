# Decision Index native Engine for sealed Decision 2.0 packages

`publication.decision_index_native_engine:NativeDecisionIndexEngine` implements
the public [Engine contract](https://github.com/apolinario/decision-index/blob/19ad28ec9485493cc4f7fc07d91c178f948e6434/docs/engines.md)
through a sealed Decision 2.0 package's `Decision2.system_one` method. It
accepts Choice and Noul only and returns their original calibrated native
probabilities. It never invokes the kit's causal-LM token scorer or projects a
Score head into Noul/Choice. The caller's state, question IDs, instructions,
option order, option keys, and descriptions reach the sealed package unchanged.

Set the private `DECISION2_PACKAGE_DIR` and `DECISION2_BASE_DIR` environment
variables to the exact package and pinned base snapshot on the evaluation
machine. Pass the package's exact `MODEL_MANIFEST.json` SHA-256, model ID and an
explicit BF16 GPU device as engine options. The package and base are verified
by its own loader. Filesystem paths are kept out of the kit's logged engine
options and provenance. Keep the runner's raw results private, as its traceback
or benchmark payload can contain restricted data or local paths.

```text
PYTHONPATH=<Decision2-source-root>:<Decision-Index-kit-root> \
python -m decision_index run \
  --engine publication.decision_index_native_engine:NativeDecisionIndexEngine \
  --option model_id=llm-semantic-router/dev-2.0-27b \
  --option package_manifest_sha256=<64-hex-digest> \
  --option device=cuda:0 \
  --rows <gold-free-runner-rows> --out <private-output-directory>
```

The bridge preflights the *whole request*. Incompatible native shapes (such as
structured instructions/descriptions, more than 255 options, or unsupported
Noul criteria) and a package `max_length_exceeded` refusal become the kit's
`unsupported` status for the original request. No question, option, or source
text is omitted or truncated. Missing answers, malformed probability maps,
nonfinite probabilities, incorrect native identities, and other inference
defects remain `error`; the kit's retry/stop policy applies. This currently
means some Index rows may be unsupported by the immutable 27B package even
when another entrant's inference API can process them. Report those counts and
reasons with every result. In particular, this bridge does not silently
serialize structured fields into strings, because that would change the
evaluated model's already sealed inference contract.

The public reproduction kit at commit `19ad28ec9485493cc4f7fc07d91c178f948e6434`
implements **edition 0.2**, while the current Space board publishes **0.2.1**.
An 0.2 compatibility run can validate the Engine wire format, not yield an
0.2.1 score. Before a 0.2.1 full run, pin and verify the exact 0.2.1 request
corpus and exclusions, the 38-benchmark scorer including altered native
metrics/weights, the aggregate score against published rows, and the final
package/source inference parity. A partial run or an 0.2 kit score must not be
shown as a 0.2.1 rank or Pareto point. The public Space currently provides
aggregate JSON but no complete 0.2.1 per-request rows or executable scorer;
the [protocol audit](../research/decision-index-v021-audit-2026-09-27.md)
records the exact version gap.
