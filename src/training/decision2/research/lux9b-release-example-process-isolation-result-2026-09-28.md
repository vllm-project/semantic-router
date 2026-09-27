# Lux 1.0 release example: isolated-process diagnostic result

The gold-free intervention preregistered in
`lux9b-release-example-process-isolation-prereg-2026-09-28.md` has completed.
**Process reuse alone is falsified; the Lux 1.0 v3 control remains HOLD.** No
typed FINAL, CSS15, JevBench or protected answer key was used. The frozen
numerical gate remains 0.02 with zero categorical changes.

## Fixed inputs and execution

The own published Lux package revision, bundle manifest, five-answer release
example, runtime image and unmodified native collector match the hashes in
the pre-registration. The collector SHA-256 was
`b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce`.
Both new prompt payloads matched the published `state`/`questions` and the
prior combined-process canonical input hashes. Both isolated predictions
attested the pinned package revision and reported an exact qualified runtime,
with an empty runtime-differences map. The released example and new outputs
also agree on model name, question count and input-token usage.

An idle GPU was verified immediately before each launch. The same pinned
runtime and native collector ran one request per fresh process. The first
process completed at 39.138 seconds; both completed at **80.866 wall/GPU
seconds, 0.0224628 GPU-hour**, under the 300-second cap. Their private
prediction SHA-256 values are:

| Request | Prediction SHA-256 |
| --- | --- |
| `model_card` | `806082d298074ec3579df65f90e15ad1f49a3145603f979c52215c4dafb167d4` |
| `usage` | `923536bbf4d0f0a971e62e4aca4a9ae74828817e0b1ebd8aeeddae5155111a1c` |

The joined two-record prediction SHA-256 is
`d0894ebb0d3167d7631c8553069a5ef3752eea5216c2d988189f13793e7beb56`.
The GPU was released. Raw requests, predictions, logs and private machine
paths remain outside the repository and gist.

## Falsification and remaining question

| Native answer | Kind | Prior combined vs published maximum drift | Fresh isolated vs published maximum drift | Fresh vs combined |
| --- | --- | ---: | ---: | ---: |
| `model_card/route` | Choice | 0.0027770114 | 0.0027770114 | 0 |
| `model_card/refund_requested` | Noul | 0.0031171527 | 0.0031171527 | 0 |
| `usage/owner` | Choice | 0.0055232821 | 0.0055232821 | 0 |
| `usage/active` | Noul | 0.0018864061 | 0.0018864061 | 0 |
| `usage/severity` | Score | **0.0212842536** | **0.0212842536** | **0** |

All five categories match the published example, but the unchanged verifier
returns `Published release example mismatch` on the isolated output because
the Score probability exceeds 0.02. Isolating processes changes **none** of
the five predictions numerically. The earlier mismatch therefore cannot be
explained by running the two requests sequentially with one model instance.
The next likely area is numerical backend or original example provenance
below the published runtime version/profile granularity; this experiment
does not identify a kernel or prove a specific cause. The published package
and current collector code, request inputs and reported runtime versions
match. No threshold adjustment or formal rerun is justified by this result.

The CPU preflight initially looked for the already mirrored collector under
the wrong local mirror directory and stopped before GPU use. The exact
original collector path and hash were then verified before the one GPU
intervention. No additional candidate or GPU retry was performed.
