# Demo category routing

The source of truth is
[`../config-openshift.yaml`](../config-openshift.yaml). Its domain decisions map
request categories to the two demo aliases:

| Model alias | Domain decisions |
| --- | --- |
| `Model-A` | biology, chemistry, computer science, economics, engineering, history, math, other, physics |
| `Model-B` | business, health, law, philosophy, psychology |

The chemistry, math, and physics decisions request reasoning for `Model-A`.
Other listed decisions disable it. The `other` decision has lower priority and
acts as the broad fallback in this demo config.

## What the score means

The domain classifier returns a distribution across configured labels. A score
is useful for comparing labels in that request; it is not a calibrated promise
that the answer is correct. Classification and routing also depend on the
model artifact and Router version, so saved scores from an earlier run are not
part of this mapping.

Use [`curl-examples.sh`](curl-examples.sh) for a live classification check and
record the current config, model revision, and Router image when comparing
results. A small list of demo prompts is not an accuracy benchmark.

To change the mapping, edit the corresponding decision in the `default`
Recipe. Model cards and connections remain under top-level `models`; the
Entrypoint's decision assignments select those Models. Keep assignment names
and backend served-model names aligned.
