# Paid-exploration stream

Replay for [#5374](https://github.com/vllm-project/semantic-router/issues/5374):
after a cheap coding agent finishes, when does paying to restart the task on a
stronger model beat the best single model?

This directory is the part of that measurement the Terminal-Bench evaluator
does not cover. [policy_eval](https://github.com/vllm-project/semantic-router/pull/5373)
scores policies when both models' outcomes are already known on the calibration
tasks. That is an upper bound. `explore.py` learns the threshold the way a
deployment would: the strong outcome is visible only on tasks it escalated, and
those tasks are paid for in full. [The judge answers](https://github.com/vllm-project/semantic-router/pull/5375)
for the public Terminal-Bench and TwinRouterBench sources stay in that pull
request.

Nothing here calls a model, trains a judge, or changes routing. A run is one
finished attempt. The cheap transcript is not handed to the strong model.

## Rows

`explore.py` reads the same JSONL as the policy evaluator:

```json
{"id": "repo-1::cheap::1", "task": "repo-1",
 "attempts": {"cheap": {"ok": 0, "cost": 0.02},
              "cheap#retry": {"ok": 0, "cost": 0.02},
              "strong": {"ok": 1, "cost": 0.4}}}
```

Scores are `{"id", "score"}`. A higher score asks for a restart. The same id
with two different scores is rejected with the file and line. An identical
repeat is kept once.

## What the stream does

The threshold starts at never-escalate. Tasks arrive in a seeded order (50
orders by default). Every row of a task uses the threshold from before that
task. A row escalates when its score is above the threshold or when its
exploration draw fires. Exploration is paid: the cheap attempt and the second
attempt are both charged, and the second outcome is the one that is kept.
After the task, the threshold is refit on escalated rows only. A task that was
not escalated never reveals the other model.

Utility is `V * resolved - cost`, with `V` in `$1`, `$2`, and `$4`. The report
is the mean gain over orders against the better of the cheap model and the
strong model run alone. The retry column is not a static baseline.

Hindsight fits one threshold with every outcome visible and applies it with
exploration off. The report labels that gain in-sample. It is the comparison
point in the issue, not a number a deployment can claim.

The retry arm uses the same exploration draws and the same threshold rule, with
`<cheap>#retry` (or `--retry`) as the second attempt. If that column is missing,
the arm is omitted. An earlier SWE-Smith comparison found retry matching
escalation and could not be rerun; this arm is what makes that comparison
rerunnable on a local dump.

```bash
python src/training/model_selection/escalation_stream/explore.py \
  --rows rows.jsonl --scores scores.jsonl \
  --cheap deepseek-v3.2 --strong gpt-5.2 \
  --epsilon 0.1 --orders 50 --values 1,2,4
```

With `--epsilon 0` the stream never escalates, because the threshold has no
observation to move it. Published dollar gains on SWE-bench need the local
trajectories and are not reproduced by the unit tests.

## Local SWE inputs

`swe_local.py` reads a directory of `.jsonl` files the caller already has. It
does not download them. SWE-Router trajectories (SWE-bench Verified and
SWE-Smith) have no license tag. Keep them, and any judge answer that quotes
them, out of git. Compare policies on the same tasks: absolute resolve rates
on SWE-bench Verified are inflated by contamination.

Each record needs `id`, `task`, `instruction`, `actions` (`command`, `output`),
`final_message`, and `attempts`. `diff` is optional.

| View | What the judge reads |
| --- | --- |
| `task` | the instruction |
| `actions-1` .. `actions-4` | the instruction and the first n actions |
| `finished` | commands, the last four actions, and the final message, without the final diff |
| `finished-diff` | the finished state with the diff kept |

A diff that is also the last tool output is replaced with
`[withheld: final diff]` in `finished`. Caps match the Terminal-Bench judge
protocol for this study: instruction 6000, command 300, action 600, final
message 3000, state 24000. A command list that still overflows keeps its first
and last quarter.

Questions in `questions/swe_end.json` are the fixed ones from the issue: will
the attempt resolve, and is the outcome that the agent resolves it, that a
stronger model starting from scratch is needed, or that neither does.

```bash
python src/training/model_selection/escalation_stream/swe_local.py \
  requests --trajectories /path/to/local-dump --out requests.jsonl
python src/training/model_selection/escalation_stream/swe_local.py \
  rows --trajectories /path/to/local-dump --out rows.jsonl
python src/training/model_selection/escalation_stream/swe_local.py \
  checkpoint --answers new-answers.jsonl --checkpoint answers.jsonl
python src/training/model_selection/escalation_stream/swe_local.py \
  scores --checkpoint answers.jsonl --out-dir scores/
python src/training/model_selection/escalation_stream/swe_local.py \
  auroc --rows rows.jsonl --scores scores/resolve_failure.jsonl \
  --cheap gpt-5-mini --strong gemini-3-pro
```

The checkpoint is append-only `{id, view, probs}`. A completed `(id, view)` is
skipped. The same key with different probabilities raises, with the file and
line. `scores` writes `P(stronger_needed)` and `1 - P(resolve)`. Noul
probabilities are ordered `false`, then `true`. The outcome list follows the
criteria order in `swe_end.json`. `auroc` reports cheap-attempt failure
(overall and within task) and whether the strong model rescues a cheap
failure. Ties count half.

Answer the requests with the d3-turbo judge at the revision named in the issue
(`17a01e225d5e`). This tree does not run that model.

## Tests

```bash
python -m unittest discover \
  -s src/training/model_selection/escalation_stream/tests -p 'test_*.py'
```

The tests use synthetic rows. They check the paid cost, the information
constraint, hindsight, the retry comparison, duplicate scores, the judge views,
and checkpoint resume.
