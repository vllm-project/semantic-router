# Counterfactual routing policy evaluation

`policy_eval.py` replays routing policies on recorded outcomes. When several
models have each attempted the same tasks and every attempt has a verified
outcome and a dollar cost, a policy such as "run the cheap model, and run the
strong one when a judge rejects the attempt" can be scored on the recordings
without calling any model. It answers whether a policy beats simply mixing
single models, at which budgets, and how much of that survives when its
threshold has to be learned.

It needs NumPy and nothing else. It does no inference: judge scores are an
input you compute elsewhere.

## Inputs

Rows, one per attempt of the cheap model, as JSONL:

```json
{"id": "fix-git::deepseek-v3.2::trial-3", "task": "fix-git",
 "attempts": {"deepseek-v3.2": {"ok": 1.0, "cost": 0.012},
              "deepseek-v3.2#retry": {"ok": 0.75, "cost": 0.011},
              "claude-opus-4-6": {"ok": 0.8, "cost": 0.41}}}
```

`ok` is the verified outcome in [0, 1] (a mean when it stands for several
independent attempts) and `cost` is dollars per attempt. Scores are
`{"id", "score"}` per row id, where a higher score should mean the cheap
attempt needs replacing. Boundary scores are `{"id", "score"}` per task, for a
decision taken before any attempt.

## Policies

| Policy | What runs | Cost |
| --- | --- | --- |
| static | the cheap or the strong model on every task | that model |
| static mix | a random per-task mix of the single models (upper concave hull) | interpolated |
| escalate | the cheap model, then the strong one when score > t | both when escalated |
| retry | the cheap model, then another cheap attempt (`--retry`) when score > t | both when retried |
| boundary | the strong or the cheap model, chosen from the task score | one |
| oracle | escalate exactly the attempts the strong model rescues | both when escalated |

Escalation restarts the task, so a rejected attempt is still paid for.

## Report

- Resolve rate at each budget per task, the best any threshold reaches, next
  to the static mix. A policy is only worth deploying where it is above the mix.
- Utility `V x resolved - cost` minus the best single model's, with the
  threshold chosen by task-grouped cross-validation (every attempt of a task in
  one fold) and a 95% interval from resampling tasks.
- The same gain when the threshold is chosen from the first n tasks of a random
  stream (`--stream 25,50,100`). Both models' outcomes are known on those
  tasks, which a deployment has only where it explores, so this is an upper
  bound on what an outcome loop learns from n tasks.
- The score's AUROC for the cheap attempt failing, overall and within task.
  Within task compares failed and solved attempts of the same task only, so a
  score that only knows which tasks are hard gets 0.5 there.

## Terminal-Bench 2.0

`prepare_terminal_bench.py` builds rows from the public leaderboard
trajectories (`yoonholee/terminalbench-trajectories`, Apache-2.0, pinned
revision) under the terminus-2 scaffold: one row per trial of deepseek-v3.2 or
gpt-5-mini, with the mean of that model's other trials on the task as
`<model>#retry` and each strong model's mean over its own trials. It needs
pandas, pyarrow and huggingface_hub.

```bash
python prepare_terminal_bench.py --out tb2_rows.jsonl
python policy_eval.py --rows tb2_rows.jsonl --cheap deepseek-v3.2 --strong claude-opus-4-6 \
    --scores tb2_end_scores.jsonl --boundary-scores tb2_task_scores.jsonl
```

The judge's answers ship in `data/` (see below). The score files above come
from them:

```bash
python d3_judge.py scores --answers data/tb2_end.d3-turbo.answers.jsonl \
    --question outcome --index 1 --out tb2_end_scores.jsonl
python d3_judge.py scores --answers data/tb2_task.d3-turbo.answers.jsonl \
    --question outcome --index 1 --out tb2_task_scores.jsonl
```

With d3-turbo's probability that a stronger model is needed, read from the
finished trial (instruction, commands, last outputs and final message), the
report for deepseek-v3.2 and Claude Opus 4.6 over 373 trials of 87 tasks is:

```text
   score AUROC for the cheap attempt failing: overall 0.856, within task 0.737 over 118 pairs
   policy       $0.02  $0.05  $0.1   $0.2   $0.3   $0.5    | utility minus best single model, cross-validated [95% interval]
   static mix   0.414  0.428  0.452  0.501  0.549  0.623
   escalate     0.418  0.445  0.489  0.553  0.614  0.643   | V=$1 -0.016 [-0.044, +0.007]  V=$2 +0.103 [-0.034, +0.243]  V=$5 +0.156 [-0.026, +0.329]
```

With gpt-5-mini as the cheap model the same judge reaches 0.683 and escalation
loses to the best single model at every value. With 87 tasks the intervals
cross zero.

## Judge answers

`data/` holds the answers of vllm-sr/d3-turbo (Apache-2.0, revision
17a01e225d5e) for the public sources, so the reports above and the
TwinRouterBench result run without the model:

| File | Rows | State the judge read | Questions |
| --- | --- | --- | --- |
| `tb2_end.d3-turbo.answers.jsonl` | 809, one per Terminal-Bench 2.0 trial in the rows | instruction, commands, last four actions, final message | `questions/end.json` |
| `tb2_task.d3-turbo.answers.jsonl` | 89, one per task | the instruction alone | `questions/end.json` |
| `twinrouterbench.d3-turbo.answers.jsonl` | 970, one per static-track row | the benchmark's own classifier prompt | `questions/twinrouterbench.json` |

Each row is `{"id", "k", "probs"}` with one probability list per question, in
the order of the question's criteria. Thirteen TwinRouterBench rows are longer
than the model's validated length and were answered under its long-context
cap, marked `"long": true`.

`d3_judge.py` regenerates them. `requests` builds the states from the pinned
sources (the Terminal-Bench 2.0 parquet files, a checkout of
harbor-framework/terminal-bench-2 for the instructions, and a checkout of
CommonstackAI/TwinRouterBench, at the commits it names), `answer` runs a d3
package on a GPU, and `scores` writes `{"id", "score"}` files for
`policy_eval.py`. The states it builds are identical to the ones the shipped
answers were computed from.

```bash
python d3_judge.py requests --source tb2 --tb2-tasks terminal-bench-2 --out-dir req
python d3_judge.py answer --model-dir d3-turbo --requests req/tb2_end.requests.jsonl \
    --questions questions/end.json --out tb2_end.answers.jsonl
python twinrouterbench_eval.py --checkout TwinRouterBench
python twinrouterbench_eval.py --checkout TwinRouterBench --constant low
```

On the static track, d3-turbo's arg-max tier scores 56.83 combined, against
63.68 for always choosing the low tier, because 71% of the rows are labelled
low.

## Other recordings

Any set where several models ran the same tasks fits, for example the
SWE-Router trajectories (arXiv 2607.00053) or the outcomes of a sr-bench run of
each candidate model. Those trajectories carry no license tag, so nothing here
downloads them. `--baselines` adds further single models to the static mix and
to the best single model, and `--quantiles 41` matches the threshold grid of
the original SWE-bench analysis.

Absolute resolve rates on SWE-bench Verified are inflated by contamination.
Compare policies on the same tasks rather than reading them as model quality.
