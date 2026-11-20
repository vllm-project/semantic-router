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
decision taken before any attempt. An id that appears twice in either score
file is an error naming the file and line, so the order of lines cannot pick
the score.

## Policies

| Policy | What runs | Cost |
| --- | --- | --- |
| static | the cheap or the strong model on every task | that model |
| static mix | a random per-task mix of the single models (upper concave hull) | interpolated |
| escalate | the cheap model, then the strong one when score > t | both when escalated |
| retry | the cheap model, then another cheap attempt (`--retry`) when score > t | both when retried |
| boundary | the strong or the cheap model, chosen from the task score | one |
| oracle | the cheap or the strong model per task, knowing both outcomes, the most resolve per dollar first | one, interpolated between tasks |

Escalation restarts the task, so a rejected attempt is still paid for.

The oracle is an upper bound on every policy over the two models at every budget: no escalation or boundary choice reaches a higher resolve rate for the same cost.

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

The repository ships no scores. With d3-turbo's probability that a stronger
model is needed, read from the finished trial (instruction, commands, last
outputs and final message), the report for deepseek-v3.2 and Claude Opus 4.6
over 373 trials of 87 tasks is:

```text
   score AUROC for the cheap attempt failing: overall 0.856, within task 0.737 over 118 pairs
   policy       $0.02  $0.05  $0.1   $0.2   $0.3   $0.5    | utility minus best single model, cross-validated [95% interval]
   static mix   0.414  0.428  0.452  0.501  0.549  0.623
   escalate     0.418  0.445  0.489  0.553  0.614  0.643   | V=$1 -0.016 [-0.044, +0.007]  V=$2 +0.103 [-0.034, +0.243]  V=$5 +0.156 [-0.026, +0.329]
```

With gpt-5-mini as the cheap model the same judge reaches 0.683 and escalation
loses to the best single model at every value. With 87 tasks the intervals
cross zero.

## Other recordings

Any set where several models ran the same tasks fits, for example the
SWE-Router trajectories (arXiv 2607.00053) or the outcomes of a sr-bench run of
each candidate model. Those trajectories carry no license tag, so nothing here
downloads them. `--baselines` adds further single models to the static mix and
to the best single model, and `--quantiles 41` matches the threshold grid of
the original SWE-bench analysis.

Absolute resolve rates on SWE-bench Verified are inflated by contamination.
Compare policies on the same tasks rather than reading them as model quality.

## Candidate routing from per-candidate estimates

`candidate_eval.py` replays the routing rules of CONCUR
([CONCUR: A Framework for Continual Constrained and Unconstrained
Routing](https://arxiv.org/abs/2512.09386), Chen, Li, Roth, Cafarella, Madden,
Andreas; [project page](https://peterbaile.github.io/concur/),
[code](https://github.com/peterbaile/concur)) on the same rows. Each candidate
is a model with a reasoning setting, and each has two estimates per task: the
probability that it answers correctly and its cost. It is the offline
evaluation step of
[per-request strategy routing](../../../../website/docs/proposals/concur-strategy-routing.md)
(#6772) and needs NumPy only.

Estimates are JSONL, one line per row and candidate:

```json
{"id": "q-0017", "candidate": "qwen3-8b-high", "success_probability": 0.86,
 "estimated_cost": 0.00041, "expected_output_tokens": 1180}
```

`expected_output_tokens` is optional; the token error is reported when it and
the rows' `output_tokens` are both present. Candidates are estimated
independently, so a file per candidate can be concatenated. A repeated id and
candidate pair is an error naming the file and line. Rows missing any
candidate's outcome or estimate are skipped and counted in the report.

| Policy | What runs |
| --- | --- |
| utility | per task, the candidate with the highest `v x p - c` on the estimates (CONCUR's unconstrained rule) |
| local | per task, the highest `p` among candidates with `c <= b`, or the lowest `c` when none fits (CONCUR's local constrained rule) |
| global | the assignment with the highest total `p` whose total `c` is at most `b` per task over a batch of tasks (CONCUR's global constrained rule, a dynamic program) |
| oracle | any candidate per task, knowing every outcome; an upper bound at every budget |

The report gives:

- Each candidate alone, and the static mix of all candidates and `--baselines`
  (for example a recorded MoM target), as in `policy_eval.py`.
- Per candidate: the Brier score and expected calibration error of
  `success_probability`, its AUROC for solving, and the mean absolute error and
  bias of `estimated_cost`.
- The utility policy's resolve rate at each mean budget per task, sweeping `v`,
  and its utility `V x resolved - cost` minus the best single model's, with `v`
  set to `V` (plug-in, which assumes calibrated estimates) and with `v` chosen
  by task-grouped cross-validation.
- The local and global policies at each budget. Both apply the budget to
  estimated cost, so the report also gives the mean recorded cost and, for
  local, the share of tasks whose recorded cost exceeds the budget.

The global dynamic program keeps each task's cheapest estimate exact and rounds
what another candidate adds to it up to units of `budget / --dp-resolution`, so
the chosen assignment never exceeds the budget on the estimates. Rounding can
rule out an upgrade that would otherwise fit, so the choice can fall short of
the best assignment within the budget. Whether a batch fits at all is decided
before any rounding, with a tolerance for floating-point error that does not
depend on `--dp-resolution`: the report counts a batch as infeasible only when
its cheapest estimates alone exceed the budget, and such a batch takes the
cheapest estimate on every task. Rows are solved in consecutive batches of `--dp-batch` rows in file
order. Batching on the request path is out of scope for the Router; this is
offline evaluation only.

### From sr-bench

Register one target per candidate, with its reasoning setting frozen in the
target's `request_params`, and run them on the same quick/dev tasks in a priced
live run (`cost_policy: require_priced`, the default). Then:

```bash
vllm-sr benchmark export DEV_RUN_ID --output training-matrix.json
python prepare_sr_bench.py --export training-matrix.json --out rows.jsonl
python candidate_eval.py --rows rows.jsonl --estimates estimates.jsonl
```

A row is one case, and its attempts are keyed by target id; an unpriced target
is an error rather than free. The same rows work with `policy_eval.py`. Export
accepts dev runs only, so these results guide tuning: a quality or savings
claim still needs the live holdout comparison sr-bench prescribes.

The repository ships no estimates. Running with a subset of `--candidates` and
then the full set shows what adding a candidate changes; since each candidate
is estimated on its own, the other candidates' estimates stay the same.
