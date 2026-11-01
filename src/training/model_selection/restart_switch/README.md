# Switching models at a compaction or a cleared conversation

Does a coding agent's run keep its success if it moves to another model at a
restart, and is that switch cheap enough to build? A restart here is a
compaction or a cleared conversation. This is the measurement for
[#5540](https://github.com/vllm-project/semantic-router/issues/5540). It does
not add a re-route toggle.

A restart is a cheap point only relative to an ordinary turn. The prompt is
mostly recomputed anyway, but the cached prefix that is still there, usually
the system prompt and tools, is lost if the model changes. [#5374](https://github.com/vllm-project/semantic-router/issues/5374)
restarts a finished attempt. [#4080](https://github.com/vllm-project/semantic-router/issues/4080)
switches between ordinary turns. Neither is a restart inside a run.
[#5361](https://github.com/vllm-project/semantic-router/issues/5361),
[#5360](https://github.com/vllm-project/semantic-router/issues/5360), and
[#5537](https://github.com/vllm-project/semantic-router/issues/5537) are still
open, so this run has no Router replay or live trial to score.

The analysis calls no model and needs only the Python standard library.

## Detection

Two rules read the request, and a third keeps the prefix-reuse heuristic as a
cross-check. All three are applied to main-agent calls. A subagent call is a
different stream.

- **Marker.** The request's `type` or `item_type` is `context_compaction`,
  `compaction` is true, or `items`, `input`, `content`, or `messages` contains
  an item of that type. Block ids are not a marker. This capture has no prompt
  text. Its request types are `s`, `n`, and `subagent`. `s` is the ordinary
  main call. `n` sometimes extends the previous prompt and sometimes does not,
  so neither code is a compaction marker. Codex's `context_compaction` item
  and Claude Code's `/compact` bytes are not in these fields.
- **Prefix extension.** The main request does not start with the entire
  previous main request. A rewind and an edited history break the prefix too.
- **Continued restart.** A prefix break whose next main request continues it,
  and whose prompt is at least 4,096 tokens. Continuation is the prefix-reuse
  check: the next request's leading blocks are drawn from this request's
  block set and cover at least 90% of this request. A prefix break the next
  request does not continue is rewind- or edit-shaped. A break that is
  continued but shorter than 4,096 tokens is counted separately.

The recorded sessions have no compaction, rewind, or edit labels, so they
cannot give a true error rate. On a labeled fixture of 11 turns (one marked
compaction, one cleared conversation, one short clear, one rewind, one edit
the next turn continues, and ordinary extensions) the rates are:

| Rule | False positives | False negatives |
| --- | --- | --- |
| Marker | 0/8 | 2/3 |
| Prefix extension | 2/8 | 0/3 |
| Continued restart | 1/8 | 1/3 |

The marker misses the cleared conversation and the short clear. The prefix
rule flags the rewind and the edit. The continued-restart rule flags the edit,
because the next turn continues the edited prompt, and it misses the short
clear. Those counts are the fixture, not the production capture. On the
capture, the prefix rule and the continued-restart rule disagree wherever a
prefix breaks and the turn is not a continued restart. That disagreement is
not an accuracy.

## Frequency and cache cost

[semianalysisai/cc-traces-weka-with-subagents-060526](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-060526),
Apache-2.0, pinned at revision `8a5df339ba3c31a48dd79a2f5ceb81e08679266b`.
Each prompt is a list of hash ids. In this pin `block_size` is 64, so a
cached prefix of *k* blocks is 64*k* tokens. The file is fetched, checked
against SHA-256 `954157040cbae47fef3bec70532bdb3406c75ed894e207e502461d70fa06639c`,
and not committed. There is no task outcome.

Staying on the same model loses none of the cached prefix. Switching loses
it, because the other model does not hold that prefix. The uncached suffix is
paid either way. The trace has no run id. A run here is the derived span of
main turns between continued restarts, so a session with *R* continued
restarts has *R* + 1 runs.

The half-drop cross-check is the restart from
[#5357](https://github.com/vllm-project/semantic-router/issues/5357) / PR
[#5371](https://github.com/vllm-project/semantic-router/pull/5371): a main
turn that reuses under half its prompt, is at least 4,096 tokens, is at most
half the longest long prompt so far, is not a model change, and is continued
by the next turn. Short calls do not set that length. It is not the detection
rule above. Block ids still cannot tell a compaction from a cleared
conversation.

## Production counts

Measured at the pinned revision. Every session has `block_size` 64.

| Measure | Value |
| --- | --- |
| Sessions | 336 |
| Main turns, including first turns | 79,036, including 336 first turns |
| Marker turns | 0 |
| Prefix breaks | 34,779 |
| Continued restarts | 5,796, in 334 sessions |
| Short continued breaks | 1 |
| Not continued (rewind- or edit-shaped) | 28,982 |
| Prefix rule disagreements | 28,983 |
| Derived runs | 6,132 |
| Continued restarts per session | 17.25 |
| Continued restarts per derived run | 0.945 |
| Cached tokens a switch loses at a continued restart | median 103,040, total 938,903,808 |
| Cached tokens a switch loses at an ordinary turn | median 168,064, total 9,388,643,840, over 43,921 turns |
| Cached tokens a stay loses | 0 |
| Half-drop cross-check | 60 turns in 37 sessions, median cached prefix 21,888, 0.0166 of main uncached tokens |

The derived-run rate is continued restarts divided by sessions plus continued
restarts, because each continued restart opens a run. The per-session rate is
the one that is not fixed by that definition.

The half-drop cross-check is the rare restart described for this pin: same 37
sessions, same median cached prefix of 21,888. An earlier pass reported 56
turns and 0.015 of main uncached tokens. This count is the rule pinned by the
unit test, and it is 60 turns and 0.0166. Block ids still cannot tell a
compaction from a cleared conversation.

## Success and cost

Paired rows, one file per direction, look like:

```json
{"stay_ok": 1, "switch_ok": 0, "stay_cost": 0.02, "switch_cost": 0.4}
```

`stay_ok` and `switch_ok` are 0 or 1. Costs are whatever unit the caller
used. When a file is present and has rows, success rates get a Wilson 95%
interval and the mean of `switch_ok - stay_ok` and of `switch_cost - stay_cost`
get a normal 95% interval once there are two rows. This pin has no outcome
file. Both directions have n = 0, so the intervals are omitted. The sample is
too small to decide whether a switch keeps success, in either direction.

The programmed gate returns `go` only when both directions have a
success-difference interval whose lower bound is at least 0. Cost does not
flip the gate; it is reported beside it. Empty directions are `no-go`.

## Recommendation

**No-go.** Do not open a feature issue for a re-route-at-restart toggle.

The marker rule does not fire on this capture, so it cannot find a
compaction here. The prefix rule fires on 34,779 main turns, and 28,982 of
those breaks are not continued, so most of them are not a compaction the
conversation keeps. The continued-restart rule fires 5,796 times, 17.25 per
session, in 334 of 336 sessions. That is not a rare compaction. A switch
there still loses a median 103,040 cached tokens, against 168,064 on an
ordinary turn and 0 if the model stays. The half-drop point is rare, and a
switch there still loses the median 21,888 cached tokens. Neither a stronger
model nor a cheaper one has a success interval. The sample is too small to
decide.

A later sample would have to come from Router replay or a live trial, and it
would have to report the two directions separately. Until both intervals
exclude a success drop, the default of any such toggle would be off.

Today's protection hold would not release itself at a restart. Conversation
scope holds every turn that shares `x-conversation-id`, and session scope
holds `x-session-id`. A compaction does not mint a new id.
[#5363](https://github.com/vllm-project/semantic-router/issues/5363) would
hold each subagent on its own model; that hold does not exist yet, and the
optional agent header does not change the hold key. A restart of the main
run would not be a reason to release a subagent's model.

## Run

```bash
python src/training/model_selection/restart_switch/restart_switch.py \
  download /tmp/sa-traces
python src/training/model_selection/restart_switch/restart_switch.py \
  report /tmp/sa-traces/traces.jsonl --json
```

Optional `--stronger rows.jsonl` and `--cheaper rows.jsonl` score paired
outcomes. They are not part of the published result.

## Tests

```bash
python -m unittest discover \
  -s src/training/model_selection/restart_switch/tests -p 'test_*.py'
```

The tests use synthetic sessions. They pin the fixture error rates, the
cached prefix a switch loses, the half-drop cross-check when a short side
call must not reset the conversation length, the Wilson and cost intervals,
and the go / no-go gate. They do not download the traces.
