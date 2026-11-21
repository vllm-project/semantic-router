# Prefix reuse in production coding-agent sessions

How much of a coding agent's prompt a KV cache can serve, and where in a session a model change is cheap.
The analysis reads production Claude Code sessions with subagents and reports, for the main agent and for
each subagent, the share of prompt tokens that is a prefix of an earlier call on the same stream, how much a
subagent's first call shares with its parent and with earlier sibling subagents, how many tokens subagents
carry, and how often a request arrives after its stream sat idle longer than a cache TTL.

It calls no model and needs only the Python standard library.

## Data

[semianalysisai/cc-traces-weka-with-subagents-060526](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-060526),
Apache-2.0, pinned at revision `8a5df339ba3c31a48dd79a2f5ceb81e08679266b`: 336 sessions, 93,994 requests.
Each request lists its prompt as 64-token blocks with hash ids, token counts and timing, and marks
subagent calls. There is no prompt text and no task outcome. The traces are fetched, checked against their
SHA-256, and never committed.

Block ids are chained: in these traces no id appears at two positions (4,735,268 distinct ids across the
336 sessions). A seen id therefore means its whole prefix was seen, and the leading run of a request's
blocks already seen on a stream is exactly the prefix a cache keyed by prefix could serve.

## Run

```bash
cd src/training/model_selection/agent_prefix_reuse
python prefix_reuse.py download /tmp/sa-traces          # 1.1 GB at the pinned revision
python prefix_reuse.py report /tmp/sa-traces/traces.jsonl
python prefix_reuse.py report /tmp/sa-traces/traces.jsonl --json
```

## Results at the pinned revision

| Measure | Value |
| --- | --- |
| Sessions, with subagents, subagent groups | 336, 88, 499 |
| Main agent: prompt tokens that are a cached prefix of the same session (78,700 turns) | 0.980 of tokens, median turn 0.989 |
| Subagent first call: median prompt | 24,704 tokens |
| Subagent first call: shared with the parent's stream | 0.092 of tokens |
| Subagent first call: shared with an earlier sibling | 0.634 of tokens, and 42.5% of first calls share at least 90% |
| Subagent later calls: cached prefix of the same subagent (14,459 calls) | 0.943 of tokens |
| Subagent share of input and output tokens | 0.100 and 0.098 |
| Requests after more than 5 minutes idle on their stream | 1,511 of 93,994 (1.6%), median prompt 185,728 tokens, 2.6% of input tokens |
| Requests after more than 1 hour idle | 341 (0.4%), median prompt 220,160 tokens, 0.6% of input tokens |
| Main turns reusing under half their prompt | 3,995, carrying 0.400 of the main agent's uncached tokens |
| of which after a model change | 675 turns, 0.032 of uncached tokens |
| of which short calls under 4,096 tokens | 1,863 turns, 0.020 |
| of which the first long turn of a session | 267 turns, 0.048 |
| of which restarts | 56 turns in 37 sessions, 0.015, median cached prefix 21,888 tokens |
| of which other | 1,134 turns, 0.286 |

Calls by model: main agents 67,769 on claude-opus-4-8, 4,726 on claude-haiku-4-5, 3,388 on
claude-opus-4-6 and 1,355 on claude-opus-4-7. Subagents 5,343, 4,795, 3,451 and 1,111 on the same models.

## Reading

- A model change inside a main agent's run discards a cache that holds almost the whole prompt, so a run
  belongs on one model and on the replica that holds its prefix.
- A subagent starts from a fresh prompt that shares a tenth of its tokens with the parent, so its first call
  is the cheapest point in a session to choose a different model. Siblings share most of their opening
  prompt, so subagents of one run belong on one model, or on a shared cache pool.
- Subagents carry a tenth of the tokens, which bounds what routing them alone can save.
- Cold resumes are rare by count, but each reloads a long prefix, which is where an offloaded KV tier pays.
- A restart is a turn whose prompt is at most half of the conversation so far and that the next turn
  continues, which is how a history compaction or a cleared conversation looks. Block ids cannot tell the
  two apart, and restarts carry 1.5% of the main agent's uncached tokens, so neither is where these
  sessions lose their prefix. The other low-reuse turns are not explained by the block ids either.

The Coding Agents Through the Router proposal cites these numbers.

## Tests

```bash
python -m unittest discover -s src/training/model_selection/agent_prefix_reuse/tests -p 'test_*.py'
```

They pin the prefix arithmetic on small synthetic traces written in the test: the cached prefix stops at
the first unseen block, the main share is token weighted and skips a session's first turn, a subagent's
first call is split between parent and sibling, later calls count only their own stream, and an idle gap
runs from the end of the same stream's previous call.
