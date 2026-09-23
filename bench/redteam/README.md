# Red-team evasion benchmark

Measures how easily a router classifier signal can be talked out of a
detection. The attack is black box: it reads the score the classifier returns
and nothing else, so it needs no gradients, weights, or architecture knowledge.

The runner reports baseline recall, flip rate, suffix length, and query count,
which is the shape a graduation gate needs.

## Attack

`greedy_suffix_attack` appends benign filler words to a prompt the classifier
already flagged. Each round scores every remaining word once and keeps the one
that lowers the jailbreak probability most, stopping when the score falls below
the threshold or when no word improves it. The word pool is mundane on purpose
(`please`, `weather`, `recipe`), so a flip cannot be blamed on the suffix
smuggling in adversarial content of its own.

This is deliberately the weakest useful attack, which makes every number here a
lower bound on what a real adversary achieves.

## Running it

```bash
python3 -m bench.redteam.evaluate \
    --model models/mmbert32k-jailbreak-detector-merged \
    --dataset jailbreakbench \
    --threshold 0.7
```

`--dataset` accepts `jailbreakbench` to pull the 100 harmful behaviors from
the Hub, or a path to a local JSON or JSONL file for an offline run. Pass
`--min-baseline-recall` or `--max-flip-rate` to make the run exit non-zero when
a model misses the bar, which is what turns it into a gate.

## Measured baseline

`llm-semantic-router/mmbert32k-jailbreak-detector-merged` against the
JailbreakBench harmful split, threshold 0.7, on one NVIDIA H100 80GB:

| metric | value |
| --- | --- |
| prompts | 100 |
| detected at baseline | 39 (39%) |
| flipped to benign | 34 of 39 (87%) |
| suffix words to flip, mean | 2.44 |
| suffix words to flip, median | 2 |
| suffix words to flip, max | 10 |
| queries per prompt, mean | 35.2 |

Two results matter here. The detector misses 61 of 100 harmful behaviors before
any attack runs, and of the 39 it does catch, appending a median of two harmless
words defeats it. Raising `--max-words` from 10 to 30 changes nothing, so the
five that survive are a local minimum of the greedy search rather than a budget
limit; a wider pool or a search that tolerates a temporary score increase would
likely take them too.

## Adding a signal

Implement the `Scorer` protocol, which is a single `score(texts) -> list[float]`
returning the probability of the positive class. `TransformerScorer` covers any
local sequence-classification artifact. The attack itself never sees the model.
