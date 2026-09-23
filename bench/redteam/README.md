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

The loader accepts `Goal`, `goal`, `prompt`, `behavior`, or `text` fields. It
strips surrounding whitespace and rejects a corpus with no non-empty prompts.

## Loader validation

```bash
python3 -m pytest bench/redteam/test_datasets.py -q
```

The tests use benign local datasets with the published column names and the real
dataset loader. They cover legacy input fields and rejection of empty input
before model loading. These checks do not measure classifier recall or evasion
rates; model-backed results require a separate run with a named artifact and
dataset revision.

## Adding a signal

Implement the `Scorer` protocol, which is a single `score(texts) -> list[float]`
returning the probability of the positive class. `TransformerScorer` covers any
local sequence-classification artifact. The attack itself never sees the model.
