# Decision Index 0.2.1 independent port

This module adds the [Space's 0.2.1 protocol](https://huggingface.co/spaces/multimodalart/jev-decision-index/tree/ed7d683e6626a4a5ace0086b8bdedd8b1cc58cf4)
to the MIT-licensed [public 0.2 reproduction kit](https://github.com/apolinario/decision-index/tree/19ad28ec9485493cc4f7fc07d91c178f948e6434).
It does not modify or claim endorsement from either project. It uses the
kit's pinned native metrics and typed `Engine(state, questions)` contract.
Only public, aggregate benchmark statistics are checked into this directory;
no source examples, model predictions, credentials, or restricted data are.

## Pin and verify

The model-independent protocol in `data/protocol-021.json` pins the Space Git
revision, index/methodology JSON byte hashes, six critical public-kit file
hashes, source corpus hash, area/benchmark weights, chance levels, panel,
request counts and the 717-row exclusion rules. The compact
`data/published-021-summary.json` contains only the 68 entries' published
rounded scores and per-benchmark aggregates for arithmetic parity tests.

The pinned 0.2 kit and frozen 0.2 suite are required for a new run:

```sh
git clone https://github.com/apolinario/decision-index.git /path/to/decision-index-kit
git -C /path/to/decision-index-kit checkout 19ad28ec9485493cc4f7fc07d91c178f948e6434
PYTHONPATH=src/training/decision2:/path/to/decision-index-kit \
  python3 -m external_index021 published-parity
```

The public kit's sample Hub suite name does **not** resolve as an anonymous
public dataset. Rebuild its 0.2 suite from pinned upstream sources in a private
workspace, then provide the four kit files: `selected-rows.jsonl.gz`,
`added-rows.jsonl.gz`, `excluded-questions.json`, and `manifest.json`. The
kit's `Suite.verify` and this port reject changed rows or common exclusions.
The 0.2 source corpus hash is shared with 0.2.1. Rebuilds may need gated
sources and must follow each upstream source's terms. Use the HF CLI on the
authorized SSH environment for Hub interactions; keep restricted source rows
and prediction artifacts private.

The public Space bundle can be independently checked by downloading its
`data/index-v0.2.1.json` and `data/methodology-v0.2.1.json` at the pinned
revision with the HF CLI, then running:

```sh
PYTHONPATH=src/training/decision2 python3 -m external_index021.compile_reference \
  /private/path/index-v0.2.1.json /private/path/methodology-v0.2.1.json --check
```

## Edition and denominator

The Space's `suite.requests=120340` and `suite.scoreable=119898` count the
**base** suite; `suite.added.requests=30419` are seven additional scored
benchmarks, all in the headline. A complete run therefore schedules **150759**
requests and scores **150317** after the 442 common exclusions. Relative to
the 0.2 base, 0.2.1 removes ToolRet 315, BRIGHT 330 and Home 72 requests:
717 altogether. The index counts 38 benchmarks across five weighted areas,
with 13 gold benchmarks weighted 1.2 inside their areas. RouterBench and SGD
stay visible but leave the headline. ACOS uses review-level F1, RAGTruth's
chance is the always-hallucinated F1, and ToolRet/BRIGHT chance is recomputed
per answerable query. ForecastBench retains its Brier-to-skill transform.

Selection operates on the kit's common-exclusion-adjusted rows. ToolRet and
BRIGHT retain complete query groups only when a `scorable_id` has positive
relevance. Home drops 48 test rows whose `state` equals a generated dev row,
then one of each of 24 pairs with identical `state`. The generator recreates
those exact counts. The Space does not publish **which copy** of each Home
pair was kept; their option keys and wording differ. `--home-policy first`
is a deterministic **provisional** default; `last` is available for a
sensitivity check. `explicit` accepts an 88-run-ID keep list if the upstream
maintainer provides one. An explicit list alone still needs provenance and
per-row result comparison before an official-equivalence claim.

If upstream row IDs remain unavailable, run both predeclared policies on the
same frozen model and combine the predictions in one `results.jsonl`; running
`run` once with `first` and again with `last` in resume mode fills the second
copy of each duplicate pair. Then use `sensitivity` to report the two scores
and their range. Neither endpoint is an official 0.2.1 rank.

## Run and score

Once the frozen suite exists, `run` accepts the public kit's native Engine
module/class contract and preserves the selected row IDs. The Decision 2.0
engine must call the **packaged native Choice/Noul inference path**; the kit's
stock causal-LM option-token engine bypasses the trained decision heads.
First run its 86-request compatibility sample and verify full native response
parity; then run this entire selected suite once per frozen package.

```sh
PYTHONPATH=src/training/decision2:/path/to/decision-index-kit \
  python3 -m external_index021 verify-suite \
  --suite-dir /private/path/suite-0.2

PYTHONPATH=src/training/decision2:/path/to/decision-index-kit \
  python3 -m external_index021 run \
  --suite-dir /private/path/suite-0.2 \
  --engine your_module:NativeDecisionEngine \
  --out /private/path/run-021

PYTHONPATH=src/training/decision2:/path/to/decision-index-kit \
  python3 -m external_index021 score \
  --suite-dir /private/path/suite-0.2 \
  --results /private/path/run-021/results.jsonl \
  --out /private/path/run-021/index-021-provisional.json

PYTHONPATH=src/training/decision2:/path/to/decision-index-kit \
  python3 -m external_index021 sensitivity \
  --suite-dir /private/path/suite-0.2 \
  --results /private/path/run-021/results.jsonl \
  --out /private/path/run-021/home-copy-sensitivity.json
```

`score` marks incomplete runs and evaluator errors. Unsupported/abstained
questions stay in the denominator as failures. Its result is labelled an
**independent provisional 0.2.1 reproduction** and `official_equivalent` is
always false until upstream row identity and native per-request parity are
established. The 68-row fixture replay reaches at most 0.0058 index-point
drift from display rounding, but those are **published aggregates**: that
check establishes panel arithmetic only. It cannot prove the ACOS/Home/
retrieval native metrics, identical predictions, latency, or leaderboard
admission. No Decision 2.0 checkpoint has been scored by this port yet.

## Tests and source terms

```sh
PYTHONPATH=src/training/decision2:/path/to/decision-index-kit \
  python3 -m unittest discover -s src/training/decision2/external_index021/tests -v
make check CHANGED_FILES="src/training/decision2/external_index021"
```

The kit's code is MIT licensed. The underlying suite is assembled from
sources with different terms, including noncommercial, gated, and
nonredistributable material. Keep reconstructed rows and result payloads
private; see the pinned kit's `docs/suite.md` for each source and license.
