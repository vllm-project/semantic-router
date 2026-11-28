# recipe-lab

Closed-loop experiments for semantic-router routing recipes (RSI workbench).

This is the first slice of the [recipe-lab umbrella](https://github.com/vllm-project/semantic-router/issues/5404).
It turns "routing accuracy is X%" into a prioritized, executable fix queue:
every wrong outcome on a labeled dataset is classified into an actionable
error class ([#5489](https://github.com/vllm-project/semantic-router/issues/5489)),
and the class distribution decides where the later recipe-lab phases spend
effort. On top of the attribution report this slice closes the smallest
replay-tier loop ([#5490](https://github.com/vllm-project/semantic-router/issues/5490)):
propose candidate configs deterministically, evaluate them through the real
router against a frozen answer bank, and rank the results into an append-only
ledger - no LLM generation anywhere in the replay tier.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

## Commands

| command | purpose |
|---|---|
| `plan` | load and validate an experiment spec, print it normalized |
| `hash` | frozen dataset identity: sha256 over the raw bytes (A4) |
| `split` | deterministic group-wise calib / dev / frozen_test split (B) |
| `collect-replays` | pull `router_replay` records from the router management API and join them against the labeled case list (`--unlabeled` emits raw diagnostic rows) |
| `attribute` | run the E1 / decision_mismatch / E2 / E3 misroute attribution over labeled records |
| `bank` | build / apply the per-case, per-model answer bank |
| `propose-thresholds` | bounded single-signal threshold perturbations over a base config |
| `propose-rule-thresholds` | move every predicate site of one signal inside a decision's rule at once, with site provenance (C) |
| `run` | replay-tier evaluation of candidate configs -> RANKED ledger records (A1/A5) |
| `ledger` | inspect or append to the evidence ledger (`show` / `append` / `top`) |
| `cases-to-srbench` | render the labeled case list as an sr-bench dataset; the printed `srbench_sha256` freezes into the contract's split declaration (X3) |
| `verify` | drive sr-bench end to end for one candidate: Preview routing, `--repetitions` AB/BA pairs, per-pair compare and the shared three-state verdict (I) |
| `live-verdict` | compare two already-recorded sr-bench runs -> PASS / FAIL / INCONCLUSIVE with guard floors from `spec.gates` (A3) |

Worked examples with a runnable spec and records live in
[`examples/`](examples/):

```bash
# validate and preview the experiment contract (runs nothing)
recipe-lab plan --spec examples/spec.example.yaml

# pull replay records from a running router and join them to labels
recipe-lab collect-replays --url http://127.0.0.1:8080 \
  --cases examples/cases.example.jsonl --output attribution-input.jsonl

# classify every wrong outcome and write the report
recipe-lab attribute --spec examples/spec.example.yaml \
  --records examples/records.example.jsonl --output-dir out/
```

`attribute` writes `attribution.jsonl` (one row per case; the file path plus
its content hash is the downstream reference - later recipe-lab phases cite
it as `attribution_ref`) and `report.md` (class distribution, E1 breakdowns
per signal family and per failing signal, decision confusion pairs, policy
deviation and per-domain E3 candidates).

## Replay-tier loop

The replay tier re-plays a frozen dataset through the real router for each
candidate config and grades answers by looking them up in the answer bank -
routing decisions always come from the real router, never from a Python
reimplementation, and no generation is invoked. The workflow:

1. **Frozen identity (A4).** The spec's workload pins every split's
   `content_hash`; `attribute`, `run` and the proposers recompute it before
   touching the data and refuse any drift (exit 2).
2. **Splits (B).** `split` cuts a labeled dataset group-wise into
   calib / dev / frozen_test; `frozen_test` is quarantined (replay / propose
   / attribute refuse to touch it) and reserved for the live verdict.
3. **Answer bank.** `bank build` asks each candidate model directly for every
   case once, offline; `bank apply` joins correctness onto replay records by
   (case, routed model). Cases routed to a model without a saved answer are
   no-evidence, never wrong; a candidate above the coverage threshold is
   flagged with `coverage_warning`.
4. **Proposals.** `propose-thresholds` perturbs a single decision-signal
   threshold (bounded, clamped, baseline dropped); `propose-rule-thresholds`
   moves all predicate sites of one signal within a decision rule in one
   calibrated step. Both read the canonical v0.5 layout (signals and decisions
   live in `recipes[]`); a document with several recipes takes `--recipe
   <name>` or `--public-model <name>` to name the one to edit. Every
   candidate YAML is accompanied by a
   `<stem>.candidate.json` sidecar (kind / ref / change_summary /
   provenance / sites).
5. **Replay evaluation (`run`).** Each candidate boots the real router with
   the candidate config, the frozen cases are routed, grades come from the
   bank, and one RANKED record per candidate is appended to the ledger with
   the dataset hash and proposer provenance. `--reload hot` keeps one router
   alive and swaps candidates through the management API, falling back to a
   per-candidate restart when the API rejects the change. `--validate`
   pre-flights each candidate with the validator binary; rejected candidates
   never reach the ledger (exit 2 when all are rejected).
6. **Ledger.** `ledger show` filters by round / verdict / proposer, and
   `ledger top --spec` prints the best replay candidate by the spec's
   objective metric. Records are append-only, fingerprinted with stable
   content-derived IDs, and publication is atomic (a killed writer leaves
   the previous ledger intact). Appending the same content again stores
   nothing new, and writers are serialized by a `<ledger>.lock` file next to
   the ledger.

```bash
# cut the frozen dataset into splits (same dataset + args => byte-identical output)
recipe-lab split --cases labeled.jsonl --output-dir splits/

# calibrate a rule-level gate in one step (all difficulty sites 2.5 -> 1.75)
recipe-lab propose-rule-thresholds --base-config config.yaml \
  --signal difficulty --from 2.5 --values 1.75 --output-dir cands/

# a document with several recipes: edit the one a public model serves
recipe-lab propose-thresholds --base-config config.yaml \
  --public-model vllm-sr/mom-v1-blend --signal verification --deltas 0.1 \
  --output-dir cands/

# replay evaluation over the dev split with pre-flight validation
recipe-lab run --spec spec.yaml --split dev --cases splits/dev.jsonl \
  --bank bank.jsonl --candidates cands/ --ledger ledger.jsonl \
  --router-bin build/bin/semantic-router \
  --mock-backend scripts/mock_backend.py \
  --output-dir out/ --validate --reload hot

# ledger
recipe-lab ledger show --file ledger.jsonl --tier replay --json
recipe-lab ledger top --file ledger.jsonl --spec spec.yaml
```

A ledger record example lives at
[`examples/ledger-record.example.json`](examples/ledger-record.example.json).

## Insight references

`run` and `ledger append` accept comma-separated provenance references which
are merged into the record's `candidate.provenance.insight_refs` (dedup);
unverifiable references are rejected (exit 2):

- `rec-<id>`: an existing ledger record ID
- `bench:<run_id>`: a benchmark run ID (format-checked only)
- `attr:<path>@sha256:<hex>`: an attribution artifact; the hash must match
  the file bytes
- `diag:<path>@sha256:<hex>#<anchor>`: a diagnosis artifact anchored at a
  location inside the file

## Record format (attribution input)

One JSON object per line (the `router_replay` adapter emits this shape):

```json
{"case_id": "c1", "truth": "math", "matched_decision": "code", "routed": "local/model-b",
 "family": "d3", "answer_ok": false}
```

- `answer_ok`: `true` / `false` / `null` (`null` = no evidence; by the
  evidence-first rule unknown correctness is never counted as an error).
- `matched_decision` is the decision the router actually matched (renamed
  from the earlier `pred`; reading old records with a top-level `pred` field
  still works).
- `truth` is either the hand-labeled expected decision for that prompt (a
  string), or an object `{"signals": {...}}` carrying per-signal
  ground-truth facts: each fact is a label string (`"domain": "code"`), a
  number (`"difficulty": 3`), or a numeric predicate (`"difficulty": {gte: 3}`).
  Facts drive both the oracle policy's rule evaluation and the E1 signal
  judgment.
- Signal evidence (`signal_values` / `signal_confidences` /
  `decision_tier`) is carried verbatim from the replay record. Keys are
  `<type>:<name>` (e.g. `decision:difficulty`); choice-type decision signals
  store one probability per choice under `decision:<name>:<choice>`, while
  score-type signals keep the numeric score under `decision:<name>`. When
  reading a signal, values fall back per key from `signal_values` to
  `signal_confidences` (matching the router's own lookup).

## Error classes

| class | meaning |
|---|---|
| `correct` | routed as intended and answered correctly |
| `E1_signal` | a truth signal contradicts the record's observed signal evidence (needs signal-level truth: truth = `{"signals": ...}`) |
| `decision_mismatch` | matched decision != expected decision, judged without usable signal-level truth (fallback; with fully verified signal truth a decision miss is `E2_policy`) |
| `E2_policy` | decision or routed model deviates from the oracle-policy expectation, answer wrong |
| `E3_capability` | routing exactly as intended, answer still wrong |
| `suboptimal_routing` | policy deviation but still a correct answer (counted, never an error) |
| `no_evidence` | answer correctness unknown; never an error |
| `policy_gap` | truth (label or signal facts) not covered by the declared oracle policy |

## Oracle policy

The oracle policy is declared in the experiment spec under
`attribution.oracle_policy`; attribution never recomputes it from router
logic. Two forms are supported:

```yaml
attribution:
  oracle_policy:
    # form 1 (legacy): decision label -> intended model
    math: local/model-math
    code: local/model-code

    # form 2: signal-fact rules, evaluated top-down; first match wins
    rules:
      - when: {domain: code, difficulty: {gte: 3}}
        expect: {decision: domain_code_complex, models: [m/pro, m/max]}
      - when: {domain: code}
        expect: {decision: domain_code}
    default: {decision: simple_general}
```

String truths resolve through the label mapping; `{"signals": ...}` truths
evaluate the rules against the truth facts (unmatched and default-less cases
become `policy_gap`). A truth whose signals contradict the record's observed
signal evidence is `E1_signal` (the failing signal and its observed value are
recorded on the row).

## Signal family

`family` on attribution rows is derived from the case's own signal evidence
key type: `classifier:` -> `vela_classify`, `decision:` -> `vela_decisions`,
`projection:` -> `d3`; other key type segments surface verbatim. Records
without signal evidence keep whatever `family` tag the input record carries
(e.g. from `collect-replays --family`).

## Live tier

The replay tier's winner is cheap but not proof: it assumes answers are
routing-independent and knows nothing about real service behavior. The live
tier attaches the loop to sr-bench and adds evidence gates strong enough to
support a keep-or-rollback decision:

1. **sr-bench integration, not a parallel engine.** recipe-lab drives sr-bench
   through a thin `SrbenchCli` wrapper (`experiment` / `preview` /
   `candidate-plan` / `compare` / `report`); the ledger records sr-bench run
   IDs (`bench_refs`) and never copies evaluation results - sr-bench stays the
   single source of truth.
2. **`verify`** drives one candidate end to end: Preview confirms the routing
   distribution first, then `--repetitions` AB/BA pairs (even pairs lead with
   the baseline, odd pairs with the candidate), each pair re-running the
   baseline's frozen manifest and submitting a fresh candidate-plan, with
   per-pair compares and the shared three-state verdict. Reference-arm run
   IDs are cached so repeated invocations do not re-run expensive arms (M4).
3. **`live-verdict`** produces the three-state verdict for two recorded runs:
   INCONCLUSIVE > FAIL > PASS. Timeout or interrupted trials are INCONCLUSIVE,
   never FAIL; a declared guard floor (privacy / jailbreak safety, quality)
   that is breached fails the verdict even when the objective metric wins;
   missing guard evidence is also INCONCLUSIVE. Guard measurements are passed
   as rows (`--guard-measurements`) and never override the spec's floors.
4. **Identity discipline.** Frozen dataset and metric identity is enforced
   (A4 re-hash before touching data; sr-bench runs must bind to the split's
   dataset identity, X3); `frozen_test` is quarantined behind `--final`;
   transfer workloads are final-verify only (M5). AB/BA arm order is paired
   and explicit; paired statistics across repetitions carry an honest
   "statistical significance is not established" note. Cost gates read from
   the contract's objective constraints (E) with explicit cost assumptions.
5. **Promotion evidence replay.** `verify_promotion_evidence` re-checks a
   recorded PASS verdict from its ledger record alone, so a promotion claim
   can be audited without re-running anything.
6. **`score-candidate`** is the agent's own channel: it scores one deployed
   candidate against a saved sr-bench baseline run through the installed
   `vllm-sr benchmark` worker. The candidate plan must reproduce the frozen
   baseline (same cases, profile, seed and subject protocol) and match the
   expected active runtime hash before anything is dispatched. The receipt
   (`recipe-lab-sr-bench-run-v1`) is printed as one JSON line and saved
   under `<workspace>/sr-bench`. Exit codes follow the verdict: 0 PASS,
   1 FAIL, 2 INCONCLUSIVE.

## Scope of this slice

This slice owns the replay tier and the live tier: contracts
(spec/candidate/gates/budget/constraints), splits, the answer bank, the
deterministic proposers, replay evaluation through the real router, the
ledger, sr-bench integration and the three-state live verdict. Reports,
context export and provenance manifests arrive with
[sub-issue 4](https://github.com/vllm-project/semantic-router/issues/5492).

## Tests

```bash
pytest
```

`tests/native_srbench_run.py` is opt-in: it drives `score-candidate` through
the installed `recipe-lab` executable against the real sr-bench worker with
local HTTP fixtures, so it needs the Recipe Lab wheel on `PATH`:

```bash
pytest tests/native_srbench_run.py
```
