# Tune routing from application examples

Use this reference after discovering the running schema and selecting the
deployment through the [operations skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md). Start with an application
objective: better answers, faster responses, lower cost, safer handling, or more
stable agent runs. Record the acceptable tradeoffs before changing a threshold.

## Map the complete path

Inspect every entrypoint, including its named Recipe and model assignments.
Write down what each part contributes:

| Part | Design question |
| --- | --- |
| Signals | What evidence distinguishes requests that need different handling? |
| Projections | Which combinations, scores, or partitions summarize that evidence? |
| Decisions | Which distinct actions does the application need? |
| Algorithms | How does each action select or coordinate eligible models? |
| Plugins | What retrieval, tool, cache, or privacy behavior belongs to that action? |
| Models | Can the assigned endpoints execute the actual request and output budget? |

Prefer a few decisions with short names. Keep `simple`, `medium`, and `reasoning`
distinct because their service objectives differ; do not create one decision
per keyword, language, or classifier label. A fallback should also have an
explicit quality and availability policy.

## Select evidence for a purpose

Use heuristics for protocol facts and explicit constraints: active tool calls,
required tool choice, structured output, or a declared privacy requirement.
Use Embedding for semantic intent, Complexity for effort, Domain with FactCheck
for consequential questions, and Feedback with Reask for recovery. Topic alone
does not establish difficulty; FactCheck predicts a need for checking and does
not verify a statement.

Combine independent clues in a projection when no single clue is sufficient.
Review negative conditions and `on_unknown` together: an inference failure must
not become an ordinary false value under `NOT`. Include failing-model tests.
Do not assume an early keyword condition avoids learned inference. Used signal
families can execute concurrently before decision composition; measure the
whole request and the actual forwards.

Build semantic examples around operations, not application topics: reconciling
conflicting evidence, coordinating independent workers, or revising an incorrect
result. Include ordinary explanations and quoted instructions as counterexamples,
and cover distinct operations across the supported languages. An explicit request
for independent review should not depend on matching one particular example.

Separate authorization from execution details. A request to delegate work must
authorize multiple workers; mentioning agents or asking for an explanation does
not. Once authorized, the workflow owns planning and integration. Requiring the
user to name each internal stage makes otherwise valid requests brittle. Test
single-worker, negated, quoted, and informational controls alongside delegation.
Check that worker counts survive translation and inflection. In languages without
grammatical noun plurality, an independent worker does not by itself establish
multiple executors.

Feedback routing requires a prior assistant reply. Include that history in both
Preview and routed tests; a `user_feedback` condition cannot corroborate a
history-free request. Distinguish correction of an answer from ordinary editing
or a request for a different format. A direct instruction to correct a previous
result and a self-contained error report with an operative repair instruction
are useful separate cases; neither requires a Feedback match to prove the
explicit instruction. Preserve missing-history and negated-repair controls.

For comparisons between embedding intents, `value_source: raw` preserves scores
below the match threshold. A signed difference can express which intent has more
evidence; it still needs a mapping and evaluation on both positive and negative
cases. Cosine similarity is not a calibrated probability. Use per-rule matches and
values, not a family's aggregate confidence, to explain an individual predicate.

Check the effective matching settings before tuning candidates. Embedding soft
matching is opt-in: enabling it permits matches below individual rule thresholds
when no strong match exists. `top_k` can suppress otherwise valid matches; use
`top_k: 0` when decisions need several independent embedding signals. Compare
the published scores with emitted matches in Preview. Check each inference
deployment's token budget separately from the selected backend's context window.

Also inspect prototype compression. A multilingual candidate list does not
guarantee every example survives clustering and the prototype limit. Rule-local
`prototype_scoring` travels with the Recipe; without a rule override, the global
configuration applies. To retain all deduplicated candidates, use an override
with `enabled: false`, and set `best_weight` and `top_m` for the intended scoring.
Disabling compression does not turn the resulting score into a raw maximum.

Local Complexity uses the hard-prototype score minus the easy-prototype score.
Use `hard_above` and `easy_below` to calibrate its two sides independently;
remove the symmetric `threshold` when supplying that pair. A score equal to
either boundary remains medium. Moving the hard boundary closer to zero can
recover demanding tasks while increasing unnecessary escalation. Keeping the
easy boundary farther from zero preserves stronger evidence before downgrading
an ordinary request. Check all three classes and their false positives, rather
than treating more hard matches as improved quality. Candidate banks and scoring
settings can move both distributions even when one boundary stays unchanged.

Guard detects prompt attacks. Safety and Hazard identify content risk and
categories; they do not establish malicious intent. Pair harmful requests with
help-seeking, quotation, and analysis controls before choosing refusal behavior.
PII can select a restricted processing pool, but it does not redact content or
establish where a provider stores it.

## Add retrieval where evidence is available

Use Rerank after retrieval from a real knowledge base. Start with a single-model
answer path, retrieve a wider candidate set, and retain a smaller ranked set
through `rag.rerank`. Verify its recipe-scoped `rag.reranker` binding. Preserve
document identities and assess retrieved coverage, ranking, grounded answer
quality, and added latency separately.

Preview verifies selection of the RAG plugin; a routed request verifies retrieval
and reranking. A relevance score is not a truth probability. Avoid attaching RAG
to every multi-model stage without checking repeated retrieval and context growth.
Keep the ordinary no-KB path usable without provisioning a knowledge base.

## Protect agent continuity

Inspect `global.router.learning.protection` and decision-local `adaptations`;
the legacy `algorithm.type: session_aware` is not the current contract. Use stable
session and conversation identities in actual routed requests. Test a complete
tool cycle, continuation, explicit correction, failed model, decision change,
and a new conversation. Repeat the same session ID across different Recipes to
verify isolation.

Built-in recipe initialization enables conversation protection and leaves online
adaptation off. It fills missing settings while preserving explicit values in
the base configuration, including disabled protection. Inspect the generated
configuration before activation; these defaults apply across its recipes, with
decision-level bypasses preserved. Clients without stable identities cannot
obtain a hold. Evaluate application outcomes before enabling online adaptation.

Preview can return `execution_required` for protected selection. It does not
advance conversation state. Verify continuity with actual routed requests and
Replay, rather than treating Preview's proposed model as an applied hold.

Compare `apply`, `observe`, and `bypass`. An observed recommendation to stay is
not an applied hold. Verify the actual selected model, backend response, route
headers, Replay API, and Dashboard agree. Continuity cannot retain a model that
is outside the current eligible pool. Keep hard privacy boundaries and bounded
multi-model workflows explicit rather than enabling a global hold indiscriminately.

## Run a comparable experiment

Pair requests that share the same background but require different work. Vary
answer style separately: asking for a brief answer does not make a task easier.
Compare easy, medium, and hard score distributions before adjusting boundaries. When
they overlap heavily, improve the task examples or model instead of treating
more reasoning calls as evidence of better discrimination. Count related
paraphrases and tool variants as views of the same case family.

To compare model selection, assign at least two eligible, reachable models. A
single candidate can verify delivery but cannot demonstrate a tradeoff between
models. Keep route correctness, answer quality, and measured resource use separate.

For a workflow, tune planning, workers, and synthesis separately. Use
`planner.model` and `final.model` when different models suit those roles. Verify
the model's reasoning family and the actual backend request before assuming a
decision disables thinking. Test valid planner JSON, distinct completed workers,
and a complete final answer within the gateway deadline. A fallback plan,
reasoning-only response, or `finish_reason: length` does not establish successful
dynamic planning and delivery. Increasing a stage's token budget also consumes
more of the total latency budget.

1. Freeze the baseline config, assignments, runtime and model revisions. Save
   original probe identities. Add new counterexamples in a separately identified
   packet before seeing candidate outputs; keep routing controls out of training.
   If those outputs influence a later boundary choice, report them as calibration
   data and use another frozen set for a subsequent independent check.
2. Cover every decision and fallback with ordinary, boundary, negative, multilingual,
   long-input and multi-turn cases. Include instructions quoted as data, absent
   history, required tools, unknown signals, and insufficient backend capacity.
3. Validate and activate the candidate through the
   [configuration workflow](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md). Verify its active revision.
4. Recombining saved signal values can isolate a policy change, provided the
   unchanged baseline reproduces its original heuristics and decisions exactly.
   Label this as offline recomposition, then run real Preview requests with full
   messages, tools, padding and images. In the
   contributor conformance harness, choose `deployment` or `policy` scope explicitly.
   Keep strict deployment checks as the default. Report selected, immediate-response,
   execution-required and unavailable outcomes separately. Policy coverage does not
   prove backend capacity; signal errors and failed HTTP calls remain failures.
5. Send real routed requests for each execution path. Verify completed output,
   distinct workers for multi-model algorithms, plugin effects and session behavior.
6. Compare paired route outcomes and answer quality alongside cold/warm latency,
   median/p95, classifier work, backend calls, tokens and measured cost. Report
   unavailable measurements explicitly. Do not sum concurrent signal times as
   though all forwards were sequential.
7. Keep the candidate only when the stated objective and hard constraints pass.
   Otherwise restore the baseline and preserve the failed evidence. Keep raw test
   outputs outside Git; publish intentional recipes, documentation and code fixes.

Use the [evaluation loop](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md) for exact Preview, Probe, delivery
and benchmark contracts. A partial benchmark can guide tuning, but it does not
establish a full leaderboard score or performance on untested hardware.
