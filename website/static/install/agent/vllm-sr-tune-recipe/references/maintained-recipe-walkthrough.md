# Balance recipe improvement walkthrough

Use the maintained [Balance recipe card](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/README.md),
its [canonical config](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/config.yaml)
and [probe manifest](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/probes.yaml).
The example is a workflow, not a claim that the current recipe misroutes or
that a proposed threshold improves it.

1. **Reproduce a reported wrong lane.** Identify one development probe ID
   and the expected lane from `probes.yaml`. Run the active stack's traced
   preview and bounded probe through its public entrypoint. Record the active
   config hash, signal answers/unknowns, selected decision/model and delivery
   receipt. If the answer-model backend failed after correct selection, stop
   recipe tuning and diagnose delivery.
2. **Choose the responsible field.** Read the Balance `recipe.dsl` or
   canonical config for that signal, predicate, decision priority and
   algorithm. Compare an intended match with a counterexample. If the signal
   is unknown, check its model deployment and timeout first. If the signal
   answered but the intended decision did not match, propose one predicate or
   condition change. Record the original field and why the trace supports the
   candidate.
3. **Protect the evaluation.** Freeze the development probe IDs and guard
   floors before editing. Keep a separate holdout uninspected. Validate the
   candidate config and inspect `config plan`; do not replace a running stack
   without an authorized activation and rollback point. Use
   `make recipe-conformance-static` for authored assets and
   `make verify DOMAIN=maintained-recipes` when the live maintained-recipe
   environment is available.
4. **Measure and decide.** Capture before/after route and delivery evidence,
   then paired sr-bench runs on the same development cases and request
   profile. Report planned/paired/failed counts, quality interval, route
   distribution, latency, complete cost basis and guard outcomes. Keep only
   if the prespecified objective and every guard floor pass; revert on a
   breach; otherwise state inconclusive and the missing evidence. A one-probe
   correction is not a broad quality win.
5. **Hand off.** If #5497's `recipe-lab-loop` skill is available, submit the
   candidate through its own CLI and link its ledgered verdict. Otherwise
   retain the manual candidate diff, calibration report and paired run IDs.
   A human reviews promotion and the independent holdout; the development
   loop does not consume it.
