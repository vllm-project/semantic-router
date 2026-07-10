# AMD Dashboard And Playground Closure

## Goal

Ship a reliable AMD reference demo in which the Dashboard Playground uses the
router's automatic-model contract, all balance-profile aliases resolve to the
single local ROCm vLLM backend, and the authenticated console presents a
cohesive enterprise black, white, and neutral visual system.

## Scope

- fix the Playground request model and protect the automatic-model contract
- align Playground model presentation with the local aliases in `balance.yaml`
- refine the authenticated shell, auth transition, route loading, and
  Playground surfaces with a restrained AMD-aligned visual system
- apply the same visual and accessibility contract across dashboard surfaces,
  including drawer-based editing and responsive navigation
- make the Models manager safe and usable with hundreds of aliases through
  filtering, pagination, cross-page selection, and dependency-aware deletion
- validate frontend behavior locally and run the affected flow on a real AMD
  deployment built from this branch
- publish the completed change as one signed-off pull request

Non-goals:

- changing router-side alias semantics
- changing the physical AMD backend model

## Exit Criteria

- Playground no longer sends the unavailable `MoM` model
- automatic routing and every reference alias complete through local `vllm:8000`
- auth transition, shared shell, and Playground use the same neutral enterprise
  tokens, motion language, and accessible reduced-motion behavior
- dashboard unit, lint, type, build, harness, and real AMD browser/API checks pass
- the branch is signed off, pushed, and represented by a draft pull request

## Task List

- [x] `AMD-UI-01` Reproduce and fix the Playground automatic-model request
- [x] `AMD-UI-02` Add contract tests for automatic routing and local alias display
- [x] `AMD-UI-03` Consolidate neutral enterprise design tokens and shared shell
- [x] `AMD-UI-04` Align auth/loading motion and Playground presentation
- [ ] `AMD-UI-05` Complete local and AMD end-to-end regression
- [ ] `AMD-UI-06` Sign, push, and open the draft pull request

## Next Action

Build and deploy the validated branch on the AMD demo node, then run the live
alias, Playground, browser, and container-health regression before publishing.

## Progress

- Local frontend gates pass: lint, type-check, production build, 62 unit tests,
  and all 88 Chromium end-to-end tests.
- AMD recipe, dashboard handler, and CLI targeted tests pass.
- The remaining completion boundary is the real AMD deployment, live regression,
  signed commit, push, and pull request.

## Operating Rules

- keep `ChatComponent.tsx` as the transport orchestrator and move new pure
  request/model presentation logic to adjacent support modules
- use `deploy/recipes/balance.yaml` and its five aliases as the reference demo
  contract
- keep signal models on CPU during AMD validation so the vLLM backend owns GPU
  memory
- update this plan when a task or next action changes

## Related Docs

- `docs/agent/change-surfaces.md`
- `docs/agent/module-boundaries.md`
- `docs/agent/amd-local.md`
- `deploy/amd/README.md`
- `deploy/recipes/balance.yaml`
