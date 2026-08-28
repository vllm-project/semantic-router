# MoM Evaluation Maintainer Skill

Use this skill when publishing, regressing, or extending MoM end-to-end
evaluation for issue #3038.

## Read first

1. [bench/mom_eval/README.md](../../../../bench/mom_eval/README.md)
2. [config/evaluation/mom-core-suite/v1/manifest.yaml](../../../../config/evaluation/mom-core-suite/v1/manifest.yaml)
3. [PL-0038 execution plan](../../../docs/plans/pl-0038-mom-evaluation-epic.md)

## Workflow

1. Confirm decision-level probes pass:

```bash
make recipe-conformance-static
```

2. Validate MoM evaluation contracts:

```bash
make mom-eval-validate
```

3. Run smoke locally:

```bash
make mom-eval-smoke MOM_EVAL_ENTRYPOINT=vllm-sr/mom-v1-blend
```

4. With live backends, run release-candidate:

```bash
make mom-eval-rc MOM_EVAL_ENTRYPOINT=vllm-sr/mom-v1-ultra
```

5. Publish scorecards when regression passes:

```bash
make mom-eval-publish MOM_EVAL_ENTRYPOINT=vllm-sr/mom-v1-ultra
```

6. Update `evaluation-scorecard.md` in the recipe directory if aggregate
   summaries change.

## Rules

- Core suite failures block publication regardless of extension pack scores.
- Smoke runs are diagnostic only.
- Every publishable score must ship with recipe snapshot, command, raw outputs,
  and baseline arms in the result bundle.
- Do not replace or weaken decision-level probe requirements.

## Adding an extension pack

1. Add manifest under `config/evaluation/packs/<name>/v1/manifest.yaml`
2. Register in `config/evaluation/packs/registry.yaml`
3. Implement `bench/mom_eval/packs/<name>.py` with `create_pack()`
4. Declare the pack in the recipe `mom-evaluation.yaml` entrypoint section
