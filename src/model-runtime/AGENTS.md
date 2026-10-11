# Model runtime

- `vllm_srun/api/openapi.yaml` is the contract. Change it first, then the
  server, the contract tests and the generated Go client
  (`make model-runtime-client-generate`) in the same change.
- Plugin layers stay separate: families own package formats, rendering,
  readout and answers; engines own backbone execution; accelerators own
  devices and kernels; profiles own numerics and batch formation. A family
  never imports an engine and an engine never parses a package.
- The `exact` profile must stay byte-identical to the released packages'
  runtime. Anything that can change answers belongs to an opt-in profile with
  an accuracy record under `docs/records/`.
- Never import or execute code shipped inside a model package, and never use
  `trust_remote_code`.
- Unit tests use the tiny fixtures in `vllm_srun/testing/fixtures.py`.
  GPU tests carry the `gpu` marker and skip on CPU hosts.
