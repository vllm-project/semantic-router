# Model runtime

- `vllm_srun/api/openapi.yaml` is the contract. Change it first, then the
  server, the contract tests and the generated Go client
  (`make model-runtime-client-generate`) in the same change.
- Plugin layers stay separate: families own package formats, rendering,
  readout and answers; engines own backbone execution; accelerators own
  devices and kernels. The scheduler's batcher owns batch formation, from the
  engine arguments. A family never imports an engine and an engine never
  parses a package.
- The golden check's reference forwards (each request alone, in its released
  split) must stay byte-identical to the released packages' runtime. Batching
  is on by default; any other default that can change answers (shared
  context, approximate kernels, reduced copies) needs an accuracy record under
  `docs/records/` that shows no accuracy loss, and is otherwise an engine
  argument that stays off by default. `--dtype auto` keeps each model's
  published precision.
- Never import or execute code shipped inside a model package, and never use
  `trust_remote_code`.
- Unit tests use the tiny fixtures in `vllm_srun/testing/fixtures.py`.
  GPU tests carry the `gpu` marker and skip on CPU hosts.
