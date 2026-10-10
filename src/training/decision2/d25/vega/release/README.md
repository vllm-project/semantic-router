# Decision 2.5 release engineering

From a code-readout v1 export (SPEC.md) to a verified PRIVATE Hugging Face package, its model card, the CUDA
readiness run on an RTX PRO 6000 and the Decision Index submission files. Nothing here makes a repository or
dataset public.

## Package layout

The repository root is the export itself, so every code-readout loader reads it unchanged:

```text
config.json                 Qwen3_5Model backbone config + auto_map {AutoModel} + custom_pipelines {decision}
model-*.safetensors, model.safetensors.index.json, readout.safetensors, decision_config.json, tokenizer files
decision25_format.py        d25/vega/common/decision_format.py, byte for byte (the training prompt contract)
decision25_runtime.py       the runtime: Decision25.from_pretrained(dir | repo).system_one(state=, questions=)
modeling_decision25.py      AutoModel.from_pretrained(repo, trust_remote_code=True) -> Decision25Model
pipeline_decision25.py      pipeline("decision", model=repo, trust_remote_code=True)
decision25_server.py        POST /v1/systemone (FastAPI), GET /health, GET /v1/models
decision25_engine.py        decision_index Engine: --engine decision25_engine:Decision25Engine --option model=<dir>
requirements.txt  LICENSE  README.md  assets/  MODEL_MANIFEST.json (SHA-256 + size of every file, identity)
```

Answers keep the Decision 2.0 shapes: choice `{type, choice, probabilities, confidence}`, noul `{type, noul}`,
score `{type, score, probabilities, confidence, legend}`, errors `{type, error, message}` with
`invalid_question`, `max_length_exceeded` (never truncated) or `invalid_model_output`. A request's questions run
in request order, 8 per forward pass (the ws-eval KitEngine / Perplexity batching), BF16 backbone, FP32 readout.

## Commands (from `src/training/decision2`)

| Step | Command |
| --- | --- |
| test checkpoint | `python -m d25.vega.release.checkpoints zero-shot --base <Qwen3.5 dir> --out <dir>` |
| build | `python -m d25.vega.release.build --export <export> --out <pkg> --model-name Decision-2.5-Vega-27B --repo-id vllm-sr/Decision-2.5-Vega-27B --card <card dir> [--teachers t.json]` |
| headroom / upload / download / readback | `python -m d25.vega.release.hub {headroom,ensure,upload,download,readback} ...` (pod or node; `hf` CLI) |
| smoke | `python -m d25.vega.release.smoke --model <dir or repo> [--card README.md] [--server] [--plain] --out s.json` |
| parity vs engine.py | `python -m d25.vega.release.parity --package <pkg> --rows parity-600.jsonl.gz --out p.json [--sequential]` |
| samples (latency-v1 style) | `python -m d25.vega.release.sample --kit <kit> --suite-dir <suite> --n 760 --warmup 10 --seed 20260926 --out latency-760.jsonl.gz` |
| latency / cross-device answers | `python -m d25.vega.release.compare {latency,answers} ...` (stdlib) |
| CUDA job (RTX PRO 6000) | `python -m d25.vega.release.hf_job --model <repo or standin> [--revision R] --run <name> --kit <kit checkout> [--reference <mi325x results in the work dataset>]` |
| card input / card | `python -m d25.vega.release.card_input ...` then `python -m d25.vega.release.card --input card-input.json --out <dir> --logo <png> --fonts <Inter TTFs>` |
| submission | `python -m d25.vega.release.submission {shard,merge,stage,upload,pr-text} ...`; GPU part: `submission_run.sh` |
| private push | `EXPORT=... REPO=... NAME=... CARD=... TEACHERS=... WORK=... [WAIT_HOURS=h] [CUDA_JOB=1 KIT=... RUN=...] bash private_push.sh` (CPU pod on the node with the export; dry run: `REPO=vllm-sr/d25-vega-staging`) |
| SDPA backends | `python -m d25.vega.release.sdpa_probe --out sdpa.json` (which attention kernels serve our shapes on a GPU) |

The push leaves the package's bytes checked twice more: the CUDA job downloads the revision on an RTX PRO 6000
(smoke incl. the card Quickstart, latency protocol, parity vs engine.py) and an MI325X run downloads it fresh
(re-hash, smoke, parity, the kit-760 reference that `compare.py answers` checks the CUDA answers against).

`build` refuses private paths, addresses, credentials and `node NN` names in any text file; site-specific
patterns come from the file named by `D25_LEAK_PATTERNS` (kept outside git). `hub upload` and `submission
upload` create repositories PRIVATE and refuse public ones. The CUDA job runs `cuda_job.sh` inside a Hugging Face
Job and uploads everything to the private dataset `vllm-sr/d25-vega-release-work` (`runs/<name>/`).

## Tests

```bash
cd src/training/decision2 && python -m unittest d25.vega.release.tests.test_release d25.vega.release.tests.test_server
```
