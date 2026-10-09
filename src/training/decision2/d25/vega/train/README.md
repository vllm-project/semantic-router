# Vega trainer and node runners (`d25/vega/train/`)

Full-weight fine-tuning of Qwen3.8-27B with a 255-way code readout (the recipe of
perplexity-ai/pplx-decider-v1.1-27b, Apache-2.0), FSDP2 over the 8 GPUs of one node, plus an
unattended per-node runner that works through a queue of arm specs.

| File | What |
|---|---|
| `model.py` | `DecisionReadout` (Qwen3.5 text backbone + readout from `lm_head` code rows), packed varlen forward (FLA gated delta + causal conv, flash-attn varlen; causal or noncausal full attention), checkpoint reading, "code-readout v1" export |
| `data.py` | row store over JSONL(.gz) shards, `decision_format.render()` encoding with per-epoch option shuffling, token-balanced packing plan |
| `train.py` | the trainer (FSDP2, AdamW, warmup + cosine, DCP checkpoints, auto-resume, exports, dev eval) |
| `launch.py` | single-node process launcher (forwards SIGTERM, waits for the checkpoint) |
| `runner.py` | per-node queue runner (wait for data, prepare, train, pause at exports, parity, evals, results) |
| `export.py` | offline DCP -> export conversion; export reload parity (trainer code, Perplexity-style reference, ws-measure engine) |
| `merge_lora.py` | Decision 2.0 Vega LoRA merged into Qwen3.8-27B in FP32 (warm start) |
| `soup.py` | FP32 weighted averages of exports (same export format) |
| `probe_env.py`, `bench_comm.py`, `smoke_data.py` | environment probe, RCCL/GEMM benchmark, smoke rows |
| `../k8s/train_job.py` | code staging, runner Jobs, one-off arm/tool Jobs |

Node facts (hostnames, ssh targets, base-model paths) live in a private mapping file
(`~/.config/d25-vega/nodes.json`, never committed); the generator reads it.

## Appending an arm to a node queue

Every runner node has `/data/d25/vega/queue/<NN>/{pending,running,done,failed,state}`. Write a spec
JSON into `pending/`; the runner takes the lexicographically first file after its current arm. Name
files `<order>-<arm>.json` (e.g. `30-w2-t2-nc-lr1.json`).

```json
{
  "name": "w2-t2-nc-lr1",
  "train": ["/data/d25/shared/data/v1/M2T-v5"],
  "attention_mode": "noncausal_full_attention",
  "lr": 1e-6,
  "export_fractions": [0.3333, 0.6667, 1.0],
  "evals": {"intermediate": ["proxy"], "final": ["proxy", "full"]}
}
```

Every other key falls back to the runner defaults (= Perplexity's run): `init` `"base"` (the node's
verified Qwen3.8-27B @ 1d4bf0f2) or a checkpoint dir (`init_kind` `warm`/`export`), `readout_lr`
(= lr), `warmup_ratio` 0.15, `schedule` cosine, `min_lr_ratio` 0.1, `epochs` 1, `seed` 20260920,
`rows_per_update` 256, `max_length` 8192, `teacher`/`teacher_weight` (mix `meta.teachers.<name>`
into the target), `brier_weight` 0, `dev` (default `<train dir>/dev.jsonl.gz`), `wait_for` (extra
paths), `wait_timeout_h` 12, `prepare` (`[{"run": "<bash>", "creates": "<path>"}]`, skipped when
`creates` exists), `save_every` 1000, `keep_dcp` 2, `token_budget` 49152, `trainer_args` (extra
trainer flags), `max_attempts` 3, `parity` true. A train path is ready when it is a file or a
directory with a `VERIFIED` (or `READY`) file; the first ready candidate is pinned for the arm.

Tool items run shell steps on the node's GPUs without training:

```json
{"name": "t-parity-w1-t2-nc", "kind": "tool",
 "steps": [{"run": "python -m d25.vega.train.export parity --ckpt /data/d25/vega/ckpt/w1-t2-nc/step-001712 --dev /data/d25/shared/data/v1/M2T-v5/dev.jsonl.gz --engine", "timeout_h": 1}]}
```

Steps run with cwd = the node's current code tag and PYTHONPATH = FLA overlay + tag; env has
`D25_BASE_DIR`, `D25_CODE_DIR`, `HF_HOME`, `HF_TOKEN`.

## What a runner does per arm

1. Waits for data (poll every 2 min up to `wait_timeout_h`), pins the code tag (trainer code stays
   fixed for the arm), runs `prepare` steps.
2. Trains on 8 GPUs. The trainer auto-resumes from the latest complete DCP checkpoint
   (`/data/d25/vega/ckpt/<arm>/dcp/step-NNNNNN`, every `save_every` updates, SIGTERM, and at every
   pause). At each intermediate export it saves a DCP checkpoint and exits 10; the runner evaluates
   and relaunches it.
3. Exports go to `/data/d25/vega/ckpt/<arm>/step-NNNNNN/` (bf16 `Qwen3_5Model` incl. the frozen
   vision tower, FP32 `readout.safetensors`, `decision_config.json` with codes/token ids/attention
   mode/provenance, tokenizer files, `parity_rows.jsonl` = in-memory model probabilities on 32 dev
   rows).
4. The first export of each arm gets a parity check (`step-NNNNNN.parity.json`). Every export gets
   `python -m d25.vega.eval.ckpt_eval --ckpt <export> --out /data/d25/vega/results/<arm>/step-NNNNNN/
   --gpus 8 --what proxy|full`; results are copied to `/data/d25/vega/results/index/<arm>-step-NNNNNN.json`,
   events appended to `/data/d25/vega/results/events.jsonl`, both mirrored best-effort to the HF
   dataset `vllm-sr/d25-vega-results` (`<NN>/`). A failed eval leaves
   `state/<arm>/eval-step-NNNNNN-<what>.failed`; delete it to retry at the runner's next eval pass.
5. Fails an arm on NaN/inf (trainer exit 4), if the mean train loss of updates 191-200 is not below
   that of updates 6-15 (exit 3), or after `max_attempts` other trainer failures; moves the spec to
   `done/` or `failed/` (+ `.reason.txt`).

Runner restarts are safe at any time: the pod restarts in place (`restartPolicy: OnFailure`,
`backoffLimit: 6`) and resumes the spec in `running/`. Between phases the runner re-execs itself
when `src/current` carries a new `runner.py` that imports cleanly.

## Launching runners and staging code (from the worktree, `src/training/decision2`)

```bash
python3 d25/vega/k8s/train_job.py stage --node 06 --full --set-current   # whole d25 package -> /data/d25/vega/src/f-<sha12>, current -> it
python3 d25/vega/k8s/train_job.py runner --node 06 > runner-06.yaml       # Job d25-vega-runner-06 (8 GPUs)
kubectl --context vllm-sr apply -f runner-06.yaml                         # after a free-GPU check + ledger line
```

One-off Jobs (same pod layout): `train_job.py arm ...` (a single training run without the queue)
and `train_job.py tool --name <what> --node <NN> --tag <tag> --gpus N -- <command>`.

## Trainer CLI (direct use)

```bash
python -m d25.vega.train.launch --nproc 8 --log-dir <out>/logs d25.vega.train.train \
  --run <arm> --train <mixture dir|file> --dev <dev.jsonl.gz> --output /data/d25/vega/ckpt/<arm> \
  --init <base|merged|export dir> --attention-mode noncausal_full_attention \
  --token-budget 49152 --reshard-after-forward off --export-fractions 0.3333,0.6667,1.0
```

Key flags: `--lr/--readout-lr/--weight-decay/--warmup-ratio/--schedule/--min-lr-ratio/--epochs/--seed`,
`--rows-per-update`, `--max-length` (longer rows are dropped and counted, never truncated),
`--teacher/--teacher-weight`, `--brier-weight`, `--shuffle-options`, `--ac full|none|every:N|first:N`,
`--reduce-dtype fp32|bf16`, `--save-every/--keep-dcp/--dcp-threads`, `--export-steps/--export-fractions`,
`--pause-after-export`, `--dev-every/--dev-rows`, `--max-steps`, `--profile-steps`.
Environment (set by the Job generator): FLA overlay first on PYTHONPATH,
`NCCL_MIN_NCHANNELS=64` (RCCL otherwise picks 4 channels on these VMs: 13 GB/s all-gather instead of
~95 GB/s), `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, `TRITON_CACHE_DIR` +
`TRITON_CACHE_AUTOTUNING=1`.

## Other tools

```bash
# Decision 2.0 Vega warm start (FP32 merge, bf16 output, ~1 min on a GPU when the adapter is cached)
python -m d25.vega.train.merge_lora --base $D25_BASE_DIR --out /data/d25/shared/models/decision-2.0-vega-27b-merged-bf16 --device cuda
# FP32 soup of exports (weights normalised to 1; members must share codes/attention mode)
python -m d25.vega.train.soup --ckpt A --ckpt B --weights 0.5,0.5 --out /data/d25/vega/ckpt/soups/<name>
# DCP checkpoint -> export, and export parity
python -m d25.vega.train.export dcp --run-dir /data/d25/vega/ckpt/<arm> --step <N>
python -m d25.vega.train.export parity --ckpt <export> --dev <dev.jsonl.gz> --engine
```
