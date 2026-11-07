# Kai run for pilot v0.1

This is the Kai arm of the shared pilot, run on the unchanged `inputs.jsonl` and
`question.json` from the research branch. Both files were checked against the
SHA-256 values in `protocol.json` before any request was sent, and they matched.

## Pins

| Item | Value |
| --- | --- |
| Model | `llm-semantic-router/Decision-1.0-Kai-0.6B` |
| Model revision | `9d6872cde6950c2c2b5786d182ec9a06ca1bdd66` |
| Runtime | vllm-project/semantic-router PR #4086, detached checkout of `58cd660b51ba19aff01ad6f89e66f2d07f13908b` |
| Verified artifact content ID | `fab978e9f5e2ff799e1c2352b48e9c68d332ddeb05cdc8a25a88b69e0c28cb72` |
| Device, dtype, threads | CPU, float32 backbone and heads, 8 threads |
| Question key | `intent` |

## How the model was served

- The runtime pins the revision itself. `prepare_artifact.py` calls the runtime's
  own resolver with the model ID and the revision above. It downloads each file
  from Hugging Face at that exact revision, verifies it against the model's
  artifact descriptor and writes a read-only, content-addressed tree.
- The server is the runtime's entrypoint, `python -m decision_runtime.entrypoint`
  with `--backend cpu`. At startup it resolved the same revision again and
  checked the tree against the content ID. `GET /api/status` reported that
  revision and content ID back.
- This source build has no default image for `vllm-sr decision serve`. That
  command would start the same entrypoint inside a container. Here it ran
  directly in a fresh virtual environment that mirrors the image's Vela
  environment:
  - the CPU Torch wheel the image uses;
  - the runtime package, installed editable from the checkout;
  - the image's `requirements-vela.txt`.
- The image's environment variables were set: `TOKENIZERS_PARALLELISM=false`,
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `PYTHONDONTWRITEBYTECODE=1`.
- The runtime picks float32 for CPU on its own. `DECISION_CPU_THREADS=8` is the
  variable that `decision serve --cpu-threads 8` passes to the container. It
  sets the OpenMP and BLAS thread counts before Torch loads.
- Server limits were the CLI defaults for Kai: max batch 8, max concurrency 8,
  max queue 32.
- The server ran as a transient user service capped at 6 GiB of memory with no
  swap. Peak cgroup memory was about 4.3 GB. It was stopped after the run.

## Environment

- CPU: AMD Ryzen 7 H 255 w/ Radeon 780M Graphics (8 cores, 16 threads).
- OS: Ubuntu 24.04.4 LTS, Linux 6.17.0-35-generic.
- Python: 3.12.3 for both the server and the client.
- Packages: torch 2.12.0+cpu, transformers 4.57.6, tokenizers 0.22.1,
  safetensors 0.8.0, huggingface-hub 0.36.0, fastapi 0.141.1, uvicorn 0.53.0,
  numpy 2.5.3, pydantic 2.13.5.
- The client, `run_kai.py`, uses only the standard library.
- Coarse location: South Korea. Server and client were on the same host, over
  loopback.

## Execution

- Requests were sent one at a time, one attempt each, with no automatic retries.
  The timeout was 60 seconds per request.
- Each request carried the case `state`, the model ID and one question under
  `intent`. The question is the parsed `question.json` object, with its
  instructions and criteria unchanged. Expected labels were never sent.
- One warmup request came first. Its text is outside the pilot set: "Describe
  how a bicycle gear system changes the effort needed to pedal uphill." It is
  recorded in `kai-warmup.jsonl`, not as a case. It returned `other` at 0.4223
  in 261 ms and passed the contract check.
- Stop policy: continue after a call or contract failure and record it. No
  failure occurred, so all six cases ran once.
- Timing boundary: the client clock starts just before the HTTP request opens,
  with the body already serialized. It stops once the response body has been
  read and parsed as JSON. Contract validation and file writes are not
  included. Server-only inference time was not measured.
- Contract check for each case:
  - exactly the 14 protocol labels;
  - every probability a finite number in [0, 1];
  - the probabilities sum to 1 within 0.001, with no renormalization;
  - the choice is one of the labels;
  - the echoed model ID matches.
- The probabilities are stored exactly as returned, and the raw HTTP body is
  kept next to them. The runtime's `confidence` is its own statistic, the
  margin between the top two probabilities. It is stored separately and was
  not used for scoring.
- Server start: 2026-09-29T06:16:02Z. Ready by 2026-09-29T06:16:22Z.
- Run start: 2026-09-29T06:17:42.777330Z. Run finish: 2026-09-29T06:17:44.279292Z.
  The script exited normally.

## Result

There are six records, each with one attempt, HTTP 200 and a distribution that
passed the contract check. Every probability sum was within 1e-7 of one. Each
case took 204 to 209 ms.

| Case | Expected | Native top-1 | Top-1 p | p(expected) | Match |
| --- | --- | --- | ---: | ---: | --- |
| pilot-001 | biology | other | 0.5017 | 0.0800 | no |
| pilot-002 | computer science | other | 0.3947 | 0.3004 | no |
| pilot-003 | math | other | 0.4136 | 0.0524 | no |
| pilot-004 | history | other | 0.3580 | 0.3008 | no |
| pilot-005 | health | health | 0.5097 | 0.5097 | yes |
| pilot-006 | not scored | other | 0.2904 | not scored | not scored |

One of the five scored cases matches its reference. In the four misses Kai put
the most mass on `other`. The expected label came second for pilot-002 and
pilot-004, fourth for pilot-001 and fifth for pilot-003.

Diagnostic case 006 returned `other` at 0.2904 with `physics` at 0.2894, a
margin of about 0.001. That is not an exact tie. Correctness is left out for
this case rather than set to false.

Six development cases do not establish accuracy or calibration for Kai.

After the run, one more request was sent to confirm the service itself behaved
normally. It was the two-label example from the runtime's API documentation,
and it returned `billing` at 0.915 as expected. It is not part of the records.
The server log shows eight `POST /v1/systemone` calls in total, all HTTP 200:
the warmup, the six cases and that one check.

## Artifact SHA-256

Kai transport-read fix (2026-10-05): subin's commit `f19e4c31a193770597a9b2847ae158ddc1222c8f` is integrated with authorship retained. Truncated successful HTTP responses preserve received bytes and record an `IncompleteRead` failure; connections closed without a response record a connection failure. The next input is still processed. A local HTTP server regression exercises both failures and subsequent valid cases through the real runner, without model inference. The current `run_kai.py` SHA-256 is `f865b80e81ab59a4be18020061206047f82a050df097326f8e111ea8f4f738a6`. Earlier source hashes below are historical snapshots; captured model records remain unchanged.

Kai review fix (2026-10-03): subin's commit `3bf6b49a7b9774ea9aed930186ab92fcafd4ca6b` is integrated with authorship retained. Non-object responses, null/non-object answers and invalid probability values now produce recorded contract failures with raw responses preserved; the runner continues through the remaining inputs. Top-1 calculations are performed only on contract-valid responses. The current `run_kai.py` SHA-256 is `3170f8b4b17b28639d291d1b9ae13100de143d75c6de1ee0f7dfee2099ea6eff`; the source hashes below describe earlier snapshots. Captured results and warmup records remain unchanged, and no model rerun is claimed.

Run the offline regressions with:

```bash
python3 -m unittest discover -s bench/jev/pilot-v0.1/tests -p test_kai_runner.py -v
```

The tests cover malformed bodies, null answers, string probabilities and integers beyond float range, plus a complete simulated run with four malformed responses followed by valid responses. All six case records must still be written.

The table below records the original run artifacts. On 2026-10-01, the two Python scripts were formatted with repository-pinned Black 25.1.0. Ruff cleanup in `run_kai.py` also names the existing 0.001 probability-sum tolerance and HTTP 200 constant, and uses list unpacking for two error lists. Saved-response and synthetic-failure checks agree with the original validator and request runner. The exact original sources remain available at commit `efb0e7143fa52f715b3de31f84625c2031ceb6b2`. Current source hashes are `ba9b3f4354c7e962ac03200410adf5c9eb639a83d60c14232ecc96585d2de00c` for `run_kai.py` and `e312e4a736db18d54ed83a4e8ae7f0da6924ca1c6a63510ae87389aef3096ee2` for `prepare_artifact.py`. Captured requests, results and warmup records were not changed or rerun.

| File | SHA-256 |
| --- | --- |
| inputs.jsonl | 671c09f62dc9fc9b864efe54b0adfef0ec666f309f74b776dcec3d6d8cdd2ef6 |
| question.json | 33f7ef7c4826df98a4bc23c2e0fc2302ea3ab61e18b8b45d0515e51e64f69738 |
| kai-results.jsonl | 889215a12db029a86aca66749b2a39418d914e118971e9c637eac17b4e924d63 |
| kai-warmup.jsonl | 4a5f872b9ce74d4e88e18d53488b3208061510890163fcf88bbf2846839442d4 |
| run_kai.py | b8fc9e19e37b1094185508944a755ccd0c204c58ff3dd07324d07f8752f04445 |
| prepare_artifact.py | 6d4e3ae3650cd3715e8c38f75d5b3401110a04d068579f60b125a9ffc4f8a05b |

## Reproduction

Run these from a checkout of the research branch, with `bench/jev/pilot-v0.1`
present. `SR` is a clone of vllm-project/semantic-router that has fetched the
PR branch, and `WORK` is any empty directory.

```bash
git -C "$SR" fetch origin xunzhuo/decision-runtime
git -C "$SR" worktree add --detach "$WORK/runtime" 58cd660b51ba19aff01ad6f89e66f2d07f13908b

/usr/bin/python3.12 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.0
"$WORK/venv/bin/python" -m pip install -e "$WORK/runtime/src/vllm-sr" \
  -r "$WORK/runtime/src/vllm-sr/decision_runtime/image/requirements-vela.txt"

HF_HOME="$WORK/models/hf" "$WORK/venv/bin/python" bench/jev/pilot-v0.1/kai/prepare_artifact.py \
  llm-semantic-router/Decision-1.0-Kai-0.6B 9d6872cde6950c2c2b5786d182ec9a06ca1bdd66 \
  "$WORK/models/artifacts"

CID=fab978e9f5e2ff799e1c2352b48e9c68d332ddeb05cdc8a25a88b69e0c28cb72
systemd-run --user --unit=kai-pilot --collect -p MemoryMax=6G -p MemorySwapMax=0 \
  -E PATH="$PATH" -E HOME="$HOME" -E DECISION_CPU_THREADS=8 \
  -E TOKENIZERS_PARALLELISM=false -E HF_HUB_OFFLINE=1 -E TRANSFORMERS_OFFLINE=1 \
  -E PYTHONDONTWRITEBYTECODE=1 --working-directory="$WORK" \
  "$WORK/venv/bin/python" -m decision_runtime.entrypoint \
  --model llm-semantic-router/Decision-1.0-Kai-0.6B \
  --revision 9d6872cde6950c2c2b5786d182ec9a06ca1bdd66 --backend cpu \
  --artifact-root "$WORK/models/artifacts/sha256/$CID" --artifact-content-id "$CID" \
  --host 127.0.0.1 --port 18431 --max-batch 8 --max-concurrency 8 --max-queue 32
until curl -sf http://127.0.0.1:18431/ready; do sleep 2; done

python3 bench/jev/pilot-v0.1/kai/run_kai.py --base-url http://127.0.0.1:18431 \
  --pilot-dir bench/jev/pilot-v0.1 --out-dir "$WORK/out"
systemctl --user stop kai-pilot
```

`run_kai.py` checks the input and question hashes against `protocol.json` and
exits before sending anything if they differ. CPU results can vary slightly
across hardware and thread counts. They are not guaranteed to match bit for
bit.
