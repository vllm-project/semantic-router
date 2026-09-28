# Node bootstrap and pinned evaluation runtime — 2026-09-28

Eval & peers track, Milestone 1 item 1. Machines are named node A and node B;
their addresses live only in the coordinator's private notes and in the local
private alias file read by the mirror script.

## Program root (both nodes)

`/data/dev2/{src,runs,leases,hf-cache,private,logs,tools}`; `private/` is mode 700.
`tools/` is an addition to the coordination layout and holds shared tooling only:

- `tools/hf-cli/`: Python venv with `huggingface_hub==1.33.0` (last 1.x release;
  2.0.0 was published but not adopted), linked as `/usr/local/bin/hf`.
- Secrets were provisioned exactly per the coordination "Secrets" procedure:
  HF token at `/root/.cache/huggingface/token`, Jev variables at
  `/root/.config/decision2/secrets.env`, both mode 600, values sent over stdin only.
- Use `export HF_HUB_CACHE=/data/dev2/hf-cache` for new downloads. Node B's root
  filesystem is 97% full; keep large files under `/data`.

## Hugging Face access (read-only checks, both nodes)

| Check | Result |
| --- | --- |
| `hf auth whoami` | user `Xunzhuo`; orgs `llm-semantic-router`, `agentic-in` (identical on both nodes) |
| Private collection "Decision 2.0" | visible, private, **zero items**; last updated 2026-09-28T01:39:19Z |
| `llm-semantic-router/DEV2.0-0.6B` | **not found** on the Hub; a local HF-cache snapshot of revision `7ac568e6ce99cdeb7cf423b4984a4f87dba8a204` remains on node A |
| Private dataset `llm-semantic-router/decision-2.0-training-data` | visible, private, revision `b81b78e5477ed712418efda4db26a631ebd2bd2d`, 12 files |
| Decision 1.0 collection | public; Kai, Lex, Eos, Sol, Nox, Lux listed |

The empty collection and missing DEV2.0-0.6B repository contradict the
2026-09-28 release-status note (collection containing only DEV2.0-0.6B); this is
reported to the coordinator rather than repaired.

## Network

Both nodes reach `github.com`, `huggingface.co` and `pypi.org` directly. From this
workstation the full source archive streams to node A at about 7 MB/s but to
node B at about 0.5 MB/s, so a node B mirror takes roughly 15 minutes.

## Pinned evaluation runtime

Node A image `decision20-train-fast:host2`, ID
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`
(created 2026-09-26T12:45:56Z; the image used by the JPT-9B peer and most prior
native evaluations):

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| PyTorch (ROCm) | `2.12.0+git6bbd260`, HIP `7.2.53211` |
| Triton | 3.7.1 |
| Transformers / tokenizers / safetensors / NumPy | 5.17.0 / 0.23.2 / 0.8.0 / 2.3.5 |
| FLA | 0.5.2, hash-locked overlay at `/opt/decision-fla` |
| Other | huggingface_hub 1.31.0, accelerate 1.15.0, peft 0.21.0, vLLM 0.29.1rc1.dev187 |

Per-package tree digests (SHA-256 over sorted per-file SHA-256, `.pyc` excluded)
are byte-identical between this image and node B's qualified Lux image
`decision20-lux-runtime:latest` (`sha256:ce895822fc48bb6864911d4488a3946f3a18fd3dd2ec90c8a0a49b259145f2fb`,
used for the Lux r3/r4 runs):

| Tree | Files | Digest prefix |
| --- | ---: | --- |
| `/opt/decision-fla` | 479 | `0e9ad3ea6df0662b` |
| `dist-packages/torch` | 12,177 | `0d03337c98ab6aa1` |
| `dist-packages/triton` | 353 | `4db7e70a59e64187` |
| `dist-packages/transformers` | 2,682 | `fe5654bdc44b3323` |
| `dist-packages/tokenizers` | 28 | `ffb546a87f7be15d` |
| `dist-packages/safetensors` | 10 | `dfefd388eba810ca` |
| `dist-packages/numpy` | 892 | `45619355974e8184` |
| `/opt/rocm/lib` | 6,502 | `0706a2421e579506` |

So the Lux/Nox/Sol/Eos qualified ROCm+FLA stack is available on node A without
moving the 49 GB image. Decision 1.0 Kai and Lex use their qualified isolated
venv `/data/decision20-20260926/envs/kai-lex` (system site packages from the same
image; Transformers 4.57.6, tokenizers 0.22.2, NumPy 2.5.3, safetensors 0.8.0),
which matches their release validation runtime. Other external peers use this
image unless their own record pins something else (for example the APUS
Transformers 5.16.1 image).

## Decision 1.0 package identity (current Hub revisions)

All six Decision 1.0 repositories were republished on 2026-09-27 as model-only
repositories ("Publish clean Decision model repository"). Their current `main`
revisions keep the weights, tokenizer, temperature and decision config but no
longer ship the native runtime code the published loaders import
(`src/decision/`, `bundle-manifest.json`, `runtime.json`, Eos `decision/` and
`MODEL_MANIFEST.json`, Kai/Lex `decision_runtime`). The new cards point to a
separately distributed vLLM Semantic Router Decision runtime that is not in the
public repository.

| Model | Current `main` | Last runtime-bearing revision used by prior native runs | Functional files between the two |
| --- | --- | --- | --- |
| Kai 0.6B | `9d6872cde6950c2c2b5786d182ec9a06ca1bdd66` | `7185f514f54b8f93c55998b1e8f9c5cc67f0d029` | `native/` weights, config and tokenizer unchanged |
| Lex 0.6B | `6c5e3d48b9e67cd8bddbade3277e2e58506af8f0` | `ee8e74d912fca8328a353c11d174b44da3f91781` | `native/` weights, config and tokenizer unchanged |
| Eos 0.8B | `363c4a5e56afc115b1c78c837633956d0bbb63ab` | `3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd` | backbone, head, decision config and tokenizer unchanged |
| Sol 2B | `ce0c018a28de16d6639b1cd203b761bf643b89e6` | `0665a41108e8f0b33a9515c98311c45947b99399` | weights, temperature, tokenizer unchanged; `decision_config.json` differs only by the removed `runtime_file` key |
| Nox 4B | `cde2a68dbaa557ea65dc458104d410a0802ee259` | `0bb833504965c0eabdb9630b7bbd385cb2fe5cd4` | as Sol |
| Lux 9B | `cdf4d3ef2dda21518e599fe99ebbe468486b197c` | `bd45a30aee8c84032791c245c70f86dee5389cc8` | as Sol |

Comparison used the Hub tree API (LFS SHA-256 or blob ID per file) and a
key-by-key diff of `decision_config.json`. The current published 1.0 weights are
therefore byte-identical to the runtime-bearing revisions, and the only runnable
native path for them is each model's own runtime from that revision. Same-panel
1.0 baselines are pinned as "current `main` weights, native runtime of the
runtime-bearing revision"; results produced on the runtime-bearing revision are
reusable for the current package by content identity.

## Exact mirror

`src/training/decision2/v2/common/mirror_to_node.sh <node-alias> <commit> [<target-root>]`
materializes `/data/dev2/src/<full-sha>/` from `git archive` of a pushed commit,
checks the archive digest in transit and a per-file content manifest after
extraction, writes `.dev2-mirror.json` (commit, tree, archive and manifest
SHA-256, file count, UTC time) and marks the files read-only. `--verify`
re-checks an existing mirror; re-running is idempotent. Aliases `node-a` and
`node-b` resolve through the private `~/.config/decision2/nodes.env`.
The pinned handoff commit `be472b9575ef15d19040d7d29206aed87b480458` (tree
`808cff7327335837a7de2ccd64f46c4ba7433c3b`, content manifest
`bfb30c7603d0d052422d241d6cdacea5b54da7d3dcfbc77e464d64cf688fd388`, 11,492 files)
was mirrored to node A in 97 seconds.

GPU-hours for this bootstrap: zero (CPU-only container checks).
