# Official-source 4B: v3/public predictions sealed before scoring

The three candidate runs authorized by the
[post-key lock](qwen35-4b-official-best466-v3-postkey-lock-2026-09-27.md)
finished with exit 0 on three independently reserved GPUs. All candidate
prediction files and manifests, plus the six archived Nox/Kev control files,
were revalidated and sealed **before this run accessed any formal or public
answer key**. The private seal was created at `2026-09-27T14:07:25Z`, SHA-256
`20d1e222c00878b3d46c1b3966327ffb5df9c19a2117c1c9e31add1046695156`.
It binds lock `ab2602d5a1aa7b3284a61f207fec45c7758b2ba879c12680e5fd8308b4fbd494`,
roster `26436f5754d6e33e54ec9ee27a6dc64742fc1e2762b6b95565e63b22a60775fb`,
and sealer source `d49b528818802647330cb4ec075c5bb93617fb7448be791e4085194ed5fef89f`.

| Gold-free panel | Items | Answer slots | Valid | Invalid / over-budget | Prediction SHA-256 | GPU-hours |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Typed FINAL | 1,600 | 2,000 | 2,000 | 0 | `09f4b1f826f69edc05fcf11092cf7413f400df744edb70903323ceba3e2b4485` | `.041582` |
| CSS15 human transfer | 6,547 | 6,547 | 6,529 | 18 / 18 | `d22ee8d0bd4cdb8ec130ae7299971b0534647a9cd20780d96a8a93e4896e9638` | `.158350` |
| Public JevBench subset | 231 | 231 | 231 | 0 | `8240e3890d23139be54f46ca20b7d724b896d18c753e56a5d78b98c22cb1780a` | `.016913` |

The three reads consumed **`.216845 GPU-hours`** in aggregate. They ran
concurrently, starting at `13:57:23Z`; public ended `13:58:24Z`, typed
`13:59:53Z`, and CSS15 `14:06:53Z`. The 18 long CSS inputs exceeded the
locked 8,192-token budget. They must remain failures at the full denominator;
no truncation, deletion or enlarged-context rerun is permitted under this
protocol. The model, calibrated adapter and prompt identities match the
prospective lock; all three prediction manifests report zero truncation.

This is a **prediction-seal receipt**, not a performance or release result.
The panel has had prior project label access, so subsequent scores are
post-key same-panel evidence. Scoring may begin only after separate seal
review and must use the unchanged +3.0 own-Nox aggregate gate, formula,
invalid policy and paired uncertainty rule in the locked roster.
