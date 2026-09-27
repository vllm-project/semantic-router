# Lux 1.0 native JevArena v3 control: preflight HOLD

The proposed Lux 1.0 v3 control did **not** run on typed FINAL or CSS15.
Formal predictions, gold-free freeze, scoring and paired intervals are absent.
This is a model-package consistency hold, not a formal benchmark score. The
v3 panel was already post-key in the broader project; no formal gold was read
for this attempted control.

The exact published own-model package is
`llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8`.
Its bundle manifest SHA-256 is
`985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd`,
and its release manifest gives **7,940,895,744 deployed parameters**. The
native loader verified its HF revision, published package, exact runtime and
ROCm profile. The 32-item gold-free smoke completed with 32/32 answers across
Choice, Noul and Score, all under the qualified runtime.

## Failed preflight trace

| Attempt | Finding | GPU seconds |
| --- | --- | ---: |
| Generic pinned image | Native load stopped before predictions: `fla` absent. | 9.168 |
| Qualified Lux image, mistaken `PYTHONPATH` | Native load stopped before predictions because `/opt/decision-fla` was masked. The fixed path preserves both the release source and our adapter. | 9.243 |
| 32-item smoke on corrected runtime | 32/32 native answers, exact release runtime. An initially chosen older 32-item answer map differed by up to 0.0742 in Score probability. Its manifest then established that it was **a later Decision 2.0 LoRA checkpoint**, not Lux 1.0. That cross-model map was invalid as a consistency reference and was removed before any formal prediction. | 41.314 |
| Lux 1.0 package's own fixed example | Two released examples/5 answers, all categories match and runtime qualified, but one Score class probability differs by **0.0212843**, exceeding the prospective **0.0200000** numerical gate. | 39.445 |

The maximum drift is on the released three-level Score example's middle-class
probability: published `0.9581465871`, observed `0.9368623335`. Choice
probabilities differ by at most `0.005524` and Noul outputs by at most
`0.003118`. This is localized numerical variation, yet it exceeds the fixed
gate. The model remains **HOLD for this comparison**; neither threshold nor
published response was retroactively changed to pass. The four container
durations total **99.171309681 GPU-seconds = 0.027547586 GPU-hour**. The
assigned GPU is released.

The own-package example SHA-256 is
`12d68a696b50851b3614c1ed9d5e73347780ca5ad487ee2ad35e746ab40bbb18`;
gold-free release-example prediction SHA-256 and the exact new gate script
SHA-256 are retained in the private receipt and prospective pre-registration,
respectively. The release documentation itself warns that numerical kernels
may move probabilities near a boundary even with the same architecture.
The qualified runtime versions and model provenance match; the cause of this
particular Score probability difference remains unresolved. Any future Lux
v3 formal control needs a separately justified, prospective consistency
protocol and fresh gold-free prediction seal. JPT's v3 result remains an
independent peer result without a valid Lux paired interval.
