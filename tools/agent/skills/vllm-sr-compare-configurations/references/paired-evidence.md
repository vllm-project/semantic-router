# Paired evidence examples

These invented IDs and values illustrate how to read sr-bench receipts. They
are not project benchmark results. Replace them with actual `benchmark show`,
`comparison-options`, `compare` and `report` output before making a claim.

## Compatible calculations, weak verdict

| Frozen field | `single-001` | `route-a-001` | `route-b-001` |
| --- | --- | --- | --- |
| Mode / target | live / single model | live / Router config A | live / Router config B |
| Cases and request profile | Same two IDs and input digests; temperature 0, 256 output tokens | Same | Same |
| Grader and prices | Same MCQ grader and four-bucket prices | Same | Same |
| Router config identity | N/A | Hash A, matching every saved receipt | Hash B, matching every saved receipt |
| Outcomes | 2 planned, 2 graded, 0 missing | Same | Same |

`comparison-options single-001` may list both Router runs. Run
`compare single-001 route-a-001` and `compare single-001 route-b-001` and
preserve both responses. The service reports paired quality intervals for
each Router configuration against the single-model reference. It does not
report a direct A-versus-B interval. Do not subtract the two intervals and
call the difference an A/B verdict.

Even if every run grades 2/2, two development cases cannot prove quality
non-inferiority or a general saving. Report each actual interval, prices,
costs and latency, then classify a broad quality claim as **insufficient
evidence**. A narrower smoke claim can pass: the planned cases ran and were
graded under the frozen protocol.

## Inconclusive or rejected candidate

| Frozen field | `single-002` | `route-c-002` |
| --- | --- | --- |
| Cases | Same two IDs | Same two IDs |
| Mode / grader | live / MCQ grader v2 | live / MCQ grader v2 |
| Outcome | 2 graded | 1 graded, 1 missing |
| Cost | Complete prices and usage | One response has unknown usage |

The missing candidate cell blocks comparison; unknown usage also blocks a
cost-saving claim. Do not score the missing answer as zero or omit its case
from the paired denominator. Read that cell's saved call and run status,
recover it through sr-bench if eligible, and keep the recovery child linked
to its parent. If the candidate also changed the grader, request parameters
or case input, the service should reject compatibility; restore the same
frozen protocol rather than treating aggregate averages as paired.

## Report handoff

Record the single-model and Router run IDs, their frozen plan and dataset
hashes, each config and model revision, planned/paired/failed/missing/graded
denominators, both service comparison responses, quality intervals, latency
and cost basis, routing and guard deltas where recorded, the prespecified
margin, and the exact reason for pass, fail or insufficient evidence. Link
the raw saved call IDs for exceptions while redacting private prompts and
endpoints.
