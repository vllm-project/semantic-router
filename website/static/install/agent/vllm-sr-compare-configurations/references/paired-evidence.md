# Paired evidence examples

These invented IDs and values illustrate how to read sr-bench receipts. They
are not project benchmark results. Replace them with actual `benchmark show`,
`comparison-options`, `compare` and `report` output before making a claim.

## Compatible calculation, weak verdict

| Frozen field | Run `base-001` | Run `cand-001` |
| --- | --- | --- |
| Mode / role | live / dev | live / dev |
| Cases and request profile | Same two IDs and input digests; temperature 0, 256 output tokens | Same |
| Grader and prices | Same MCQ grader; complete four-bucket prices | Same |
| MoM config identity | Hash A, matching every saved receipt | Hash B, matching every saved receipt |
| Outcomes | 2 planned, 2 graded, 0 missing | 2 planned, 2 graded, 0 missing |

The service may accept this pair and report `paired_cases: 2`. A different
config hash is the intended treatment. Even if both runs grade 2/2 and the
candidate costs less, two development cases cannot prove quality
non-inferiority or a general saving. Report the actual paired quality
interval, prices, costs and latency from the service, then classify a broad
quality claim as **insufficient evidence**. A narrower smoke claim can pass:
the two planned cases ran and were graded under the frozen protocol.

## Inconclusive or rejected pair

| Frozen field | Run `base-002` | Run `cand-002` |
| --- | --- | --- |
| Cases | Same two IDs | Same two IDs |
| Mode / grader | live / MCQ grader v2 | live / MCQ grader v2 |
| Outcome | 2 graded | 1 graded, 1 missing |
| Cost | Complete prices and usage | One response has unknown usage |

The missing candidate cell blocks comparison; unknown usage also blocks a
cost-saving claim. Do not score the missing answer as zero or omit its case
from the paired denominator. Read that cell's saved call and run status,
recover it through sr-bench if eligible, and keep the recovery child linked
to its parent. Freeze a new compatible comparison after evidence is complete.
If the candidate also changed the grader, request parameters or case input,
the service should reject compatibility before a paired result; restore the
same frozen protocol rather than treating aggregate averages as paired.

## Report handoff

Record the two run IDs, their frozen plan and dataset hashes, each config and
model revision, planned/paired/failed/missing/graded denominators, the service
comparison response, quality interval, latency and cost basis, routing and
guard deltas where recorded, the prespecified margin, and the exact reason
for pass, fail or insufficient evidence. Link the raw saved call IDs for
exceptions while redacting private prompts and endpoints.
