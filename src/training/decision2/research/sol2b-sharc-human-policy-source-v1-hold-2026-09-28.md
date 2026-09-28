# Sol 2B ShARC source v1: exact-rule pairing fails

**HOLD.** The prospective [source screen](sol2b-sharc-human-policy-source-prereg-2026-09-28.md)
ran once on the publisher's TRAIN archive using the exact local script mirror
from signed commit `fb15bb986`. It generated no training rows, model outputs
or score and consumed zero GPU-hours. The private aggregate receipt remains
on the authorized experiment node; no original snippets, answers or source
paths are included here.

| Input-only inventory | Result |
| --- | ---: |
| Publisher TRAIN rows and unique utterances | 21,890 / 21,890 |
| Distinct source `tree_id` values | 628 |
| Trees with inconsistent exact `(source_url, snippet, question)` tuple | 628 |
| Eligible same-tree Yes/No Choice pairs under the frozen rule | **0** |
| Feasibility floor | 100 pairs, 50 source URLs, 100 snippets |

The archive SHA-256 was
`72dca3f4f3ba73b1d796b40e952a80d53cd2011ef90b2168b8bcaa818f5edd1e`;
the TRAIN member SHA-256 was
`d37d349758a69644fbf49827b1a0893f6099752b0ccc9c7e29bd315e5228429c`;
the mirrored audit script SHA-256 was
`ddc1d1659620e969de8b3e92c7ca2e4950df1e57f9531c67ca5c7e583af867da`.
The private receipt SHA-256 is
`bb1350ab34732d90acdd8b763df4e6f06c22b1e4f0f4ffd1984856fcccd40c77`.

This strict exact-tuple filter found no usable pairs. The result alone cannot
distinguish minor source-field variation from substantial within-tree rule
changes. Its full answer-class inventory also includes publisher follow-up
question text, so the receipt stays private and is not copied into the gist or
model card. Any normalized or separately regrouped source analysis would be a
new version with a frozen rule and new checks; it cannot retroactively turn
v1 into a pass. No Sol 2B GPU arm is authorized by this screen.
