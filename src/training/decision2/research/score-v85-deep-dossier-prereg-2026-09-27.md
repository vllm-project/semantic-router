# Score v8.5 deep-dossier CPU pilot: prospective design

**Frozen before generating a v8.5 row.** This 90-row quality pilot addresses the v8.4 independent review HOLD. It does not admit data for training, run models, or make a release claim. The private 32-byte seed has SHA-256 `b70438ffd2a2d8cb6b9c9640ea7567ae14d9b7a5534fb1327adb9e7c5dae816f`. Do not reroll it or replace unfavorable groups.

## Pool and construction

Use six mechanisms: workflow readiness, timed connection, stock fulfillment, versioned policy, two-source evidence, and business-day service obligation. Build five independently seeded case groups per mechanism, with three TRAIN and two SELECT groups. Each group has one counterfactual row for each Score level 0, 1, and 2. Group, case, item and document identifiers are unique across roles; role and group streams are HMAC-separated. Use 18 English and 12 original Chinese groups. Shuffle document order and the order of three variants. Do not encode role or label in review IDs or visible text.

Each mechanism retains an explicit, fixed scoring rule while varying the underlying cases:

- Workflow: vary two to four transitive ancestors. A failed ancestor gives 0, an unfinished ancestor without failure gives 1, and all complete gives 2. A later signed reconciliation supersedes an older board; no live descendant may be complete while its ancestor fails.
- Connection: vary named service and route, walking time, arrival interval and revised boarding cutoff. The earliest adjusted arrival beyond cutoff gives 0; an interval straddling cutoff gives 1; the latest adjusted arrival at or before cutoff gives 2.
- Stock: vary the requested SKU position among two or three items, on-hand stock, reservations, confirmed arrivals and unconfirmed arrivals. Confirmed plus unconfirmed still short gives 0; only unconfirmed units closing the gap gives 1; confirmed usable units meeting demand gives 2.
- Policy: vary application region and date relative to a scoped inspection exception and a training amendment. Expired license or another confirmed unmet mandatory requirement gives 0; a current requirement with explicitly pending proof gives 1; all applicable requirements satisfied gives 2. The effective date and region must matter in at least one case each.
- Evidence: two independent attributed sources must support a named proposition. An explicit contrary observation gives 0; a missing required observation without contrary evidence gives 1; two affirming observations give 2. Express the findings as natural source prose, not one-word outcome codes.
- Service level: vary request weekday, local holiday and two-to-four-business-day term. Late fulfillment gives 0; timely fulfillment with an unconfirmed prerequisite gives 1; timely fulfillment with the prerequisite confirmed gives 2.

In at least 24 groups, append a substantial, case-specific dossier after four core sections. The dossier must contain a decisive, level-varying, target-specific source fact at a seeded deep position, never in the four core sections. Include a same-field near-target record and an older target record with clear dates or scope, so reading the target identity and controlling source is necessary. Other dossier material must be plausible for that case rather than cross-domain padding. The decisive fact must change across the triplet while surrounding case facts stay fixed. At least 24 rows should exceed 700 tokens and six should exceed 1,500 tokens under the frozen tokenizer; failure is HOLD, not a reason to add inert padding. Six short groups may keep all operative facts in four sections.

## Frozen CPU checks and review boundary

Compute the label from structured facts before rendering. Parse the rendered text independently to recompute all 90 labels, with an explicit check that deleting the deep dossier makes each long item undecidable. Check identifier targeting, latest-source precedence, arithmetic/date boundaries, copy quality, Chinese phrasing, 0/1/2 group balance, and group-wise role separation. Scan normalized exact and bounded near overlap between roles, earlier Score pilots, parent partitions, and the available gold-free protected inventory. Record inaccessible references as a limitation and HOLD. Keep raw seed, labeled rows, key, and full audit private.

Freeze the candidate, generator hash, audit receipt and one **answer-free combined packet** before another reviewer begins. The handoff manifest may contain only file hashes, schema, item/group counts and a UTC seal; it must omit class counts, role names and per-group partition designations. The independent reviewer must answer every item and inspect source necessity, ambiguity, shortcuts and language quality before seeing any key. A systematic shortcut, unresolved contradiction, source leakage or reviewer disagreement keeps the pilot on HOLD. No data enters training or release from this pilot alone.

## Tokenizer freeze before candidate materialization

Count state, instruction and options with `Qwen/Qwen3.8-27B` tokenizer revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The tokenizer JSON has SHA-256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`; its configuration has SHA-256 `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`. This addendum precedes the first materialized candidate.
