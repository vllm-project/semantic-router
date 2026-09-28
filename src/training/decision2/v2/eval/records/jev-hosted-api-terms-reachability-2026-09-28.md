# Jev hosted API: terms, reachability and reference decision — 2026-09-28

Eval & peers track, Milestone 1 item 5. No model Output was generated: the
reachability checks used unauthenticated requests, requests with an empty body
and the model listing only. Credentials were loaded from the private secrets
file inside the probe process and never printed; endpoint values are withheld.

## Terms (read 2026-09-28)

| Document | Version | Relevant clauses |
| --- | --- | --- |
| TypeSafe Master Customer Agreement (governs API use), `typesafe.ai/legal/mca` | Last updated **Sep 23, 2026** | §2.3(b): the customer will not "use the Services or any Output … to perform model distillation, train a model to imitate the output of the Services, or develop (or to facilitate the development of) a similar or competing product or service". §2.3(j) usage limits; §2.3(l) Acceptable Use Policy. The earlier Aug 27, 2026 text also banned publishing benchmarks (§2.3(f)); that clause is absent from the current text. |
| TypeSafe Acceptable Use Policy | Sep 23, 2026 | No benchmark or evaluation clause; bars scraping, circumvention and misleading AI-content claims. |
| TypeSafe site Terms of Use | site only | Do not govern API use. |
| Mirror provider terms (independent reseller forwarding to TypeSafe; not affiliated with TypeSafe) | Sep 21, 2026 | §5: users may not use the service to violate "the upstream provider's terms", so MCA §2.3(b) applies to mirror Output as well. Model version and availability are controlled upstream. |

**Teacher targets: not allowed.** Using Jev answers or probabilities as training
targets, soft labels or distillation data is exactly what MCA §2.3(b) forbids,
through the official API or the mirror. The research & data track must not
admit Jev Output into any TRAIN, SELECT or CAL artifact.

**Evaluation reference: not run in Milestone 1.** Publishing benchmark numbers
is no longer expressly prohibited, but Decision 2.0 serves the same System One
Choice/Noul/Score interface, so running Jev on our panels to guide Decision 2.0
work is plausibly "use of the Services … to facilitate the development of a
similar or competing product". That is a legal judgment for the user (or written
permission from TypeSafe), not an engineering gate, so no v3/public-231 Jev run
was made. If the user approves, the reference must be labelled separately from
open-weight same-size ranks and kept out of rank charts and training decisions.

## Reachability and limits (no Output generated)

| Probe | Official API (`/v1/systemone`) | Mirror (`…/decide`) |
| --- | --- | --- |
| No credential | HTTP 403 | HTTP 401 ("Missing API key") |
| Credential, empty body | HTTP 422 (validation), credential accepted | HTTP 400 ("`state` is required"), credential accepted |
| Model listing (`/v1/models`) | HTTP 200: `jev-latest` and `jev-preview`, both released 2026-09-10 | — |
| Rate-limit headers | none exposed (Cloudflare) | none exposed |
| Documented limits | 429 Too Many Requests / 529 Overloaded with exponential backoff; per-account usage limits come from the Order, no public numeric rate | reseller may throttle |

Only moving aliases are offered (the `jev-1.13.0` pin used by the older client
is not listed), so a Jev run could not be pinned to an immutable model version;
any approved reference must record the alias, date and response `model` field and
treat results as non-reproducible.

GPU-hours: zero.
