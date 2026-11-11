# System One API specifications

Every System One API surface the Router exposes, calls or forwards to follows
one of these specifications. The generators, the codecs and the conformance
tests read them from here.

| File | Schema ID | What it is |
| --- | --- | --- |
| `systemone-0.2.schema.yaml` | `systemone/0.2` | The standard: our description of the public System One API, version 0.2. It records the interface's endpoints, field names, types and limits in our own words. |
| `vllm-sr-5.0.extensions.yaml` | `vllm-sr/5.0` | `systemone/0.2` plus the vllm-sr extension set 5.0: every field vllm-sr adds, where it goes, its type, the version it appeared in, the families that honour it, and what a standard target does with it. |
| `vllm-sr-4.0.openapi.yaml` | `vllm-sr/4.0` | The model runtime contract 4.0.0, frozen: the legacy schema a server not yet upgraded speaks, and the superset a `vllm-sr/4.0` listener reads. |
| `implicit-roles.json` | `vllm-sr/4.0` | The role 4.0's implicit roles give each key of an object state, whatever its case. The Router and the model runtime test against the same cases, so roles the Router writes for a 4.0 request read exactly as the request did. |
| `no-valid-question.json` | all | Requests to `POST /v1/decisions` with no valid question, and the model runtime's answer to each. The Router's tests check that it gives the same answers whatever schema its upstream speaks. |

## The standard's description

The standard is published as an OpenAPI 3.1 document at
`https://api.typesafe.ai/openapi.json`. The description here was checked
against the version whose sha256 its `source_sha256` records. The published
document carries no license, so it is not copied into this repository.

`make systemone-schemas-drift` downloads the published document when it is
reachable, into a cache outside the repository, and fails when its structure
differs from the description. A new version of the standard is a new
description file and a new schema ID (for example `systemone/0.3`); an older
one stays while a deployment may name it.

## Regenerating

After changing a specification, run `make systemone-schemas-generate`. It
writes the Go tables, the model runtime's field tables, the canonical JSON
golden file and the generated reference pages.
`make systemone-schemas-check` fails when one of them is stale, or when a
field is defined by both specifications with different definitions.

The maintainer guides under `website/docs/community/system-one-schemas/` say
how to add a schema codec, a deviation or an extension field.
