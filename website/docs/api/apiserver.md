# Router Apiserver API Reference

Router apiserver is the HTTP control surface served on `:8080`.

## Base URL

`http://localhost:8080`

## Live Schema

- `GET /api/v1`: discovery index
- `GET /openapi.json`: live OpenAPI schema
- `GET /docs`: Swagger UI

Use `/openapi.json` or `/docs` for the exact live request and response schema. This page is a compact reference to the currently documented surface.

## Discovery and Health

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/ready` | Readiness endpoint that turns green only after startup completes |
| `GET` | `/api/v1` | API discovery and documentation |
| `GET` | `/openapi.json` | OpenAPI 3.0 specification |
| `GET` | `/docs` | Interactive Swagger UI documentation |

## Classification

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/classify/intent` | Classify user queries into routing categories |
| `POST` | `/api/v1/classify/pii` | Detect personally identifiable information in text |
| `POST` | `/api/v1/classify/security` | Detect jailbreak attempts and security threats |
| `POST` | `/api/v1/classify/fact-check` | Classify whether text needs fact-checking |
| `POST` | `/api/v1/classify/user-feedback` | Classify user feedback type |
| `POST` | `/api/v1/classify/combined` | Combined classification endpoint |
| `POST` | `/api/v1/classify/batch` | Batch classification with `task_type` selection |

## Models and Info

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/info/models` | Get information about loaded models |
| `GET` | `/info/classifier` | Get classifier information and status |
| `GET` | `/v1/models` | OpenAI-compatible model listing |
| `GET` | `/metrics/classification` | Get classification metrics and statistics |

## Router Config

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/config/router` | Get the current router config as JSON |
| `PATCH` | `/config/router` | Merge a router config update |
| `PUT` | `/config/router` | Replace the router config |
| `GET` | `/config/router/versions` | List available router config backup versions |
| `POST` | `/config/router/rollback` | Roll back to a previous router config version |

## Config Semantics

- `GET /config/router` returns the current router config document.
- `GET /config/router` returns canonical config only. DSL-only authoring constructs such as `DECISION_TREE` / `IF ELSE` are already lowered into flat `routing.decisions`.
- `PATCH /config/router` uses merge semantics.
- `PUT /config/router` uses replace semantics.
- `PATCH` and `PUT` both validate, back up, write, and hot-reload before returning.
- The optional `dsl` field on `PATCH` / `PUT` requests is archived as source text for audit trail only. It does not extend the runtime config schema or restore tree metadata on later reads.

## Notes

- The endpoint list above mirrors the router apiserver discovery surface exposed by `GET /api/v1` and `GET /openapi.json`.
- Some documented endpoints may still be placeholder implementations in the current runtime, such as `POST /api/v1/classify/combined` and `GET /metrics/classification`.
