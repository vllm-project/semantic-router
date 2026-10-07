// Package routing is the transport-agnostic contract of the Router's routing
// core, shared by every gateway mode.
//
// A request moves through four phases in every mode: request headers, request
// body, response headers and response body. A Session processes those phases
// for one request and answers each with an Effect: header and body mutations,
// a route-cache refresh, a switch to a streamed response body, or an
// immediate response. The ext_proc adapter encodes effects as ext_proc
// messages and Envoy applies them. The native gateway uses Engine, which
// drives a Session with the processing mode of the local Envoy template and
// applies effects with Envoy's rules, so both modes forward the same upstream
// request and return the same client response.
//
// This package imports no Envoy types and no transport; adapters depend on it,
// never the reverse.
package routing
