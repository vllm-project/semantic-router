// Package upstream sends routed requests to model backends in process, doing
// what the CLI-managed Envoy does for upstream traffic: one cluster per
// provider model alias, selected by the route key (the x-selected-model
// value), with the endpoints, load balancing, connection pools, host and path
// rewrites, route headers and upstream TLS that the Envoy template renders.
//
// A Set is an immutable view of every cluster compiled from one router
// configuration. Build compiles a config into a new Set; when it replaces an
// older Set, clusters whose specification did not change carry over with
// their connection pools and endpoint state, and the older Set drains its
// in-flight requests when it is closed.
//
// Set.Do returns once the response is ready to commit to the client: the
// response headers have arrived, and the first body byte too when the
// first-byte timeout is set. Every attempt before that point is recorded on
// the response. The body then streams without read-ahead, so a slow client
// slows the upstream instead of growing a buffer.
//
// The package never imports Envoy types or the ext_proc pipeline: the native
// frontend and the request-graph executor call it directly.
package upstream
