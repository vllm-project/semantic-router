// Package gatewayparity holds the composition test of the native gateway: the
// routing core, the upstream layer and the frontend serving the parity corpus
// together, and a throughput benchmark of the same composition. It runs in its
// own test process, apart from cmd's tests, which shut down process-wide
// runtime state.
package gatewayparity
