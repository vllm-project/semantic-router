// Package configsnapshot compiles the canonical configuration document into
// immutable, versioned snapshots of typed resources, and it is the single owner
// of their lifecycle.
//
// A snapshot names every resource it holds (listeners, routes, programs,
// clusters, endpoints, secrets and runtime models) and records what each one
// references, so a broken reference is rejected before anything is built.
//
// Every update runs the same stages whatever its source (the config file, the
// management API or Kubernetes): compile, validate, warm and activate. A failed
// stage rejects the update with structured reasons (a NACK) and the active
// snapshot keeps serving. A successful update (an ACK) activates the candidate
// in one atomic step under the next version; the snapshot it replaces drains
// before what it alone used is closed.
//
// The package depends only on the configuration model and metrics. The
// Router's serving parts are built and activated through the Runtime
// interface, so the routing pipeline, the upstream layer and the gateway plug
// into the lifecycle rather than the other way round.
package configsnapshot
