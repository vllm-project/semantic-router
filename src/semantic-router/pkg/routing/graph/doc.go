// Package graph executes request graphs: the model calls the Router makes
// itself while it serves one client request, such as a Looper algorithm's
// panel, escalation or synthesis.
//
// A Program is a sequence of steps. Container steps (parallel, branch, loop
// and subgraph) hold sequences of their own, so a graph nests the way its
// control flow does. Every step is a Node of a registered type; the built-in
// types are call, parallel, aggregate, branch, loop, transform and respond,
// and other packages register more through Nodes.
//
// Run executes a Program for one request with one Exec: a single deadline (the
// request's), a hop limit and token and cost ceilings that fail the run
// closed, cancellation that reaches every in-flight hop, one trace span per
// step, and bounded, content-free attempt evidence. Each model call is a hop
// that a Caller sends; SessionCaller sends it through the routing core in
// process, so the hop runs its own plugin chain and upstream policy without
// a request on the wire.
//
// The package imports no transport: the routing contract is its only view of
// the routing core, and the upstream layer reaches it through the Upstream
// interface it declares.
package graph
