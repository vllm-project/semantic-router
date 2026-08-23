// Package accesspublisher publishes PostgreSQL desired state into immutable,
// revisioned Redis runtime projections.
//
// Publication is deliberately separate from both the management mutation path
// and the inference hot path. Management commits desired state and an outbox row
// in one PostgreSQL transaction. This package consumes only committed outbox
// work, compiles a complete namespace image, installs conservative deny
// barriers, stages content-addressed documents, waits for data-plane
// acknowledgements, and changes one namespace publication gate. Per-key and
// per-credential pointers are promoted after gate activation in retry-safe
// batches; activation never performs a 10,000-key atomic rewrite.
package accesspublisher
