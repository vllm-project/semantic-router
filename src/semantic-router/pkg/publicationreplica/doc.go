// Package publicationreplica keeps one Router process aligned with the active
// immutable routing publications in the configured publication authority.
//
// It discovers namespaces through a bounded directory, verifies and warms
// candidate snapshots, acknowledges them under renewable leases, and switches
// a namespace only after the coupled access/routing gate becomes active. The
// package depends only on the Store contract: PostgreSQL backs durable routing
// without native access, while the distributed runtime store backs native
// access. It never authorizes a request by itself.
package publicationreplica
