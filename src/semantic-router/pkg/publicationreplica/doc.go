// Package publicationreplica keeps one Router process aligned with the active
// immutable routing publications in Redis.
//
// It discovers namespaces through a bounded directory, verifies and warms
// candidate snapshots, acknowledges them under renewable leases, and switches
// a namespace only after the coupled access/routing gate becomes active. The
// package never queries PostgreSQL and never authorizes a request by itself.
package publicationreplica
