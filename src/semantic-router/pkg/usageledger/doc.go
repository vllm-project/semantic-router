// Package usageledger durably projects terminal quota-runtime stream events
// into the PostgreSQL usage ledger and verified analytical rollups.
//
// The package deliberately has no inference-side fallback: Redis delivery is
// at least once, PostgreSQL persistence is logically exactly once, and a
// stream item is acknowledged only after its complete ledger transaction and
// every affected analytical grain have committed. Unknown or partial provider
// usage remains explicit all the way through persistence and queries; it is
// never coerced to zero.
package usageledger
