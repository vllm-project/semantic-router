// Package quotaruntime applies compiled quota rules to one Redis or Valkey
// partition. It owns read-only access checks, atomic admission, dispatch
// journaling, mixed-evidence finalization, exact counters, unknown-usage
// fences, concurrency release, usage-stream publication, and live meter reads.
//
// Policy persistence and request routing deliberately live outside this
// package. Every operation goes to the shared store; the runtime has no local
// positive cache.
package quotaruntime
