// Package accessruntime authenticates each inference credential once into an
// opaque request session, evaluates its immutable access projection, and
// submits every authorization, discovery, and quota decision through atomic
// live-state guards.
//
// The package deliberately has no Dashboard dependency and keeps no positive
// authorization cache. A session retains no raw credential or verifier, and
// every operation fails closed when the shared publication, policy, key, or
// resource state changes.
package accessruntime
