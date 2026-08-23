// Package backenddispatch composes the process-owned HTTP dispatch boundary
// used to invoke physical model backends from immutable routing snapshots.
//
// The package owns capability verification and transport lifetime. It borrows
// snapshot, credential, protocol, journal, and response-accounting services;
// none of those dependencies can be replaced by request input or a fallback.
package backenddispatch
