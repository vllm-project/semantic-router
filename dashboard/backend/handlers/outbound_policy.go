package handlers

import (
	"net/netip"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/safefetch"
)

// Hard ceilings on what a caller-supplied fetch may return. These bound the
// decoded body: safefetch disables compression so a small archive cannot
// expand past them.
const (
	openWebMaxResponseBytes     int64 = 8 * 1024 * 1024
	fetchRawMaxResponseBytes    int64 = 2 * 1024 * 1024
	setupImportMaxResponseBytes int64 = 1 << 20
)

// setupImportTimeout bounds a remote config import.
const setupImportTimeout = 10 * time.Second

// outboundResolver is swapped in tests to drive address classes and answer
// changes without real DNS. Nil means the system resolver.
var outboundResolver safefetch.IPResolver

// outboundAllowedPrivatePrefixes is the declared-internal-target allowlist.
// Empty in production: reaching a private destination is an operator decision
// expressed as a narrow prefix, and there is no configuration surface for it
// yet. Tests that serve fixtures from loopback declare it here.
var outboundAllowedPrivatePrefixes []netip.Prefix

// outboundPolicy is the single outbound-fetch policy for every handler that
// dials a destination the caller chose. Consumers adjust the deadline and the
// accepted schemes; the address check is not theirs to relax.
func outboundPolicy(timeout time.Duration) safefetch.Policy {
	return safefetch.DefaultPolicy().
		WithTimeout(timeout).
		WithResolver(outboundResolver).
		AllowingPrivate(outboundAllowedPrivatePrefixes...)
}
