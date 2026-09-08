package handlers

import (
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/safefetch"
)

// Hard ceilings on what a caller-supplied fetch may return. These bound the
// decoded body: safefetch disables compression so a small archive cannot
// expand past them.
const (
	openWebMaxResponseBytes  int64 = 8 * 1024 * 1024
	fetchRawMaxResponseBytes int64 = 2 * 1024 * 1024
)

// outboundResolver is swapped in tests to drive address classes and answer
// changes without real DNS. Nil means the system resolver.
var outboundResolver safefetch.IPResolver

// outboundPolicy is the single outbound-fetch policy for every handler that
// dials a destination the caller chose. Consumers adjust the deadline and the
// accepted schemes; the address check is not theirs to relax.
func outboundPolicy(timeout time.Duration) safefetch.Policy {
	return safefetch.DefaultPolicy().
		WithTimeout(timeout).
		WithResolver(outboundResolver)
}
