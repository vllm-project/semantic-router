package gateway

import (
	"context"
	"time"
)

// drainPoll is how often Drain looks whether the requests in flight finished.
const drainPoll = 25 * time.Millisecond

// InFlight is how many requests the handler is serving.
func (h *Handler) InFlight() int {
	return int(h.inFlight.Load())
}

// Drain waits until the handler serves no request, or ctx ends, and returns
// how many requests were still in flight then. It follows the server's
// Shutdown, which stops new requests but does not wait for h2c streams: the
// h2c handler hijacks their connections from the server.
func (h *Handler) Drain(ctx context.Context) int {
	ticker := time.NewTicker(drainPoll)
	defer ticker.Stop()
	for h.inFlight.Load() > 0 {
		select {
		case <-ctx.Done():
			return h.InFlight()
		case <-ticker.C:
		}
	}
	return 0
}
