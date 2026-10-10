package topiccontinuity

import (
	"context"
	"time"
)

// cancellationProbe reports whether evaluation must stop. It compares the
// deadline with the clock itself instead of relying on ctx.Err alone: a
// context's deadline error is set by a timer callback that needs a free
// processor to run, so a CPU-bound evaluation on one CPU could otherwise
// finish long after its deadline without ever observing it.
//
// Probes run at message and turn boundaries during preparation and between
// bounded extraction stages. Work between probes is limited to one turn's
// retained text, at most MaxTurnBytes, and the fixed structural caps.
type cancellationProbe struct {
	ctx         context.Context
	deadline    time.Time
	hasDeadline bool
}

func newCancellationProbe(ctx context.Context) cancellationProbe {
	deadline, ok := ctx.Deadline()
	return cancellationProbe{ctx: ctx, deadline: deadline, hasDeadline: ok}
}

func (p cancellationProbe) cancelled() bool {
	if p.ctx.Err() != nil {
		return true
	}
	return p.hasDeadline && !time.Now().Before(p.deadline)
}
