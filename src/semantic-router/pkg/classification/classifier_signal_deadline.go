package classification

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// withSignalDeadline gives a request's signals a deadline below the request's,
// so a slow model signal resolves through its own policy (on_error for a
// routing signal, on_unscanned for a safety one) instead of failing the whole
// request. It is the configured timeout, and at most the request's deadline
// less a tenth of the time left (at least a second) to answer; a request
// without a deadline takes the timeout or config.DefaultSignalTimeout.
func withSignalDeadline(ctx context.Context, timeout time.Duration, now time.Time) (context.Context, context.CancelFunc) {
	deadline := now.Add(config.DefaultSignalTimeout)
	if timeout > 0 {
		deadline = now.Add(timeout)
	}
	if request, ok := ctx.Deadline(); ok {
		derived := request.Add(-max(time.Second, request.Sub(now)/10))
		if timeout <= 0 || derived.Before(deadline) {
			deadline = derived
		}
		if !deadline.After(now) {
			deadline = request
		}
	}
	return context.WithDeadline(ctx, deadline)
}
