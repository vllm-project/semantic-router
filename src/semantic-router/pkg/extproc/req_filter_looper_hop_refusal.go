package extproc

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

// looperHopRefusalsKey marks a Looper run's context with the refusals of its
// hops. Each hop's session derives its context from the run's, so its
// dispatch finds them there.
type looperHopRefusalsKey struct{}

// looperHopRefusals are the hops of a Looper run whose dispatch refused the
// request as the client sent it, for a stream of several choices, before any
// backend call. It keeps the first refusal, which the run answers with when
// it refused every hop the run sent: no candidate can serve the request, and
// the client has to change it.
type looperHopRefusals struct {
	mu      sync.Mutex
	hops    map[*RequestContext]struct{}
	refusal *llmprotocol.ProtocolError
}

func withLooperHopRefusals(ctx context.Context) (context.Context, *looperHopRefusals) {
	refusals := &looperHopRefusals{}
	return context.WithValue(ctx, looperHopRefusalsKey{}, refusals), refusals
}

// noteLooperHopRefusal records err, the refusal of a hop's dispatch, for the
// run the hop serves. It does nothing for a request that is not a hop.
func noteLooperHopRefusal(ctx *RequestContext, err error) {
	if ctx == nil || !ctx.LooperRequest || ctx.TraceContext == nil {
		return
	}
	refusals, ok := ctx.TraceContext.Value(looperHopRefusalsKey{}).(*looperHopRefusals)
	var protocolError *llmprotocol.ProtocolError
	if !ok || !errors.As(err, &protocolError) {
		return
	}
	refusals.mu.Lock()
	defer refusals.mu.Unlock()
	if refusals.hops == nil {
		refusals.hops = map[*RequestContext]struct{}{}
	}
	refusals.hops[ctx] = struct{}{}
	if refusals.refusal == nil {
		refusals.refusal = protocolError
	}
}

// ofEveryHop is the refusal when it refused each hop of a run that sent
// any, and nil when a hop got past it.
func (s *looperHopRefusals) ofEveryHop(outcome *graph.Outcome) *llmprotocol.ProtocolError {
	if s == nil || outcome == nil || outcome.Hops == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.hops) != outcome.Hops {
		return nil
	}
	return s.refusal
}

// looperHopsRefusedError fails a Looper run whose every hop its dispatch
// refused: the client gets that refusal, not a server error it would retry.
type looperHopsRefusedError struct {
	run     error
	refusal *llmprotocol.ProtocolError
}

func (e *looperHopsRefusedError) Error() string {
	return fmt.Sprintf("%v: every hop was refused: %v", e.run, e.refusal)
}

func (e *looperHopsRefusedError) Unwrap() error { return e.run }

// explainLooperRunError marks the error of a failed run whose every hop was
// refused, and returns any other error unchanged.
func explainLooperRunError(err error, refusals *looperHopRefusals, outcome *graph.Outcome) error {
	if refusal := refusals.ofEveryHop(outcome); refusal != nil {
		return &looperHopsRefusedError{run: err, refusal: refusal}
	}
	return err
}
