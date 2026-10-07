package routing

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

// Hop marks a routing session as one model call of a request graph: a call
// the Router makes itself while it serves a client request. The session runs
// the hop's plugin chain in process, as the named decision configures it,
// instead of routing the request afresh. Only code in the Router's process
// can mark a session; no request header can, so a client cannot pose as a
// hop.
type Hop struct {
	// Decision names the decision whose plugin chain the hop runs. Empty
	// runs none: the hop goes through the provider dispatch only.
	Decision string
	// Recipe scopes Decision; empty means the default recipe.
	Recipe string
	// Iteration numbers the hops of one client request from 1.
	Iteration int
	// Fallback overrides the decision's cross-model fallback for this hop,
	// field by field: the request-graph step's layer. Nil keeps the
	// decision's.
	Fallback *fallback.FallbackOverride
}

type hopKey struct{}

// WithHop returns ctx marked with hop; a session opened with it serves hop.
func WithHop(ctx context.Context, hop Hop) context.Context {
	return context.WithValue(ctx, hopKey{}, hop)
}

// HopFrom returns the hop ctx is marked with.
func HopFrom(ctx context.Context) (Hop, bool) {
	if ctx == nil {
		return Hop{}, false
	}
	hop, ok := ctx.Value(hopKey{}).(Hop)
	return hop, ok
}
