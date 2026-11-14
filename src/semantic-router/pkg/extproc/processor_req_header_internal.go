package extproc

import (
	"slices"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

var droppedClientHeaders = headers.DroppedFromClients()

// markLooperHop makes a request a Looper hop only when the Router serves it
// in process, with its routing context typed. The Router's own headers a
// client request carries, its internal context among them, are never
// believed: they are dropped, so they influence neither routing nor plugins,
// and the request is an ordinary one.
func markLooperHop(ctx *RequestContext) {
	if ctx == nil {
		return
	}
	ctx.LooperRequest = ctx.Hop != nil
	for _, header := range droppedClientHeaders {
		removeHeaderValueCI(ctx, header)
	}
}

func removeHeaderValueCI(ctx *RequestContext, canonical string) {
	if ctx == nil || canonical == "" {
		return
	}
	for key := range ctx.Headers {
		if strings.EqualFold(key, canonical) {
			delete(ctx.Headers, key)
		}
	}
}

// providerBoundRemovals are the request headers removed before a request
// reaches a provider: the Router's own headers a client supplied, every
// header the registry keeps from providers, such as the Router's control
// headers, and the headers identity reads the caller from.
func providerBoundRemovals(identity config.IdentityConfig) []string {
	removals := slices.Clone(droppedClientHeaders)
	for _, name := range append(headers.KeptFromProviders(), identity.HeaderNames()...) {
		if !slices.Contains(removals, name) {
			removals = append(removals, name)
		}
	}
	return removals
}

func buildLooperInternalHeaderRemovalMutation(identity config.IdentityConfig) *ext_proc.HeaderMutation {
	return &ext_proc.HeaderMutation{
		RemoveHeaders: providerBoundRemovals(identity),
	}
}

// looperHopDecision names the decision whose plugin chain a hop runs.
func looperHopDecision(ctx *RequestContext) string {
	if ctx.Hop == nil {
		return ""
	}
	return ctx.Hop.Decision
}

// looperHopRecipe names the recipe a hop's decision belongs to.
func looperHopRecipe(ctx *RequestContext) string {
	if ctx.Hop == nil {
		return ""
	}
	return strings.TrimSpace(ctx.Hop.Recipe)
}
