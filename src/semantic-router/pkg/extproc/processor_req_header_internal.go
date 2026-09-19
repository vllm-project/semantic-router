package extproc

import (
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

var looperInternalContextHeaders = []string{
	headers.VSRInternalAuth,
	headers.VSRLooperRequest,
	headers.VSRLooperIteration,
	headers.VSRLooperDecision,
	headers.VSRFusionDepth,
	headers.VSRSelectedRecipe,
}

// markLooperHop makes a request a Looper hop only when the Router serves it
// in process, with its routing context typed. Internal context headers a
// request carries are never believed: they are dropped, so they influence
// neither routing nor plugins, and the request is an ordinary one.
func markLooperHop(ctx *RequestContext) {
	if ctx == nil {
		return
	}
	ctx.LooperRequest = ctx.Hop != nil
	for _, header := range looperInternalContextHeaders {
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

func looperInternalHeadersForRemoval() []string {
	return append([]string(nil), looperInternalContextHeaders...)
}

func buildLooperInternalHeaderRemovalMutation(extraRemovals ...string) *ext_proc.HeaderMutation {
	return &ext_proc.HeaderMutation{
		RemoveHeaders: append(looperInternalHeadersForRemoval(), extraRemovals...),
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
