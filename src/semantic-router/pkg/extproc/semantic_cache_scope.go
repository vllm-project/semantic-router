package extproc

func (r *OpenAIRouter) semanticCacheEnabledForRequest(ctx *RequestContext) bool {
	if r == nil || r.Config == nil {
		return true
	}
	if requestBypassesRouting(ctx) {
		return false
	}
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		return r.Config.IsCacheEnabledForDecisionObject(ctx.VSRSelectedDecision)
	}
	if r.Config.HasRoutingDecisions() {
		return false
	}
	return r.Config.Enabled
}
