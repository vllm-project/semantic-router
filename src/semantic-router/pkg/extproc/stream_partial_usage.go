package extproc

// recordInterruptedStreamUsage keeps the usage a stream reported before it was
// cut short, by its client or its backend, priced and marked partial, so the
// call is not left unpriced:
// Anthropic counts the prompt in message_start and the output only as far as
// the stream had counted it. A stream that reported none, as an OpenAI stream
// does before its last chunk, keeps none. It reports whether it kept usage.
func (r *OpenAIRouter) recordInterruptedStreamUsage(ctx *RequestContext) bool {
	state := ctx.SemanticStreamState
	if state == nil || (state.terminal && state.failed == nil) {
		return false
	}
	usage := responseUsageFromSemanticUsage(state.usage)
	if usage.invalid {
		return false
	}
	replayUsage := r.buildReplayUsageCost(ctx, usage)
	replayUsage.CostAccounting.PartialUsage = true
	if replayUsage.ActualCost != nil && replayUsage.Currency != nil {
		// The spend budget settles the reservation at this price.
		ctx.RequestCost, ctx.RequestCostCurrency, ctx.RequestCostPriced = *replayUsage.ActualCost, *replayUsage.Currency, true
	}
	r.writeRouterReplayUsageCost(ctx, replayUsage)
	return true
}
