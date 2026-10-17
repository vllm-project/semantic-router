package extproc

// hasEnvelopeRoutingFacts reports whether a request carries anything worth
// routing on besides prompt text. A request with no text but with routing
// facts is not an empty request: decision evaluation still has to run, or
// rules written against those facts could never match.
//
// Accepted agentic facts count for the same reason untrusted request metadata
// does, and with a stronger claim: they passed the configured trust boundary
// and schema validation before reaching here.
func hasEnvelopeRoutingFacts(history signalConversationHistory, ctx *RequestContext) bool {
	return len(history.metadata) > 0 ||
		history.imageContentCount > 0 ||
		history.inputModality.AudioContentCount > 0 ||
		history.inputModality.VideoContentCount > 0 ||
		history.hasDeveloperMessage ||
		hasEnvelopeMessageFacts(history) ||
		hasEnvelopeToolFacts(history) ||
		history.lastMessageRole != "" ||
		(ctx != nil && ctx.AgenticFacts.HasFacts())
}

func hasEnvelopeMessageFacts(history signalConversationHistory) bool {
	return history.userMessageCount > 0 ||
		history.assistantMessageCount > 0 ||
		history.systemMessageCount > 0 ||
		history.toolMessageCount > 0
}

func hasEnvelopeToolFacts(history signalConversationHistory) bool {
	return history.toolDefinitionCount > 0 ||
		history.toolChoiceRequired ||
		history.toolChoiceNone ||
		history.assistantToolCallCount > 0 ||
		history.toolResultCount > 0
}
