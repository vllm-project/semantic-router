package extproc

func hasEnvelopeRoutingFacts(history signalConversationHistory) bool {
	return len(history.metadata) > 0 ||
		history.imageContentCount > 0 ||
		history.inputModality.AudioContentCount > 0 ||
		history.inputModality.VideoContentCount > 0 ||
		history.hasDeveloperMessage ||
		history.userMessageCount > 0 ||
		history.assistantMessageCount > 0 ||
		history.systemMessageCount > 0 ||
		history.toolMessageCount > 0 ||
		history.toolDefinitionCount > 0 ||
		history.assistantToolCallCount > 0 ||
		history.toolResultCount > 0 ||
		history.lastMessageRole != ""
}
