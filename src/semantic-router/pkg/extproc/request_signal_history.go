package extproc

import "github.com/openai/openai-go"

type signalConversationHistory struct {
	currentUserMessage string
	priorUserMessages  []string
	nonUserMessages    []string
	hasAssistantReply  bool

	// Conversation-shape facts for the conversation signal family.
	hasDeveloperMessage    bool
	userMessageCount       int
	assistantMessageCount  int
	systemMessageCount     int
	toolMessageCount       int
	toolDefinitionCount    int
	assistantToolCallCount int
	completedToolCycles    int
}

func signalConversationHistoryFromFastExtract(result *FastExtractResult) signalConversationHistory {
	if result == nil {
		return signalConversationHistory{}
	}
	return signalConversationHistory{
		currentUserMessage:     result.UserContent,
		priorUserMessages:      append([]string(nil), result.PriorUserMessages...),
		nonUserMessages:        append([]string(nil), result.NonUserMessages...),
		hasAssistantReply:      result.HasAssistantReply,
		hasDeveloperMessage:    result.HasDeveloperMessage,
		userMessageCount:       result.UserMessageCount,
		assistantMessageCount:  result.AssistantMessageCount,
		systemMessageCount:     result.SystemMessageCount,
		toolMessageCount:       result.ToolMessageCount,
		toolDefinitionCount:    result.ToolDefinitionCount,
		assistantToolCallCount: result.AssistantToolCallCount,
		completedToolCycles:    result.CompletedToolCycles,
	}
}

func extractSignalConversationHistory(req *openai.ChatCompletionNewParams) signalConversationHistory {
	var history signalConversationHistory

	for _, msg := range req.Messages {
		role, textContent := extractMessageRoleAndContent(msg)

		switch role {
		case "user":
			history.userMessageCount++
			if textContent == "" {
				continue
			}
			if history.currentUserMessage != "" {
				history.priorUserMessages = append(history.priorUserMessages, history.currentUserMessage)
			}
			history.currentUserMessage = textContent
		case "system":
			history.systemMessageCount++
			if textContent != "" {
				history.nonUserMessages = append(history.nonUserMessages, textContent)
			}
		case "assistant":
			history.assistantMessageCount++
			if textContent != "" {
				history.nonUserMessages = append(history.nonUserMessages, textContent)
			}
			history.hasAssistantReply = true
			history.assistantToolCallCount += len(msg.OfAssistant.ToolCalls)
		case "developer":
			history.hasDeveloperMessage = true
			if textContent != "" {
				history.nonUserMessages = append(history.nonUserMessages, textContent)
			}
		case "tool":
			history.toolMessageCount++
			history.completedToolCycles++
		}
	}

	history.toolDefinitionCount = countSDKToolDefinitions(req)
	return history
}

func countSDKToolDefinitions(req *openai.ChatCompletionNewParams) int {
	if req == nil {
		return 0
	}
	return len(req.Tools)
}
