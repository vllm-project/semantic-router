package extproc

import (
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

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
	toolResultCount        int
	assistantToolNames     []string
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
		toolResultCount:        result.ToolResultCount,
		assistantToolNames:     append([]string(nil), result.AssistantToolNames...),
	}
}

func extractToolTransitionContextFromRequest(req *openai.ChatCompletionNewParams, historyWindow int, ctx *RequestContext) tools.ToolTransitionContext {
	if req == nil {
		return toolTransitionContextFromConversationHistory(signalConversationHistory{}, historyWindow, ctx)
	}
	return toolTransitionContextFromConversationHistory(extractSignalConversationHistory(req), historyWindow, ctx)
}

func toolTransitionContextFromConversationHistory(history signalConversationHistory, historyWindow int, ctx *RequestContext) tools.ToolTransitionContext {
	return tools.ToolTransitionContext{
		RecentToolNames:  recentToolNames(history.assistantToolNames, historyWindow),
		UserMessageCount: history.userMessageCount,
		ToolResultCount:  history.toolResultCount,
		SelectedDecision: selectedDecisionName(ctx),
		SelectedCategory: selectedCategoryName(ctx),
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
			history.assistantToolNames = append(history.assistantToolNames, toolNamesFromAssistantMessage(msg)...)
		case "developer":
			history.hasDeveloperMessage = true
			if textContent != "" {
				history.nonUserMessages = append(history.nonUserMessages, textContent)
			}
		case "tool":
			history.toolMessageCount++
			history.toolResultCount++
		}
	}

	history.toolDefinitionCount = countSDKToolDefinitions(req)
	return history
}

func toolNamesFromAssistantMessage(msg openai.ChatCompletionMessageParamUnion) []string {
	if msg.OfAssistant == nil || len(msg.OfAssistant.ToolCalls) == 0 {
		return nil
	}
	names := make([]string, 0, len(msg.OfAssistant.ToolCalls))
	for _, toolCall := range msg.OfAssistant.ToolCalls {
		if toolCall.Function.Name != "" {
			names = append(names, toolCall.Function.Name)
		}
	}
	return names
}

func recentToolNames(names []string, historyWindow int) []string {
	if len(names) == 0 {
		return nil
	}
	if historyWindow <= 0 || historyWindow >= len(names) {
		return append([]string(nil), names...)
	}
	return append([]string(nil), names[len(names)-historyWindow:]...)
}

func selectedDecisionName(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	if ctx.VSRSelectedDecision != nil {
		return ctx.VSRSelectedDecision.Name
	}
	return ctx.VSRSelectedDecisionName
}

func selectedCategoryName(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	return ctx.VSRSelectedCategory
}

func countSDKToolDefinitions(req *openai.ChatCompletionNewParams) int {
	if req == nil {
		return 0
	}
	return len(req.Tools)
}
