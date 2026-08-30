package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

type signalConversationHistory struct {
	currentUserMessage     string
	priorUserMessages      []string
	nonUserMessages        []string
	hasAssistantReply      bool
	metadata               map[string]string
	contextTokenFloor      int
	contextTextBytes       int
	contextEquivalentBytes int
	contextHasNonText      bool

	// Conversation-shape facts for the conversation signal family.
	hasDeveloperMessage     bool
	userMessageCount        int
	assistantMessageCount   int
	systemMessageCount      int
	toolMessageCount        int
	toolDefinitionCount     int
	assistantToolCallCount  int
	toolResultCount         int
	imageContentCount       int
	assistantToolNames      []string
	lastMessageRole         string
	lastMessageToolResult   bool
	lastUserAfterToolResult bool
}

func signalConversationHistoryFromSnapshot(result *requestSignalSnapshot) signalConversationHistory {
	if result == nil {
		return signalConversationHistory{}
	}
	return signalConversationHistory{
		currentUserMessage:      result.UserContent,
		priorUserMessages:       append([]string(nil), result.PriorUserMessages...),
		nonUserMessages:         append([]string(nil), result.NonUserMessages...),
		hasAssistantReply:       result.HasAssistantReply,
		metadata:                cloneRoutingMetadata(result.Metadata),
		contextTokenFloor:       result.ContextTokenFloor,
		contextTextBytes:        result.ContextTextBytes,
		contextEquivalentBytes:  result.ContextEquivalentBytes,
		contextHasNonText:       result.ContextHasNonText,
		hasDeveloperMessage:     result.HasDeveloperMessage,
		userMessageCount:        result.UserMessageCount,
		assistantMessageCount:   result.AssistantMessageCount,
		systemMessageCount:      result.SystemMessageCount,
		toolMessageCount:        result.ToolMessageCount,
		toolDefinitionCount:     result.ToolDefinitionCount,
		assistantToolCallCount:  result.AssistantToolCallCount,
		toolResultCount:         result.ToolResultCount,
		imageContentCount:       result.ImageContentCount,
		assistantToolNames:      append([]string(nil), result.AssistantToolNames...),
		lastMessageRole:         result.LastMessageRole,
		lastMessageToolResult:   result.LastMessageToolResult,
		lastUserAfterToolResult: result.LastUserAfterToolResult,
	}
}

func cloneRoutingMetadata(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(values))
	for key, value := range values {
		cloned[key] = value
	}
	return cloned
}

func extractToolTransitionContextFromRequest(req *llmprotocol.Request, historyWindow int, ctx *RequestContext) tools.ToolTransitionContext {
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

func extractSignalConversationHistory(req *llmprotocol.Request) signalConversationHistory {
	if req == nil {
		return signalConversationHistory{}
	}
	return signalConversationHistoryFromSnapshot(extractSemanticRequestSignals(req))
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
