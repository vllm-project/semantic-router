package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routinginput"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/decisionmedia"
)

type signalConversationHistory struct {
	decisionMedia            []*decisionmedia.Source
	currentUserMessage       string
	priorUserMessages        []string
	nonUserMessages          []string
	toolResultTexts          []string
	toolResultScanIncomplete bool
	jailbreakInput           *classification.JailbreakInput
	routingInput             routinginput.Outcome
	hasAssistantReply        bool
	lastUserHasText          bool
	metadata                 map[string]string
	contextTokenFloor        int
	contextTextBytes         int
	contextEquivalentBytes   int
	contextHasNonText        bool
	contextReserveTokens     int

	// Conversation-shape facts for the conversation signal family.
	hasDeveloperMessage       bool
	userMessageCount          int
	assistantMessageCount     int
	systemMessageCount        int
	toolMessageCount          int
	toolDefinitionCount       int
	toolChoiceRequired        bool
	toolChoiceNone            bool
	assistantToolCallCount    int
	toolResultCount           int
	imageContentCount         int
	inputModality             classification.InputModalityFacts
	assistantToolNames        []string
	lastMessageRole           string
	lastMessageToolResult     bool
	lastMessageFlowToolResult bool
	lastAssistantToolCall     bool
	lastUserAfterToolResult   bool
}

func signalConversationHistoryFromSnapshot(result *requestSignalSnapshot) signalConversationHistory {
	if result == nil {
		return signalConversationHistory{}
	}
	return signalConversationHistory{
		decisionMedia:             result.DecisionMedia,
		currentUserMessage:        result.UserContent,
		priorUserMessages:         append([]string(nil), result.PriorUserMessages...),
		nonUserMessages:           append([]string(nil), result.NonUserMessages...),
		jailbreakInput:            result.JailbreakInput,
		routingInput:              result.RoutingInput,
		hasAssistantReply:         result.HasAssistantReply,
		lastUserHasText:           result.LastUserHasText,
		metadata:                  cloneRoutingMetadata(result.Metadata),
		contextTokenFloor:         result.ContextTokenFloor,
		contextTextBytes:          result.ContextTextBytes,
		contextEquivalentBytes:    result.ContextEquivalentBytes,
		contextHasNonText:         result.ContextHasNonText,
		contextReserveTokens:      result.ContextReserveTokens,
		hasDeveloperMessage:       result.HasDeveloperMessage,
		userMessageCount:          result.UserMessageCount,
		assistantMessageCount:     result.AssistantMessageCount,
		systemMessageCount:        result.SystemMessageCount,
		toolMessageCount:          result.ToolMessageCount,
		toolDefinitionCount:       result.ToolDefinitionCount,
		toolChoiceRequired:        result.ToolChoiceRequired,
		toolChoiceNone:            result.ToolChoiceNone,
		assistantToolCallCount:    result.AssistantToolCallCount,
		toolResultCount:           result.ToolResultCount,
		imageContentCount:         result.ImageContentCount,
		inputModality:             result.InputModality,
		assistantToolNames:        append([]string(nil), result.AssistantToolNames...),
		lastMessageRole:           result.LastMessageRole,
		lastMessageToolResult:     result.LastMessageToolResult,
		lastMessageFlowToolResult: result.LastMessageFlowToolResult,
		lastAssistantToolCall:     result.LastAssistantToolCall,
		lastUserAfterToolResult:   result.LastUserAfterToolResult,
	}
}

func signalConversationHistoryFromRequest(req *llmprotocol.Request, snapshot *requestSignalSnapshot) signalConversationHistory {
	history := signalConversationHistoryFromSnapshot(snapshot)
	if req != nil {
		extraction := extractToolResultTexts(req)
		history.toolResultTexts = extraction.texts
		history.toolResultScanIncomplete = extraction.incomplete
	}
	return history
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

func extractToolTransitionContextFromRequest(req *llmprotocol.Request, historyWindow int, ctx *RequestContext, selector *routinginput.Selector) tools.ToolTransitionContext {
	if req == nil {
		return toolTransitionContextFromConversationHistory(signalConversationHistory{}, historyWindow, ctx)
	}
	return toolTransitionContextFromConversationHistory(extractSignalConversationHistoryWith(req, selector), historyWindow, ctx)
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
	return extractSignalConversationHistoryWith(req, nil)
}

func extractSignalConversationHistoryWith(req *llmprotocol.Request, selector *routinginput.Selector) signalConversationHistory {
	if req == nil {
		return signalConversationHistory{}
	}
	return signalConversationHistoryFromRequest(req, extractSemanticRequestSignalsWith(req, selector))
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
