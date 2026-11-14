package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routinginput"
)

type signalConversationHistory struct {
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
	inputModality             classification.InputModalityFacts
	lastAssistantToolNames    []string
	lastAssistantToolArgs     []string
	turns                     []classification.ConversationTurn
	images                    []string
	userVideos                []string
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
		inputModality:             result.InputModality,
		lastAssistantToolNames:    append([]string(nil), result.LastAssistantToolNames...),
		lastAssistantToolArgs:     append([]string(nil), result.LastAssistantToolArgs...),
		turns:                     result.PriorTurns(),
		images:                    append([]string(nil), result.Images...),
		userVideos:                append([]string(nil), result.UserVideos...),
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
