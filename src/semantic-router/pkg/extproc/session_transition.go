package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// populateSessionTransitionFields derives session-aware metadata from the
// request context. Must be called after ResponseAPICtx or ChatCompletionMessages
// are populated.
func populateSessionTransitionFields(ctx *RequestContext) {
	if ctx == nil {
		return
	}

	isResponseAPI := ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest
	if isResponseAPI {
		ctx.SessionID = ctx.ResponseAPICtx.ConversationID
		ctx.PreviousResponseID = ctx.ResponseAPICtx.PreviousResponseID
		history := ctx.ResponseAPICtx.ConversationHistory
		ctx.TurnIndex = len(history)
		if len(history) > 0 {
			ctx.PreviousModel = history[len(history)-1].Model
		}
		ctx.HistoryTokenCount = historyTokensFromStoredResponses(history)
		return
	}

	if sid := strings.TrimSpace(headerValueCI(ctx, headers.XSessionID)); sid != "" {
		ctx.SessionID = sid
	}

	if len(ctx.ChatCompletionMessages) == 0 {
		if ctx.SessionID == "" {
			ctx.SessionID = deriveSessionIDFromRequestID(ctx)
		}
		return
	}

	populateChatCompletionSessionIDIfNeeded(ctx)

	ctx.TurnIndex = sessiontelemetry.ChatTurnNumber(sessionTransitionChatMessages(ctx.ChatCompletionMessages)) - 1
	ctx.HistoryTokenCount = historyTokensFromChatMessages(ctx.ChatCompletionMessages)
	// TODO: populate PreviousModel for Chat Completions once per-turn model history is available.
}

// populateChatCompletionSessionIDIfNeeded sets ctx.SessionID when not already
// pinned (e.g. by x-session-id). Kept separate to avoid deep nesting in
// populateSessionTransitionFields (nestif).
func populateChatCompletionSessionIDIfNeeded(ctx *RequestContext) {
	if ctx.SessionID != "" {
		return
	}
	userID := extractUserID(ctx)
	if userID != "" {
		ctx.SessionID = deriveSessionIDFromMessages(ctx.ChatCompletionMessages, userID)
		return
	}
	ctx.SessionID = deriveSessionIDFromMessagesStructure(ctx.ChatCompletionMessages)
	if ctx.SessionID == "" {
		ctx.SessionID = deriveSessionIDFromRequestID(ctx)
	}
	if ctx.SessionID == "" {
		logging.Debugf("Session: could not derive session ID for chat completion request")
	}
}

func sessionTransitionChatMessages(messages []ChatCompletionMessage) []sessiontelemetry.ChatMessage {
	converted := make([]sessiontelemetry.ChatMessage, len(messages))
	for i := range messages {
		converted[i] = sessiontelemetry.ChatMessage{
			Role:    messages[i].Role,
			Content: messages[i].Content,
		}
	}
	return converted
}

func historyTokensFromStoredResponses(history []*responseapi.StoredResponse) int {
	total := 0
	for _, resp := range history {
		if resp == nil {
			continue
		}
		if resp.Usage != nil {
			total += resp.Usage.InputTokens + resp.Usage.OutputTokens
		} else {
			total += estimateStoredResponseTokens(resp)
		}
	}
	return total
}

const historyTokensCap = 8192

func estimateStoredResponseTokens(resp *responseapi.StoredResponse) int {
	total := 0
	for _, item := range resp.Input {
		total += estimateTokenLength(extractContentFromInputItem(item))
	}

	if resp.OutputText != "" {
		return total + estimateTokenLength(resp.OutputText)
	}

	for _, item := range resp.Output {
		total += estimateTokenLength(extractContentFromOutputItem(item))
	}

	return total
}

func historyTokensFromChatMessages(messages []ChatCompletionMessage) int {
	if len(messages) == 0 {
		return 0
	}
	prior := messages
	if messages[len(messages)-1].Role == "user" {
		prior = messages[:len(messages)-1]
	}
	total := 0
	for _, msg := range prior {
		total += estimateTokenLength(msg.Content)
		if total >= historyTokensCap {
			return historyTokensCap
		}
	}
	return total
}

func estimateTokenLength(text string) int {
	return len(text) / 4
}
