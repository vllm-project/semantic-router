package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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
		history := ctx.ResponseAPICtx.ConversationHistory
		ctx.TurnIndex = len(history)
		if len(history) > 0 {
			ctx.PreviousModel = history[len(history)-1].Model
		}
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

	if ctx.SessionID == "" {
		userID := extractUserID(ctx)
		if userID != "" {
			ctx.SessionID = deriveSessionIDFromMessages(ctx.ChatCompletionMessages, userID)
		} else {
			ctx.SessionID = deriveSessionIDFromMessagesStructure(ctx.ChatCompletionMessages)
			if ctx.SessionID == "" {
				ctx.SessionID = deriveSessionIDFromRequestID(ctx)
			}
			if ctx.SessionID == "" {
				logging.Debugf("Session: could not derive session ID for chat completion request")
			}
		}
	}

	ctx.TurnIndex = sessiontelemetry.ChatTurnNumber(sessionTransitionChatMessages(ctx.ChatCompletionMessages)) - 1
	// TODO: populate PreviousModel for Chat Completions once per-turn model history is available.
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
