package extproc

import (
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

	if len(ctx.ChatCompletionMessages) > 0 {
		userID := extractUserID(ctx)
		if userID == "" {
			logging.Debugf("Session: no user ID, skipping session ID derivation")
		} else {
			ctx.SessionID = deriveSessionIDFromMessages(ctx.ChatCompletionMessages, userID)
		}
		ctx.TurnIndex = sessiontelemetry.ChatTurnNumber(sessionTransitionChatMessages(ctx.ChatCompletionMessages)) - 1
		// TODO: populate PreviousModel for Chat Completions once per-turn model history is available.
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
