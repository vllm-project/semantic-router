package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func recordSessionTurn(ctx *RequestContext, usage responseUsageMetrics) {
	if ctx == nil || usage.promptTokens+usage.completionTokens <= 0 {
		return
	}
	domain := consts.UnknownLabel
	if ctx.VSRSelectedCategory != "" {
		domain = ctx.VSRSelectedCategory
	}
	p := sessiontelemetry.TurnParams{
		RequestID:        ctx.RequestID,
		Model:            ctx.RequestModel,
		Domain:           domain,
		PromptTokens:     usage.promptTokens,
		CompletionTokens: usage.completionTokens,
	}
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest {
		if ctx.ResponseAPICtx.ConversationID == "" {
			return
		}
		p.ResponseAPI = &sessiontelemetry.ResponseAPIInput{
			ConversationID: ctx.ResponseAPICtx.ConversationID,
			HistoryLen:     len(ctx.ResponseAPICtx.ConversationHistory),
		}
	} else {
		userID := extractUserID(ctx)
		if userID == "" || len(ctx.ChatCompletionMessages) == 0 {
			return
		}
		msgs := make([]sessiontelemetry.ChatMessage, len(ctx.ChatCompletionMessages))
		for i := range ctx.ChatCompletionMessages {
			msgs[i] = sessiontelemetry.ChatMessage{
				Role:    ctx.ChatCompletionMessages[i].Role,
				Content: ctx.ChatCompletionMessages[i].Content,
			}
		}
		p.Chat = &sessiontelemetry.ChatInput{UserID: userID, Messages: msgs}
	}
	sessiontelemetry.RecordTurn(p)
}
