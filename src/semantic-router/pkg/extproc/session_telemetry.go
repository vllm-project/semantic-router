package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// sessionTurnPricing looks up the active pricing for model from the router config
// and converts it to the sessiontelemetry value type.
func (r *OpenAIRouter) sessionTurnPricing(model string) sessiontelemetry.TurnPricing {
	if r.Config == nil {
		return sessiontelemetry.TurnPricing{}
	}
	p, ok := r.Config.GetFullModelPricing(model)
	if !ok {
		return sessiontelemetry.TurnPricing{}
	}
	return sessiontelemetry.TurnPricing{
		Currency:         p.Currency,
		PromptPer1M:      p.PromptPer1M,
		CompletionPer1M:  p.CompletionPer1M,
		CachedInputPer1M: p.CachedInputPer1M,
	}
}

func recordSessionTurn(ctx *RequestContext, usage responseUsageMetrics, pricing sessiontelemetry.TurnPricing) {
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
		Pricing:          pricing,
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
