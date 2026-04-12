package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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
		ctx.TurnIndex = len(ctx.ChatCompletionMessages) / 2
		// TODO: populate PreviousModel for Chat Completions once per-turn model history is available.
	}
}

// maybeEmitTransitionEvent records a ModelTransitionEvent on model change.
// Must be called after ctx.TTFTSeconds and ctx.CacheWarmthEstimate are set.
func maybeEmitTransitionEvent(ctx *RequestContext) {
	if ctx == nil || ctx.SessionID == "" || ctx.RequestModel == "" {
		return
	}
	if ctx.PreviousModel == "" || ctx.PreviousModel == ctx.RequestModel {
		return
	}

	previousResponseID := ""
	if ctx.ResponseAPICtx != nil {
		previousResponseID = ctx.ResponseAPICtx.PreviousResponseID
	}

	evt := latency.ModelTransitionEvent{
		SessionID:           ctx.SessionID,
		TurnIndex:           ctx.TurnIndex,
		FromModel:           ctx.PreviousModel,
		ToModel:             ctx.RequestModel,
		TTFTMs:              ctx.TTFTSeconds * 1000,
		CacheWarmthEstimate: ctx.CacheWarmthEstimate,
		PreviousResponseID:  previousResponseID,
		Timestamp:           time.Now(),
	}
	latency.RecordTransition(evt)
	logging.ComponentDebugEvent("session", "model_transition", map[string]interface{}{
		"session_id":            evt.SessionID,
		"turn_index":            evt.TurnIndex,
		"from_model":            evt.FromModel,
		"to_model":              evt.ToModel,
		"ttft_ms":               evt.TTFTMs,
		"cache_warmth_estimate": evt.CacheWarmthEstimate,
	})
}
