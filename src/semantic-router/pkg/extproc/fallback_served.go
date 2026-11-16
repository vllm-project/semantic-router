package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// recordServedFallback makes Replay and protection follow the fallback
// candidate that answered in place of the routed model. The record names the
// candidate and the reasoning it was sent with, while the failed attempts stay
// in its fallback outcome. The protection key the dispatch took for the routed
// model passes to the candidate, so the next request is held on the model
// whose prompt cache it can read rather than sent to the failing one again.
func (r *OpenAIRouter) recordServedFallback(ctx *RequestContext, candidate *fallbackCandidate) {
	if ctx == nil || candidate == nil || candidate.dispatch == nil {
		return
	}
	if owner := ctx.committedSessionDecision; owner != nil && candidate.ref != nil {
		sessiontelemetry.RecordSessionServedModel(owner.SessionID, owner.SelectedModel, candidate.ref.Model, candidate.ref)
	}
	recorder := r.responseGuardReplayRecorder(ctx)
	if recorder == nil {
		return
	}
	served := routerreplay.ServedModel{
		Model:           candidate.model,
		ReasoningMode:   candidate.dispatch.reasoningMode,
		ReasoningEffort: candidate.dispatch.reasoningEffort,
	}
	if err := recorder.UpdateServedModel(ctx.RouterReplayID, served); err != nil {
		logging.ComponentErrorEvent("extproc", "router_replay_served_model_update_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"replay_id":  ctx.RouterReplayID,
			"error":      err.Error(),
		})
	}
}
