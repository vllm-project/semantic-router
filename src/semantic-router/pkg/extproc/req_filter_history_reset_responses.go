package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Responses requests carry part of their conversation behind
// previous_response_id. Dispatch normally materializes that stored history
// just before the provider call, which is after the context stage has run: a
// reset would then remove nothing and dispatch would restore the very turns
// the policy was meant to drop.
//
// For a reset-enabled decision the router resolves the permitted history
// first, so the original-history snapshot, the topic evidence bound to it, and
// the transformation view all describe the conversation the provider would
// actually receive. Materialization stays idempotent, so the later dispatch
// call becomes a no-op instead of prepending the history a second time.
func (r *OpenAIRouter) resolveHistoryResetRequestHistory(ctx *RequestContext) {
	if ctx == nil || !ctx.HistoryResetPolicy.IsEnabled() || ctx.SemanticRequest == nil {
		return
	}
	if ctx.ResponseObjectState == nil || ctx.ResponseObjectState.ProviderContextApplied {
		return
	}
	changed, err := r.materializeResponseObjectContext(ctx.SemanticRequest, ctx)
	if err != nil {
		// The permitted history could not be resolved, so no view of it can be
		// trusted. Block the action and let the configured failure mode decide
		// between preserving the request and rejecting it.
		ctx.HistoryResetBlocked = historyreset.ReasonHistoryUnresolved
		logging.ComponentWarnEvent("extproc", "history_reset_history_unresolved", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      err.Error(),
		})
		return
	}
	if !changed {
		return
	}
	// The snapshot taken at ingress predates the stored history. Retake it now,
	// before RAG and Memory enrichment, so it still describes original history
	// only. Signal extraction and the original request-demand estimate keep
	// their existing ingress inputs.
	ctx.OriginalContextHistory = nil
	captureOriginalContextHistory(ctx)
}
