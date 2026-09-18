package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// stageAgenticSessionDecision snapshots the selection without publishing a new
// owner. Plugins, credentials and final encoding can still reject this request.
func stageAgenticSessionDecision(
	selCtx *selection.SelectionContext,
	result *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
	ctx *RequestContext,
) {
	if selCtx == nil || selectedModelRef == nil || ctx == nil || selCtx.SessionID == "" {
		return
	}
	policy := sessionPolicyMapForTelemetry(ctx, result)
	activeToolLoop := false
	previousModel := ctx.PreviousModel
	if selCtx.AgenticSession != nil {
		activeToolLoop = selCtx.AgenticSession.ActiveToolLoop
		previousModel = selCtx.AgenticSession.PreviousModel
	}
	ctx.pendingSessionDecision = &sessiontelemetry.SessionDecisionParams{
		SessionID:         selectionSessionStateKey(selCtx),
		UserID:            selCtx.UserID,
		PreviousModel:     previousModel,
		SelectedModel:     selectedModelRef.Model,
		SelectedCandidate: (&selection.SelectionResult{}).WithCandidate(*selectedModelRef).SelectedCandidate,
		DecisionName:      selectionDecisionStateKey(selCtx),
		TurnIndex:         ctx.TurnIndex,
		ActiveToolLoop:    activeToolLoop,
		Policy:            policy,
	}
}

// commitAgenticSessionDecision runs after final provider encoding, never on an
// immediate response. No rollback is needed, so a failed request cannot undo a
// newer dispatch's ownership. Clearing the staged value makes commitment once-only.
func commitAgenticSessionDecision(ctx *RequestContext) error {
	if err := selectionRequestContext(ctx).Err(); err != nil {
		return err
	}
	if ctx != nil && ctx.pendingSessionDecision != nil {
		ctx.pendingSessionDecision.Timestamp = time.Now()
		sessiontelemetry.RecordSessionDecision(*ctx.pendingSessionDecision)
		ctx.pendingSessionDecision = nil
	}
	return nil
}

func sessionPolicyMapForTelemetry(
	ctx *RequestContext,
	result *selection.SelectionResult,
) map[string]interface{} {
	if policy, ok := protectionLearningPolicyForContext(ctx); ok {
		return policy.ToMap()
	}
	if result != nil && result.SessionPolicy != nil {
		return result.SessionPolicy.ToMap()
	}
	return nil
}
