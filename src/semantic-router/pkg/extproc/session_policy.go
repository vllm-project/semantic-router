package extproc

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func recordAgenticSessionDecision(
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
	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:      selCtx.SessionID,
		UserID:         selCtx.UserID,
		PreviousModel:  previousModel,
		SelectedModel:  selectedModelRef.Model,
		DecisionName:   selCtx.DecisionName,
		TurnIndex:      ctx.TurnIndex,
		ActiveToolLoop: activeToolLoop,
		Policy:         policy,
		Timestamp:      time.Now(),
	})
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
