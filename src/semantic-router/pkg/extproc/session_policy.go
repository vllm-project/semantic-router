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
	policy := ctx.VSRSessionPolicy
	if policy == nil && result != nil && result.SessionPolicy != nil {
		policy = result.SessionPolicy.ToMap()
		ctx.VSRSessionPolicy = policy
	}
	activeToolLoop := false
	previousModel := ctx.PreviousModel
	if selCtx.AgenticSession != nil {
		activeToolLoop = selCtx.AgenticSession.ActiveToolLoop
		if previousModel == "" {
			previousModel = selCtx.AgenticSession.PreviousModel
		}
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
