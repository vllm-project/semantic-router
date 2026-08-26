package extproc

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

// inferenceAdmissionRecovery captures only immutable usage-ledger identity.
// It deliberately cannot observe request headers, credentials, prompt or
// response bodies, tool arguments, or arbitrary metadata.
func (r *OpenAIRouter) inferenceAdmissionRecovery(
	request *RequestContext,
	state *inferenceRequestAccess,
	model string,
) (*quotaruntime.AdmissionRecoveryContext, error) {
	if request == nil || state == nil || r == nil || r.Config == nil {
		return nil, fmt.Errorf("admission recovery requires routed request state")
	}
	if state.tenant.PolicyRevision > math.MaxInt64 {
		return nil, fmt.Errorf("access policy revision exceeds recovery range")
	}
	model = strings.TrimSpace(model)
	params, exists := r.Config.ModelConfig[model]
	if !exists || strings.TrimSpace(params.ResourceID) == "" || params.ResourceRevision <= 0 {
		return nil, fmt.Errorf("admission recovery Model is not pinned")
	}
	occurredAt := request.StartTime.UTC()
	if occurredAt.IsZero() {
		occurredAt = request.ProcessingStartTime.UTC()
	}
	if occurredAt.IsZero() {
		occurredAt = time.Now().UTC()
	}
	routing := quotaruntime.RecoveryRouting{AccessRevision: int64(state.tenant.PolicyRevision)}
	if state.entrypoint != nil {
		entrypointID, err := durableResourceID("recovery entrypoint", state.entrypoint.ID)
		if err != nil {
			return nil, err
		}
		routing.EntrypointID = entrypointID
		routing.EntrypointName = state.entrypoint.Name
		routing.RoutingRevision = state.entrypoint.Revision
	}
	if state.rule != nil {
		ruleID, err := durableResourceID("recovery entrypoint rule", state.rule.ID)
		if err != nil {
			return nil, err
		}
		routing.EntrypointRuleID = ruleID
		routing.EntrypointRuleName = state.rule.Name
	}
	if recipe := request.Routing.SelectedRecipe(); recipe != nil {
		recipeID, err := durableResourceID("recovery recipe", recipe.ID)
		if err != nil {
			return nil, err
		}
		routing.RecipeID = recipeID
		routing.RecipeName = string(recipe.Name)
		routing.RecipeRevision = recipe.Revision
	}
	modelID, err := durableResourceID("recovery Model", params.ResourceID)
	if err != nil {
		return nil, err
	}
	return &quotaruntime.AdmissionRecoveryContext{
		EventID: uuid.NewString(), FenceID: uuid.NewString(),
		NamespaceID:       state.tenant.NamespaceID,
		ExternalRequestID: terminalExternalRequestID(request.RequestID),
		ReplayID:          request.RouterReplayID,
		Protocol:          inferenceUsageProtocol(request), Path: normalizedInferencePath(request),
		OccurredAt: occurredAt, Stream: request.ExpectStreamingResponse,
		Principal: quotaruntime.RecoveryPrincipal{
			APIKeyID: state.tenant.APIKeyID,
			UserID:   state.tenant.UserID,
			TeamID:   state.tenant.TeamID,
		},
		Routing: routing,
		FallbackDispatch: quotaruntime.RecoveryDispatch{
			ModelID: modelID, ModelName: model,
			ModelRevision: params.ResourceRevision, Currency: state.tenant.BillingCurrency,
		},
	}, nil
}
