package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	replaySessionActionNone     = "none"
	replaySessionActionSelect   = "select"
	replaySessionActionStay     = "stay"
	replaySessionActionSwitch   = "switch"
	replaySessionActionHardLock = "hard_lock"
)

func buildReplayRouteDiagnostics(
	ctx *RequestContext,
	originalModel string,
	selectedModel string,
	decisionName string,
	decisionTier int,
	decisionPriority int,
) *routerreplay.RouteDiagnostics {
	finalModel := replaySelectedModel(originalModel, selectedModel)
	diagnostics := &routerreplay.RouteDiagnostics{
		Decision:         decisionName,
		DecisionTier:     decisionTier,
		DecisionPriority: decisionPriority,
		SelectionMethod:  ctx.VSRSelectionMethod,
		OriginalModel:    originalModel,
		ProposalModel:    finalModel,
		SelectedModel:    finalModel,
		SessionAction:    replaySessionActionNone,
	}

	if policy, ok := sessionAwareLearningPolicyForContext(ctx); ok {
		diagnostics.SessionPolicyApplied = true
		diagnostics.SessionPhase = policy.SessionPhase()
		diagnostics.PreviousModel = policy.CurrentModel()
		diagnostics.ProposalModel = firstNonEmpty(policy.BaseSelectedModel(), diagnostics.ProposalModel)
		diagnostics.SelectedModel = firstNonEmpty(policy.SelectedModel(), diagnostics.SelectedModel)
		diagnostics.HardLockReason = policy.HardLockReason()
		diagnostics.DecisionReason = policy.DecisionReason()
		diagnostics.SessionAction = replaySessionAction(diagnostics, policy.HardLocked())
		diagnostics.SessionReason = replaySessionReason(diagnostics)
		return diagnostics
	}

	policy := ctx.VSRSessionPolicy
	if len(policy) == 0 {
		return diagnostics
	}

	diagnostics.SessionPolicyApplied = true
	diagnostics.SessionPhase = replayPolicyString(policy, string(learningPolicyFieldPhase))
	diagnostics.PreviousModel = replayPolicyString(policy, string(learningPolicyFieldCurrentModel))
	diagnostics.ProposalModel = firstNonEmpty(replayPolicyString(policy, string(learningPolicyFieldBaseSelectedModel)), diagnostics.ProposalModel)
	diagnostics.SelectedModel = firstNonEmpty(replayPolicyString(policy, string(learningPolicyFieldSelectedModel)), diagnostics.SelectedModel)
	diagnostics.HardLockReason = replayPolicyString(policy, string(learningPolicyFieldHardLockReason))
	diagnostics.DecisionReason = replayPolicyString(policy, string(learningPolicyFieldDecisionReason))
	diagnostics.SessionAction = replaySessionAction(diagnostics, replayPolicyBool(policy, string(learningPolicyFieldHardLocked)))
	diagnostics.SessionReason = replaySessionReason(diagnostics)
	return diagnostics
}

func buildReplayLearningDiagnostics(ctx *RequestContext) *routerreplay.LearningDiagnostics {
	policies := learningPoliciesForReplay(ctx)
	if len(policies) == 0 {
		return nil
	}
	adaptations := make(map[string]map[string]interface{}, len(policies))
	for method, policy := range policies {
		adaptations[string(method)] = policy.ToMap()
	}
	return &routerreplay.LearningDiagnostics{
		Adaptations: adaptations,
	}
}

func learningPoliciesForReplay(ctx *RequestContext) map[routerLearningMethod]routerLearningPolicy {
	if ctx == nil {
		return nil
	}
	if len(ctx.VSRLearningPolicies) > 0 {
		return ctx.VSRLearningPolicies
	}
	if ctx.VSRLearningPolicy == nil || ctx.VSRLearningPolicy.Empty() {
		return nil
	}
	adaptation := ctx.VSRLearningPolicy.Adaptation
	if adaptation == "" {
		adaptation = routerLearningMethodSessionAware
	}
	return map[routerLearningMethod]routerLearningPolicy{
		adaptation: *ctx.VSRLearningPolicy,
	}
}

func replaySessionAction(diagnostics *routerreplay.RouteDiagnostics, hardLocked bool) string {
	if hardLocked {
		return replaySessionActionHardLock
	}
	if diagnostics.PreviousModel == "" {
		return replaySessionActionSelect
	}
	if diagnostics.SelectedModel == diagnostics.PreviousModel {
		return replaySessionActionStay
	}
	return replaySessionActionSwitch
}

func replaySessionReason(diagnostics *routerreplay.RouteDiagnostics) string {
	if diagnostics.SessionAction == replaySessionActionHardLock && diagnostics.HardLockReason != "" {
		return diagnostics.HardLockReason
	}
	if diagnostics.DecisionReason != "" {
		return diagnostics.DecisionReason
	}
	return diagnostics.SessionAction
}

func replayPolicyString(policy map[string]interface{}, key string) string {
	value, ok := policy[key]
	if !ok {
		return ""
	}
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	default:
		return ""
	}
}

func replayPolicyBool(policy map[string]interface{}, key string) bool {
	value, ok := policy[key]
	if !ok {
		return false
	}
	typed, ok := value.(bool)
	return ok && typed
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
