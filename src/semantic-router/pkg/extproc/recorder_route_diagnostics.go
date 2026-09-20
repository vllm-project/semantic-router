package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	replaySessionActionNone              = "none"
	replaySessionActionEstablishBaseline = "establish_baseline"
	replaySessionActionStay              = "stay"
	replaySessionActionSwitch            = "switch"
	replaySessionActionHardLock          = "hard_lock"
)

func buildReplayRouteDiagnostics(
	ctx *RequestContext,
	originalModel string,
	selectedModel string,
	decisionName string,
	decisionTier int,
	decisionPriority int,
) *routerreplay.RouteDiagnostics {
	finalModel := replaySelectedModel(selectedModel)
	diagnostics := &routerreplay.RouteDiagnostics{
		Decision:                       decisionName,
		DecisionTier:                   decisionTier,
		DecisionPriority:               decisionPriority,
		SelectionMethod:                ctx.VSRSelectionMethod,
		SelectionReasoning:             ctx.VSRSelectionReasoning,
		SelectionTrace:                 ctx.VSRSelectionTrace.Clone(),
		FusionQuorum:                   ctx.VSRFusionQuorum,
		Looper:                         ctx.VSRLooperDiagnostics,
		PromptHelperModel:              ctx.VSRPromptHelperModel,
		PromptHelperPromptTokens:       ctx.VSRPromptHelperPromptTokens,
		PromptHelperCompletionTokens:   ctx.VSRPromptHelperCompletionTokens,
		PromptHelperTotalTokens:        ctx.VSRPromptHelperTotalTokens,
		PromptHelperLatencyMs:          ctx.VSRPromptHelperLatencyMs,
		OriginalModel:                  originalModel,
		ProposalModel:                  finalModel,
		SelectedModel:                  finalModel,
		SessionAction:                  replaySessionActionNone,
		MemoryBackend:                  ctx.MemoryBackend,
		MemoryStatus:                   ctx.MemoryStatus,
		MemoryReason:                   ctx.MemoryReason,
		MemoryFallbackReason:           ctx.MemoryFallbackReason,
		MemoryFailOpen:                 ctx.MemoryFailOpen,
		MemoryResultCount:              ctx.MemoryResultCount,
		ContextCompressionApplied:      ctx.ContextCompressionApplied,
		ContextCompressionBefore:       ctx.ContextCompressionBefore,
		ContextCompressionAfter:        ctx.ContextCompressionAfter,
		ContextCompressionMessages:     ctx.ContextCompressionMessages,
		ContextCompressionFormat:       ctx.ContextCompressionFormat,
		ContextCompressionOmitted:      ctx.ContextCompressionOmitted,
		ContextCompressionSkipReason:   ctx.ContextCompressionSkipReason,
		ContextCompressionStrategy:     ctx.ContextCompressionStrategy,
		ContextCompressionBudgetMode:   ctx.ContextCompressionBudgetMode,
		ContextCompressionTokenSource:  ctx.ContextCompressionTokenSource,
		ContextCompressionTrigger:      ctx.ContextCompressionTrigger,
		ContextCompressionRevision:     ctx.ContextCompressionRevision,
		ContextCompressionRecoveryKeys: len(ctx.ContextCompressionRecoveryKeys),
		ContextCompressionQuality:      ctx.ContextCompressionQuality,
		ContextCompressionFallback:     ctx.ContextCompressionFallback,
		ContextCompressionCostSaved:    ctx.ContextCompressionCostSaved,
		RequestDemandSnapshots:         cloneRequestDemandSnapshots(ctx.RequestDemandSnapshots),
		SignalErrors:                   cloneReplayStringMap(ctx.VSRSignalErrors),
		AppliedUnknownPolicies:         ctx.VSRDecisionDiagnostics.AppliedUnknownPolicies,
	}
	diagnostics.ContextDedup = contextDedupReplayDiagnostics(ctx)
	if ctx.VSRSelectedDecision != nil {
		diagnostics.Annotations = ctx.VSRSelectedDecision.Annotations
	}

	if policy, ok := protectionLearningPolicyForContext(ctx); ok {
		diagnostics.SessionPolicyApplied = policy.Mode == config.DecisionAdaptationModeApply &&
			policy.Details.ProtectionTrace() != nil
		diagnostics.SessionPhase = policy.SessionPhase()
		diagnostics.PreviousModel = policy.CurrentModel()
		diagnostics.ProposalModel = firstNonEmpty(policy.BaseSelectedModel(), diagnostics.ProposalModel)
		diagnostics.DecisionReason = policy.DecisionReason()
		// The dispatch result is authoritative. Observe-mode traces describe a
		// counterfactual selection and must not turn a real switch into a hold.
		hardLocked := diagnostics.SessionPolicyApplied && policy.HardLocked() &&
			diagnostics.SelectedModel == diagnostics.PreviousModel
		if hardLocked {
			diagnostics.HardLockReason = policy.HardLockReason()
		}
		if policy.Details.ProtectionTrace() != nil {
			diagnostics.SessionAction = replaySessionAction(diagnostics, hardLocked)
		}
		if policy.Mode == config.DecisionAdaptationModeObserve {
			diagnostics.DecisionReason = "observe_only"
		}
		diagnostics.SessionReason = replaySessionReason(diagnostics, policy)
		return diagnostics
	}

	return diagnostics
}

func buildReplayLearningDiagnostics(ctx *RequestContext) *routerreplay.LearningDiagnostics {
	policies := learningPoliciesForReplay(ctx)
	if policies.Empty() && (ctx == nil || ctx.VSRLearningProtectionPreflight == nil) {
		return nil
	}
	diagnostics := &routerreplay.LearningDiagnostics{}
	if ctx != nil && ctx.VSRLearningProtectionPreflight != nil {
		diagnostics.ProtectionPreflight = ctx.VSRLearningProtectionPreflight
	}
	if policy, ok := policies.Policy(routerLearningMethodAdaptation); ok {
		diagnostics.Adaptation = policy.toReplayAdaptation()
	}
	if policy, ok := policies.Policy(routerLearningMethodProtection); ok {
		diagnostics.Protection = policy.toReplayProtection()
	}
	return diagnostics
}

func learningPoliciesForReplay(ctx *RequestContext) routerLearningPolicies {
	if ctx == nil {
		return routerLearningPolicies{}
	}
	if !ctx.VSRLearningPolicies.Empty() {
		return ctx.VSRLearningPolicies
	}
	if ctx.VSRLearningPolicy == nil || ctx.VSRLearningPolicy.Empty() {
		return routerLearningPolicies{}
	}
	method := ctx.VSRLearningPolicy.Method
	if method == "" {
		method = routerLearningMethodProtection
	}
	policies := routerLearningPolicies{}
	policy := *ctx.VSRLearningPolicy
	policy.Method = method
	policies.Set(policy)
	return policies
}

func replaySessionAction(diagnostics *routerreplay.RouteDiagnostics, hardLocked bool) string {
	if hardLocked {
		return replaySessionActionHardLock
	}
	if diagnostics.PreviousModel == "" {
		return replaySessionActionEstablishBaseline
	}
	if diagnostics.SelectedModel == diagnostics.PreviousModel {
		return replaySessionActionStay
	}
	return replaySessionActionSwitch
}

func replaySessionReason(diagnostics *routerreplay.RouteDiagnostics, policy routerLearningPolicy) string {
	if policyReason := strings.TrimSpace(policy.Reason); policyReason != "" {
		return policyReason
	}
	if diagnostics.SessionAction == replaySessionActionHardLock && diagnostics.HardLockReason != "" {
		return diagnostics.HardLockReason
	}
	if diagnostics.DecisionReason != "" {
		return diagnostics.DecisionReason
	}
	return diagnostics.SessionAction
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func sessionPolicyMapForReplay(ctx *RequestContext) map[string]interface{} {
	policy, ok := protectionLearningPolicyForContext(ctx)
	if !ok {
		return nil
	}
	return cloneReplayInterfaceMap(policy.ToMap())
}
