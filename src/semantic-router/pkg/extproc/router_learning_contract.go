package extproc

import (
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const (
	routerLearningMethodSessionAware routerLearningMethod = "session_aware"

	routerLearningActionNoop     routerLearningAction = "noop"
	routerLearningActionBypass   routerLearningAction = "bypass"
	routerLearningActionHardLock routerLearningAction = "hard_lock"
	routerLearningActionSelect   routerLearningAction = "select"
	routerLearningActionStay     routerLearningAction = "stay"
	routerLearningActionSwitch   routerLearningAction = "switch"

	routerLearningReasonNoopResult = "no_change"
)

type routerLearningMethod string

type routerLearningAction string

type routerLearningInput struct {
	selCtx           *selection.SelectionContext
	baseResult       *selection.SelectionResult
	selectedModelRef *config.ModelRef
	ctx              *RequestContext
	experience       routerLearningExperienceSnapshot
}

type routerLearningAdaptationResult struct {
	method           routerLearningMethod
	mode             string
	scope            string
	action           routerLearningAction
	reason           string
	hard             bool
	changesModel     bool
	selectionContext *selection.SelectionContext
	selectionResult  *selection.SelectionResult
	selectedModelRef *config.ModelRef
	policy           routerLearningPolicy
}

type routerLearningComposition struct {
	selectionContext *selection.SelectionContext
	selectionResult  *selection.SelectionResult
	selectedModelRef *config.ModelRef
	applied          bool
}

func composeRouterLearning(
	input routerLearningInput,
	results []routerLearningAdaptationResult,
) routerLearningComposition {
	composed := routerLearningComposition{
		selectionContext: input.selCtx,
		selectionResult:  input.baseResult,
		selectedModelRef: input.selectedModelRef,
	}
	if input.ctx == nil || len(results) == 0 {
		return composed
	}

	recordRouterLearningAdaptationPolicies(input.ctx, results)

	final := chooseRouterLearningFinalResult(results)
	if final == nil || !final.changesModel {
		if observed := firstObservedRouterLearningResult(results); observed != nil {
			composed.selectionContext = firstNonNilSelectionContext(observed.selectionContext, input.selCtx)
		}
		composed.applied = hasObservedRouterLearningResult(results)
		return composed
	}

	composed.selectionContext = firstNonNilSelectionContext(final.selectionContext, input.selCtx)
	composed.selectionResult = firstNonNilSelectionResult(final.selectionResult, input.baseResult)
	composed.selectedModelRef = firstNonNilModelRef(final.selectedModelRef, input.selectedModelRef)
	composed.applied = true
	return composed
}

func chooseRouterLearningFinalResult(results []routerLearningAdaptationResult) *routerLearningAdaptationResult {
	for i := range results {
		result := &results[i]
		if result.hard && result.changesModel && result.mode != config.DecisionAdaptationModeObserve {
			return result
		}
	}
	for i := range results {
		result := &results[i]
		if result.changesModel && result.mode != config.DecisionAdaptationModeObserve {
			return result
		}
	}
	return nil
}

func firstObservedRouterLearningResult(results []routerLearningAdaptationResult) *routerLearningAdaptationResult {
	for i := range results {
		result := &results[i]
		if result.mode == config.DecisionAdaptationModeObserve && result.selectionResult != nil {
			return result
		}
	}
	return nil
}

func hasObservedRouterLearningResult(results []routerLearningAdaptationResult) bool {
	for _, result := range results {
		if result.mode == config.DecisionAdaptationModeObserve && result.selectionResult != nil {
			return true
		}
	}
	return false
}

func recordRouterLearningAdaptationPolicies(ctx *RequestContext, results []routerLearningAdaptationResult) {
	if ctx == nil {
		return
	}
	if ctx.VSRLearningPolicies == nil {
		ctx.VSRLearningPolicies = map[routerLearningMethod]routerLearningPolicy{}
	}
	for _, result := range results {
		method := result.method
		if method == "" {
			continue
		}
		policy := result.policy
		policy.Adaptation = firstNonEmptyLearningMethod(policy.Adaptation, method)
		if strings.TrimSpace(result.mode) != "" {
			policy.Mode = result.mode
		}
		if strings.TrimSpace(result.scope) != "" {
			policy.Scope = result.scope
		}
		if result.action != "" {
			policy.Action = result.action
		}
		if strings.TrimSpace(result.reason) != "" {
			policy.Reason = result.reason
		}
		ctx.VSRLearningPolicies[method] = policy
	}
	ctx.VSRLearningPolicy = primaryRouterLearningPolicy(ctx.VSRLearningPolicies)
	if policy, ok := ctx.VSRLearningPolicies[routerLearningMethodSessionAware]; ok {
		ctx.VSRSessionPolicy = policy.ToMap()
	}
}

func primaryRouterLearningPolicy(policies map[routerLearningMethod]routerLearningPolicy) *routerLearningPolicy {
	if len(policies) == 0 {
		return nil
	}
	if policy, ok := policies[routerLearningMethodSessionAware]; ok {
		return &policy
	}
	methods := sortedRouterLearningPolicyMethods(policies)
	if len(methods) == 0 {
		return nil
	}
	policy := policies[methods[0]]
	return &policy
}

func sortedRouterLearningPolicyMethods(policies map[routerLearningMethod]routerLearningPolicy) []routerLearningMethod {
	methods := make([]routerLearningMethod, 0, len(policies))
	for method := range policies {
		if strings.TrimSpace(string(method)) != "" {
			methods = append(methods, method)
		}
	}
	sort.Slice(methods, func(i, j int) bool {
		return methods[i] < methods[j]
	})
	return methods
}

func firstNonEmptyLearningMethod(values ...routerLearningMethod) routerLearningMethod {
	for _, value := range values {
		if strings.TrimSpace(string(value)) != "" {
			return value
		}
	}
	return ""
}

func firstNonNilSelectionContext(values ...*selection.SelectionContext) *selection.SelectionContext {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstNonNilSelectionResult(values ...*selection.SelectionResult) *selection.SelectionResult {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstNonNilModelRef(values ...*config.ModelRef) *config.ModelRef {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func learningChangesModel(baseResult *selection.SelectionResult, result *selection.SelectionResult) bool {
	if baseResult == nil || result == nil {
		return false
	}
	return strings.TrimSpace(baseResult.SelectedModel) != "" &&
		strings.TrimSpace(result.SelectedModel) != "" &&
		baseResult.SelectedModel != result.SelectedModel
}
