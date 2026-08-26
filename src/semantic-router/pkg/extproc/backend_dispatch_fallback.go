package extproc

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

type primaryDispatchCandidate struct {
	model    string
	priority int
}

// primaryDispatchCandidates projects the matched rule's one canonical
// assignment set into the ordered physical candidate chain. The routing
// algorithm has already selected one priority-zero Model; other active-tier
// Models are not silently converted into failover candidates. Each later
// priority tier contributes exactly one request-affine weighted Model, so only
// a priority increase is a cross-Model fallback transition.
func (r *OpenAIRouter) primaryDispatchCandidates(
	request *RequestContext,
	selectedModel string,
) ([]primaryDispatchCandidate, backendinvoker.FallbackPolicy, error) {
	selectedModel = strings.TrimSpace(selectedModel)
	if selectedModel == "" {
		return nil, backendinvoker.FallbackPolicy{}, fmt.Errorf("selected model is unavailable")
	}
	single := []primaryDispatchCandidate{{model: selectedModel, priority: 0}}
	if request == nil || request.VSRSelectedDecision == nil {
		return single, backendinvoker.FallbackPolicy{}, nil
	}
	rule := r.assignmentRuleForRequest(request)
	if rule == nil {
		return single, backendinvoker.FallbackPolicy{}, nil
	}
	assignmentSet, found := rule.Action.Assignments[request.VSRSelectedDecision.ID]
	if !found || assignmentSet.Fallback == nil {
		return single, backendinvoker.FallbackPolicy{}, nil
	}

	selectedAssigned := false
	candidates := make([]primaryDispatchCandidate, 0, len(assignmentSet.Models))
	candidates = append(candidates, primaryDispatchCandidate{model: selectedModel, priority: 0})
	fallbackTiers := make(map[int][]config.ModelRef)
	for _, assignment := range assignmentSet.Models {
		if assignment.Priority == 0 && assignment.ModelName == selectedModel {
			selectedAssigned = true
			continue
		}
		if assignment.Priority <= 0 {
			continue
		}
		weight, err := strconv.ParseFloat(assignment.Weight, 64)
		if err != nil {
			return nil, backendinvoker.FallbackPolicy{}, fmt.Errorf(
				"fallback assignment %q has invalid weight", assignment.ModelName,
			)
		}
		fallbackTiers[assignment.Priority] = append(
			fallbackTiers[assignment.Priority],
			config.ModelRef{Model: assignment.ModelName, LoRAName: assignment.LoRAName, Weight: weight},
		)
	}
	if !selectedAssigned {
		return nil, backendinvoker.FallbackPolicy{}, fmt.Errorf(
			"selected model %q is outside the active assignment tier", selectedModel,
		)
	}
	priorities := make([]int, 0, len(fallbackTiers))
	for priority := range fallbackTiers {
		priorities = append(priorities, priority)
	}
	sort.Ints(priorities)
	for _, priority := range priorities {
		tier := fallbackTiers[priority]
		sort.SliceStable(tier, func(left, right int) bool {
			if tier[left].Model != tier[right].Model {
				return tier[left].Model < tier[right].Model
			}
			return tier[left].LoRAName < tier[right].LoRAName
		})
		selected, _, ok := selection.WeightedModelRefForAffinity(
			tier,
			fallbackTierAffinityScope(request, request.VSRSelectedDecision.ID, priority),
		)
		if !ok || selected == nil {
			return nil, backendinvoker.FallbackPolicy{}, fmt.Errorf(
				"fallback priority %d requires valid weights and request affinity", priority,
			)
		}
		candidates = append(candidates, primaryDispatchCandidate{model: selected.Model, priority: priority})
	}
	fallback, err := compileDispatchFallback(assignmentSet.Fallback)
	if err != nil {
		return nil, backendinvoker.FallbackPolicy{}, err
	}
	return candidates, fallback, nil
}

func fallbackTierAffinityScope(request *RequestContext, decisionID string, priority int) string {
	if request == nil || strings.TrimSpace(request.RequestID) == "" {
		return ""
	}
	return fmt.Sprintf(
		"%s\x00%s\x00fallback:%d\x00%s",
		request.Routing.RuntimeScope(),
		decisionID,
		priority,
		request.RequestID,
	)
}

// assignmentRuleForRequest returns the already-authorized rule when access is
// enabled. Durable routing without native access resolve the same public
// entrypoint contract with no identity claims; this keeps fallback a Router
// capability rather than a Dashboard or access-service dependency.
func (r *OpenAIRouter) assignmentRuleForRequest(request *RequestContext) *config.EntrypointRule {
	if request == nil {
		return nil
	}
	if request.InferenceAccess != nil {
		state := request.InferenceAccess
		state.mu.Lock()
		rule := state.rule
		state.mu.Unlock()
		return rule
	}
	if r == nil || r.Config == nil {
		return nil
	}
	resolution, err := r.Config.ResolveEntrypoint(
		request.RequestModel,
		normalizedInferencePath(request),
		nil,
	)
	if err != nil || resolution.Outcome != config.EntrypointResolveMatched {
		return nil
	}
	return resolution.Rule
}

func compileDispatchFallback(policy *config.RoutingFallbackPolicy) (backendinvoker.FallbackPolicy, error) {
	if policy == nil || policy.Strategy != "priority" {
		return backendinvoker.FallbackPolicy{}, fmt.Errorf("fallback strategy is invalid")
	}
	result := backendinvoker.FallbackPolicy{On: make([]backendinvoker.FallbackTrigger, 0, len(policy.On))}
	for _, value := range policy.On {
		var trigger backendinvoker.FallbackTrigger
		switch value {
		case string(backendinvoker.FallbackUnavailable):
			trigger = backendinvoker.FallbackUnavailable
		case string(backendinvoker.FallbackTimeout):
			trigger = backendinvoker.FallbackTimeout
		default:
			return backendinvoker.FallbackPolicy{}, fmt.Errorf("fallback trigger %q is invalid", value)
		}
		result.On = append(result.On, trigger)
	}
	return result, nil
}
