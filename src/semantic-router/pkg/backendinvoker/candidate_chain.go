package backendinvoker

import (
	"fmt"
	"strings"
)

// DispatchCandidate is one immutable Model dispatch in a signed fallback
// chain. Ordinal is the request-wide dispatch ordinal; Priority is the
// contiguous route tier beginning at zero.
type DispatchCandidate struct {
	DispatchID         string `json:"dispatchId"`
	DispatchType       string `json:"dispatchType"`
	Ordinal            int    `json:"ordinal"`
	DispatchPlanDigest string `json:"dispatchPlanDigest"`
	ModelID            string `json:"modelId"`
	ModelRevision      int64  `json:"modelRevision"`
	Priority           int    `json:"priority"`
}

func validateCandidateChain(candidates []DispatchCandidate, fallback FallbackPolicy) error {
	if len(candidates) == 0 || len(candidates) > maximumDispatchCandidates {
		return fmt.Errorf("dispatch candidate chain must contain between 1 and %d candidates", maximumDispatchCandidates)
	}
	if err := validateFallbackPolicy(fallback); err != nil {
		return err
	}
	seenDispatches := make(map[string]struct{}, len(candidates))
	seenModels := make(map[string]struct{}, len(candidates))
	firstOrdinal := candidates[0].Ordinal
	previousPriority := -1
	for index, candidate := range candidates {
		if !validBoundedIdentity(candidate.DispatchID) ||
			!validBoundedIdentity(candidate.DispatchType) ||
			candidate.Ordinal < 0 || candidate.Ordinal != firstOrdinal+index ||
			!validSHA256Hex(candidate.DispatchPlanDigest) ||
			!validBoundedIdentity(candidate.ModelID) || candidate.ModelRevision <= 0 ||
			candidate.Priority < 0 || candidate.Priority > 31 {
			return fmt.Errorf("dispatch candidate %d is incomplete or non-canonical", index)
		}
		if index == 0 && candidate.Priority != 0 {
			return fmt.Errorf("dispatch candidate priority must begin at zero")
		}
		if previousPriority > candidate.Priority || candidate.Priority > previousPriority+1 {
			return fmt.Errorf("dispatch candidate priorities must be ordered and contiguous")
		}
		previousPriority = candidate.Priority
		if _, duplicate := seenDispatches[candidate.DispatchID]; duplicate {
			return fmt.Errorf("dispatch candidate %q is duplicated", candidate.DispatchID)
		}
		seenDispatches[candidate.DispatchID] = struct{}{}
		modelKey := candidate.ModelID + "\x00" + fmt.Sprintf("%d", candidate.ModelRevision)
		if _, duplicate := seenModels[modelKey]; duplicate {
			return fmt.Errorf("model candidate %q revision %d is duplicated", candidate.ModelID, candidate.ModelRevision)
		}
		seenModels[modelKey] = struct{}{}
	}
	return nil
}

func validateFallbackPolicy(policy FallbackPolicy) error {
	previous := -1
	for _, trigger := range policy.On {
		order := fallbackTriggerOrder(trigger)
		if order < 0 {
			return fmt.Errorf("unsupported fallback trigger %q", trigger)
		}
		if order <= previous {
			return fmt.Errorf("fallback triggers must be unique and in canonical order")
		}
		previous = order
	}
	return nil
}

func fallbackTriggerOrder(trigger FallbackTrigger) int {
	switch trigger {
	case FallbackUnavailable:
		return 0
	case FallbackOverloaded:
		return 1
	case FallbackTimeout:
		return 2
	default:
		return -1
	}
}

func fallbackEnabled(policy FallbackPolicy, trigger FallbackTrigger) bool {
	for _, configured := range policy.On {
		if configured == trigger {
			return true
		}
	}
	return false
}

func candidateFromPlan(plan Plan) DispatchCandidate {
	return DispatchCandidate{
		DispatchID: plan.DispatchID, DispatchType: plan.DispatchType,
		Ordinal: plan.Ordinal, DispatchPlanDigest: plan.DispatchPlanDigest,
		ModelID: plan.ModelID, ModelRevision: plan.ModelRevision, Priority: plan.Priority,
	}
}

func sameCandidate(left, right DispatchCandidate) bool {
	return left.DispatchID == right.DispatchID && left.DispatchType == right.DispatchType &&
		left.Ordinal == right.Ordinal && left.DispatchPlanDigest == right.DispatchPlanDigest &&
		left.ModelID == right.ModelID && left.ModelRevision == right.ModelRevision &&
		left.Priority == right.Priority
}

func boundedOptionalIdentity(value string, maximum int) bool {
	return len(value) <= maximum && value == strings.TrimSpace(value) && !strings.ContainsRune(value, '\x00')
}
