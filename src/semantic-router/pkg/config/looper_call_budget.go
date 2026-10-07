package config

import (
	"fmt"
	"math"
)

// MaxUpstreamCallsPerRequest is the hard per-request amplification budget for
// Looper-family algorithms. Concurrency controls (max_concurrent) bound
// simultaneous load; this budget bounds the total upstream model calls one
// request may create through fan-out, retries, and nested stages.
//
// The value stays a small multiple of the largest maintained configuration so
// that a statically knowable call count fails configuration load with an
// actionable error instead of surprising a request at runtime. Making the
// budget operator-configurable is a follow-up; the invariant is the same.
const MaxUpstreamCallsPerRequest = 32

// EstimatedReMoMUpstreamCalls returns the statically knowable upstream call
// count for a ReMoM breadth schedule: every entry is one parallel round and
// ReMoM always appends one final synthesis round.
//
// The running sum saturates at math.MaxInt instead of overflowing, so an
// oversized but otherwise well-formed schedule still compares greater than
// MaxUpstreamCallsPerRequest rather than wrapping to a small or negative value.
// Entries are expected to be positive; validateReMoMBreadthSchedule rejects
// non-positive entries first.
func EstimatedReMoMUpstreamCalls(schedule []int) int {
	total := 1 // final synthesis round
	for _, breadth := range schedule {
		if breadth > math.MaxInt-total {
			return math.MaxInt
		}
		total += breadth
	}
	return total
}

// validateReMoMBreadthScheduleBudget rejects a breadth_schedule whose static
// upstream call count already exceeds MaxUpstreamCallsPerRequest. It runs after
// well-formedness checks so a malformed schedule reports its own error first.
func validateReMoMBreadthScheduleBudget(schedule []int) error {
	estimated := EstimatedReMoMUpstreamCalls(schedule)
	if estimated <= MaxUpstreamCallsPerRequest {
		return nil
	}
	return fmt.Errorf(
		"breadth_schedule %v requests %d upstream calls (%d parallel calls + 1 final synthesis), exceeding the per-request limit of %d; reduce the schedule sum to at most %d",
		schedule,
		estimated,
		estimated-1,
		MaxUpstreamCallsPerRequest,
		MaxUpstreamCallsPerRequest-1,
	)
}

// validateDecisionModelRefBudget rejects a decision whose candidate list alone
// exceeds the per-request amplification budget. It is a backstop for fan-out
// algorithms whose call count is not modeled separately: ratings and confidence
// dispatch roughly one call per candidate, while algorithms with extra
// mandatory stages (Fusion, ReMoM) are checked by their own estimators below.
func validateDecisionModelRefBudget(decisionName string, modelRefs []ModelRef) error {
	if len(modelRefs) <= MaxUpstreamCallsPerRequest {
		return nil
	}
	return fmt.Errorf(
		"decision '%s': modelRefs lists %d candidate models, exceeding the per-request upstream-call limit of %d; reduce the modelRefs list to at most %d",
		decisionName,
		len(modelRefs),
		MaxUpstreamCallsPerRequest,
		MaxUpstreamCallsPerRequest,
	)
}

// EstimatedFusionUpstreamCalls returns Fusion's statically knowable upstream
// call count: one panel call per effective analysis model plus the mandatory
// judge stages. The separate analysis mode runs an analysis stage before the
// final synthesis (two judge calls); one_call and none dispatch a single judge
// call. Panel selection (analysis_models overriding modelRefs) is resolved by
// the caller.
func EstimatedFusionUpstreamCalls(panelSize int, analysisMode string) int {
	judgeStages := 1
	if EffectiveFusionAnalysisMode(analysisMode) == FusionAnalysisModeSeparate {
		judgeStages = 2
	}
	return panelSize + judgeStages
}

// validateFusionCallBudget rejects a Fusion decision whose effective panel plus
// mandatory judge stages already exceed the per-request amplification budget.
// The call count is knowable from configuration before any retries.
func validateFusionCallBudget(decisionName string, panelSize int, analysisMode string) error {
	estimated := EstimatedFusionUpstreamCalls(panelSize, analysisMode)
	if estimated <= MaxUpstreamCallsPerRequest {
		return nil
	}
	judgeStages := estimated - panelSize
	return fmt.Errorf(
		"decision '%s': fusion panel of %d analysis models plus %d judge stage(s) requests %d upstream calls, exceeding the per-request limit of %d; reduce analysis_models or modelRefs to at most %d",
		decisionName,
		panelSize,
		judgeStages,
		estimated,
		MaxUpstreamCallsPerRequest,
		MaxUpstreamCallsPerRequest-judgeStages,
	)
}
